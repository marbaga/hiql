import os
import time
import pickle
import tqdm
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories
from src import d4rl_ant, viz_utils
from src.agents import hiql as learner
from src.gc_dataset import GCSDataset
from src.utils import Logger
from src.envs import make_env, get_example_trajectory
from cluster_utils import cluster_main, announce_fraction_finished
import aim


@jax.jit
def get_debug_statistics(agent, batch):
    return {'v': agent.network(batch['observations'], batch['goals'], info=True, method='value')['v'].mean()}


@jax.jit
def get_gcvalue(agent, s, g):
    v1, v2 = agent.network(s, g, method='value')
    return (v1 + v2) / 2


def get_v(agent, goal, observations):
    return get_gcvalue(agent, observations, jnp.tile(goal, (observations.shape[0], 1)))


@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v1, v2 = agent.network(jax.tree_map(lambda x: x[None], s), jax.tree_map(lambda x: x[None], g), method='value')
        return (v1 + v2) / 2
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }


def update_kwargs_and_freeze(kwargs):
    kwargs = ConfigDict(kwargs)
    for k in ['p_randomgoal', 'p_trajgoal', 'p_currgoal', 'geom_sample', 'high_p_randomgoal', 'way_steps', 'discount']:
        kwargs.gcdataset[k] = kwargs[k]
    for k in ['pretrain_expectile', 'discount', 'temperature', 'high_temperature', 'use_waypoints', 'way_steps',
              'use_rep', 'rep_dim', 'policy_train_rep', 'hl_sorb', 'll_sorb']:
        kwargs.config[k] = kwargs[k]
    kwargs.config['value_hidden_dims'] = (kwargs.value_hidden_dim,) * kwargs.value_num_layers
    return FrozenConfigDict(kwargs)


@cluster_main
def main(**kwargs):
    kwargs = update_kwargs_and_freeze(kwargs)
    env, dataset, goal_infos, discrete, viz, viz_env, viz_dataset = make_env(kwargs)
    env.reset()

    pretrain_dataset = GCSDataset(dataset, **kwargs.gcdataset)
    example_batch = dataset.sample(1)
    agent = learner.create_learner(kwargs.seed, example_batch['observations'],
        example_batch['actions'] if not discrete else np.max(dataset['actions'], keepdims=True),
        visual=kwargs.visual, encoder=kwargs.encoder, discrete=discrete,
        use_layer_norm=kwargs.use_layer_norm, rep_type=kwargs.rep_type, **kwargs.config)
    example_trajectory =  get_example_trajectory(pretrain_dataset, kwargs.env_name)

    logger = Logger(kwargs)
    start_time, last_eval_time = time.time(), time.time()
    for i in tqdm.tqdm(range(1, kwargs.pretrain_steps + 1), smoothing=0.1, dynamic_ncols=True):
        
        pretrain_batch = pretrain_dataset.sample(kwargs.batch_size)
        agent, update_info = supply_rng(agent.pretrain_update)(pretrain_batch)

        if i % kwargs.log_interval == 0:
            logger.log(
            {
                **{f'training/{k}': v for k, v in update_info.items()}, 
                **{f'pretraining/debug/{k}': v for k, v in get_debug_statistics(agent, pretrain_batch).items()},
                'time/epoch_time': (time.time() - last_eval_time) / kwargs.log_interval,
                'time/total_time': (time.time() - start_time)
            }, step=i, mode='train')
            last_eval_time = time.time()

        if i == 1 or i % kwargs.eval_interval == 0:
            policy_fn = partial(supply_rng(agent.sample_actions), discrete=discrete)
            high_policy_fn = partial(supply_rng(agent.sample_high_actions))
            value_fn = agent.value
            policy_rep_fn = agent.get_policy_rep
            base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
            if 'procgen' in kwargs.env_name:
                eval_metrics = {}
                for goal_info in goal_infos:
                    eval_info, trajs, _ = evaluate_with_trajectories(
                        policy_fn, high_policy_fn, policy_rep_fn, value_fn, pretrain_dataset,
                        env, env_name=kwargs.env_name, num_episodes=kwargs.eval_episodes,
                        base_observation=base_observation, num_video_episodes=0,
                        use_waypoints=kwargs.use_waypoints,
                        eval_temperature=0, epsilon=0.05,
                        goal_info=goal_info, config=kwargs.config,
                    )
                    eval_metrics.update({f'evaluation/level{goal_info["eval_level_name"]}_{k}': v for k, v in eval_info.items()})
            else:
                eval_info, trajs, _ = evaluate_with_trajectories(
                    policy_fn, high_policy_fn, policy_rep_fn, value_fn, pretrain_dataset,
                    env, env_name=kwargs.env_name, num_episodes=kwargs.eval_episodes,
                    base_observation=base_observation, num_video_episodes=kwargs.num_video_episodes,
                    use_waypoints=kwargs.use_waypoints,
                    eval_temperature=0,
                    goal_info=None, config=kwargs.config,
                )
                eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}

                # if kwargs.num_video_episodes > 0:
                #     video = record_video('Video', i, renders=renders)
                #     eval_metrics['video'] = video

            traj_metrics = get_traj_v(agent, example_trajectory)
            value_viz = viz_utils.make_visual_no_image(traj_metrics)
            eval_metrics['value_traj_viz'] = aim.Image(value_viz)

            if kwargs.env_name.startswith('antmaze') and 'large' in kwargs.env_name:
                traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                eval_metrics['trajectories'] = aim.Image(traj_image)
                new_metrics_dist = viz.get_distance_metrics(trajs)
                eval_metrics.update({f'debugging/{k}': v for k, v in new_metrics_dist.items()})
                image_v = d4rl_ant.gcvalue_image(viz_env, viz_dataset, partial(get_v, agent))
                eval_metrics['v'] = aim.Image(image_v)

            logger.log(eval_metrics, step=i, mode='eval')
            announce_fraction_finished(i/kwargs.pretrain_steps)

        if i % kwargs.save_interval == 0:
            fname = os.path.join(kwargs.working_dir, f'params_{i}.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump({
                    'agent': flax.serialization.to_state_dict(agent),
                    'config': kwargs.config.to_dict()
                }, f)

    logger.close()
    return {k: v for k, v in eval_metrics.items() if not isinstance(v, (aim.Image, aim.Distribution))}


if __name__ == '__main__':
    main()
