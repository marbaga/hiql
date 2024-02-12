import numpy as np
import gzip
from src import d4rl_utils, d4rl_ant, ant_diagnostics
from jaxrl_m.evaluation import EpisodeMonitor
import pickle


def make_env(kwargs):
    viz, viz_env, viz_dataset = None, None, None
    goal_infos = None
    if 'antmaze' in kwargs.env_name:
        if 'ultra' in kwargs.env_name:
            import d4rl_ext
            import gym
            env = EpisodeMonitor(gym.make(kwargs.env_name))
        else:
            env = d4rl_utils.make_env(kwargs.env_name)
        dataset = d4rl_utils.get_dataset(env, kwargs.env_name)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})
        env.render(mode='rgb_array', width=200, height=200)
        if 'large' in kwargs.env_name:
            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(kwargs.env_name)
            viz = ant_diagnostics.Visualizer(kwargs.env_name, viz_env, viz_dataset, discount=kwargs.discount)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
        env.viewer.cam.lookat[0] = 26 if 'ultra' in kwargs.env_name else 18
        env.viewer.cam.lookat[1] = 28 if 'ultra' in kwargs.env_name else 12
        env.viewer.cam.distance = 70 if 'ultra' in kwargs.env_name else 50
        env.viewer.cam.elevation = -90

    elif 'kitchen' in kwargs.env_name:
        env = d4rl_utils.make_env(kwargs.env_name)
        dataset = d4rl_utils.get_dataset(env, kwargs.env_name, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30],
                                'next_observations': dataset['next_observations'][:, :30]})

    elif 'calvin' in kwargs.env_name:
        from src.envs.calvin import CalvinEnv
        from hydra import compose, initialize
        from src.envs.gym_env import GymWrapper
        from src.envs.gym_env import wrap_env
        initialize(config_path='src/envs/conf')
        cfg = compose(config_name='calvin')
        env = CalvinEnv(**cfg)
        env.max_episode_steps = cfg.max_episode_steps = 360
        env = GymWrapper(
            env=env,
            from_pixels=cfg.pixel_ob,
            from_state=cfg.state_ob,
            height=cfg.screen_size[0],
            width=cfg.screen_size[1],
            channels_first=False,
            frame_skip=cfg.action_repeat,
            return_state=False,
        )
        env = wrap_env(env, cfg)
        data = pickle.load(gzip.open('data/calvin.gz', "rb"))
        ds = []
        for i, d in enumerate(data):
            if len(d['obs']) < len(d['dones']):
                continue  # Skip incomplete trajectories.
            # Only use the first 21 states of non-floating objects.
            d['obs'] = d['obs'][:, :21]
            new_d = dict(
                observations=d['obs'][:-1],
                next_observations=d['obs'][1:],
                actions=d['actions'][:-1],
            )
            num_steps = new_d['observations'].shape[0]
            new_d['rewards'] = np.zeros(num_steps)
            new_d['terminals'] = np.zeros(num_steps, dtype=bool)
            new_d['terminals'][-1] = True
            ds.append(new_d)
        dataset = dict()
        for key in ds[0].keys():
            dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
        dataset = d4rl_utils.get_dataset(None, kwargs.env_name, dataset=dataset)

    elif 'procgen' in kwargs.env_name:
        from src.envs.procgen_env import ProcgenWrappedEnv, get_procgen_dataset
        import matplotlib
        matplotlib.use('Agg')
        n_processes = 1
        env_name = 'maze'
        env = ProcgenWrappedEnv(n_processes, env_name, 1, 1)
        if kwargs.env_name == 'procgen-500':
            dataset = get_procgen_dataset('data/procgen/level500.npz',
                                          state_based=('state' in kwargs.env_name))
            min_level, max_level = 0, 499
        elif kwargs.env_name == 'procgen-1000':
            dataset = get_procgen_dataset('data/procgen/level1000.npz',
                                          state_based=('state' in kwargs.env_name))
            min_level, max_level = 0, 999
        else:
            raise NotImplementedError
        # Test on large levels having >=20 border states
        large_levels = [12, 34, 35, 55, 96, 109, 129, 140, 143, 163, 176, 204, 234, 338, 344, 369, 370, 374, 410, 430, 468, 470, 476, 491] + \
                       [5034, 5046, 5052, 5080, 5082, 5142, 5244, 5245, 5268, 5272, 5283, 5335, 5342, 5366, 5375, 5413, 5430, 5474, 5491]
        goal_infos = [
            {'eval_level': [level for level in large_levels if min_level <= level <= max_level], 'eval_level_name': 'train'},
            {'eval_level': [level for level in large_levels if level > max_level], 'eval_level_name': 'test'}
        ]
        dones_float = 1.0 - dataset['masks']
        dones_float[-1] = 1.0
        dataset = dataset.copy({'dones_float': dones_float})
    else:
        raise NotImplementedError

    return env, dataset, goal_infos, 'procgen' in kwargs.env_name, viz, viz_env, viz_dataset


def get_example_trajectory(pretrain_dataset, env_name):
    # For debugging metrics
    start_idx = {'antmaze': 1000, 'kitchen': 0, 'calvin': 0, 'procgen': 5000}
    for k in start_idx:
        if k in env_name:
            return pretrain_dataset.sample(50, indx=np.arange(start_idx[k], start_idx[k]+50))
    raise NotImplementedError
