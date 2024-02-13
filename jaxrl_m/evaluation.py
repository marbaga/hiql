from typing import Dict
import jax
import gym
import numpy as np
from collections import defaultdict
import time
import jax.numpy as jnp


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def evaluate_with_trajectories(
        policy_fn, high_policy_fn, policy_rep_fn, value_fn, pretrain_dataset, env: gym.Env, env_name, num_episodes: int, base_observation=None, num_video_episodes=0,
        use_waypoints=False, eval_temperature=0, epsilon=0, goal_info=None,
        config=None,
) -> Dict[str, float]:
    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in range(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)

        if 'procgen' in env_name:
            from src.envs.procgen_env import ProcgenWrappedEnv
            from src.envs.procgen_viz import ProcgenLevel
            eval_level = goal_info['eval_level']
            cur_level = eval_level[np.random.choice(len(eval_level))]

            level_details = ProcgenLevel.create(cur_level)
            border_states = [i for i in range(len(level_details.locs)) if len([1 for j in range(len(level_details.locs)) if abs(level_details.locs[i][0] - level_details.locs[j][0]) + abs(level_details.locs[i][1] - level_details.locs[j][1]) < 7]) <= 2]
            target_state = border_states[np.random.choice(len(border_states))]
            goal_img = level_details.imgs[target_state]
            goal_loc = level_details.locs[target_state]
            env = ProcgenWrappedEnv(1, 'maze', cur_level, 1)

        observation, done = env.reset(), False

        # Set goal
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
        elif 'kitchen' in env_name:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
        elif 'calvin' in env_name:
            observation = observation['ob']
            goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
            obs_goal = base_observation.copy()
            obs_goal[15:21] = goal
        elif 'procgen' in env_name:
            from src.envs.procgen_viz import get_xy_single
            observation = observation[0]
            obs_goal = goal_img
        else:
            raise NotImplementedError

        if config['hl_sorb']:

            if 'antmaze' in env_name:
                _sample = pretrain_dataset.sample(1000)['observations']
                _sample[0] = observation.copy()
                _sample[-1] = obs_goal.copy()
                _sample[:, 2:] = base_observation.copy()[2:]
                _bs = 10
                distances = []
                for i in range(_sample.shape[0]//_bs):
                    _obs = jnp.repeat(_sample[i*_bs:(i+1)*_bs], _sample.shape[0], axis=0)
                    _goals = jnp.tile(_sample, (_bs, 1))
                    distance = - sum(value_fn(observations=_obs, goals=_goals)) / 2
                    # TODO: remove line below
                    distance = jnp.sqrt(((_obs[:, :2] - _goals[:, :2])**2).sum(-1))
                    distances.append(distance)
                distances = jnp.concatenate(distances, 0).reshape(_sample.shape[0], -1)

                import networkx as nx
                from heapq import heappop, heappush
                from itertools import count
                import numpy as np
                def maximum_edge_length(G, source):
                    dist, seen = {}, {source: 0}
                    c = count()
                    fringe = [(0, next(c), source)]
                    while fringe:
                        (d, _, v) = heappop(fringe)
                        if v in dist: continue  # already searched this node.
                        dist[v] = d
                        for u, e in G._adj[v].items():
                            if e.get('weight') is None: continue
                            vu_dist = max(dist[v], e.get('weight'))
                            if u in dist and dist[u] > vu_dist:
                                raise ValueError("Contradictory paths found:", "negative weights?")
                            if u not in seen or vu_dist < seen[u]:
                                seen[u] = vu_dist
                                heappush(fringe, (vu_dist, next(c), u))
                    return dist

                distances = np.asarray(distances)  # huge speedup somehow
                # density = np.exp(-(np.log(-np.stack(distances, 0)) / np.log(0.99)).clip(None, 500) / 20.0).mean(0)
                # p = 1/density
                # density_p = p/p.sum()
                graph = nx.from_numpy_array(distances, parallel_edges=False, create_using=nx.DiGraph)
                edge_weights = nx.get_edge_attributes(graph,'weight')
                goal_id = len(distances) - 1
                lengths = maximum_edge_length(graph.reverse(), goal_id)
                graph.remove_edges_from((e for e, w in edge_weights.items() if w > lengths[0]))
                distances_on_graph = nx.shortest_path_length(graph, target=goal_id, weight='weight')
                shortest_paths = nx.shortest_path(graph, target=goal_id, weight='weight')
                sample_to_goal = np.asarray([(distances_on_graph[k] if k in distances_on_graph else 1000) for k in range(len(_sample))])
                sample_subgoals = np.asarray([shortest_paths[k][min(2, len(shortest_paths[k])-1)] if k in shortest_paths else 999 for k in range(len(_sample))])
            else:
                raise NotImplementedError

        render = []
        step = 0
        while not done:
            if not use_waypoints:
                cur_obs_goal_rep = policy_rep_fn(targets=obs_goal, bases=observation) if config['use_rep'] else obs_goal
            else:
                cur_obs_goal = high_policy_fn(observations=observation, goals=obs_goal, temperature=eval_temperature)
                if config['use_rep']:
                    cur_obs_goal_rep = cur_obs_goal / np.linalg.norm(cur_obs_goal, axis=-1, keepdims=True) * np.sqrt(cur_obs_goal.shape[-1])
                else:
                    cur_obs_goal_rep = observation + cur_obs_goal

            if config['hl_sorb']:
                dist_to_sample = - sum(value_fn(observations=jnp.tile(observation, (_sample.shape[0], 1)), goals=_sample)) / 2
                idx = jnp.argmin(dist_to_sample+sample_to_goal).item()
                obs_goal = _sample[sample_subgoals[idx]].copy()
                cur_obs_goal_rep = policy_rep_fn(targets=obs_goal, bases=observation) if config['use_rep'] else obs_goal

            action = policy_fn(observations=observation, goals=cur_obs_goal_rep, low_dim_goals=True, temperature=eval_temperature)
            if 'antmaze' in env_name:
                next_observation, r, done, info = env.step(action)
            elif 'kitchen' in env_name:
                next_observation, r, done, info = env.step(action)
                next_observation = next_observation[:30]
            elif 'calvin' in env_name:
                next_observation, r, done, info = env.step({'ac': np.array(action)})
                next_observation = next_observation['ob']
                del info['robot_info']
                del info['scene_info']
            elif 'procgen' in env_name:
                if np.random.random() < epsilon:
                    action = np.random.choice([2, 3, 5, 6])

                next_observation, r, done, info = env.step(np.array([action]))
                next_observation = next_observation[0]
                r = 0.
                done = done[0]
                info = dict()

                loc = get_xy_single(next_observation)
                if np.linalg.norm(loc - goal_loc) < 4:
                    r = 1.
                    done = True

                cur_render = next_observation

            step += 1

            # Render
            if 'procgen' in env_name:
                cur_frame = cur_render.transpose(2, 0, 1).copy()
                cur_frame[2, goal_loc[1]-1:goal_loc[1]+2, goal_loc[0]-1:goal_loc[0]+2] = 255
                cur_frame[:2, goal_loc[1]-1:goal_loc[1]+2, goal_loc[0]-1:goal_loc[0]+2] = 0
                render.append(cur_frame)
            else:
                if i >= num_episodes and step % 3 == 0:
                    if 'antmaze' in env_name:
                        size = 200
                        cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                        if use_waypoints and not config['use_rep'] and ('large' in env_name or 'ultra' in env_name):
                            def xy_to_pixxy(x, y):
                                if 'large' in env_name:
                                    pixx = (x / 36) * (0.93 - 0.07) + 0.07
                                    pixy = (y / 24) * (0.21 - 0.79) + 0.79
                                elif 'ultra' in env_name:
                                    pixx = (x / 52) * (0.955 - 0.05) + 0.05
                                    pixy = (y / 36) * (0.19 - 0.81) + 0.81
                                return pixx, pixy
                            x, y = cur_obs_goal_rep[:2]
                            pixx, pixy = xy_to_pixxy(x, y)
                            cur_frame[0, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 255
                            cur_frame[1:3, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 0
                        render.append(cur_frame)
                    elif 'kitchen' in env_name:
                        render.append(kitchen_render(env, wh=200).transpose(2, 0, 1))
                    elif 'calvin' in env_name:
                        cur_frame = env.render(mode='rgb_array').transpose(2, 0, 1)
                        render.append(cur_frame)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        if 'calvin' in env_name:
            info['return'] = sum(trajectory['reward'])
        elif 'procgen' in env_name:
            info['return'] = sum(trajectory['reward'])
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        if i >= num_episodes:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories, renders


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
