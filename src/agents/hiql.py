import copy
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.typing import *
from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy, DiscretePolicy
import flax
from flax.core import freeze, unfreeze
from src.special_networks import HierarchicalActorCritic, RelativeRepresentation, MonolithicVF


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_actor_loss(agent, batch, network_params):
    # Use waypoint states as goals (for hierarchical policies)
    cur_goals = batch['low_goals' if agent.config['use_waypoints'] else 'high_goals']
    v = sum(agent.network(batch['observations'], cur_goals, method='value')) / 2
    nv = sum(agent.network(batch['next_observations'], cur_goals, method='value')) / 2
    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)
    goal_rep_grad = agent.config['policy_train_rep'] if agent.config['use_waypoints'] else True
    dist = agent.network(batch['observations'], cur_goals, state_rep_grad=True, goal_rep_grad=goal_rep_grad, method='actor', params=network_params)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()
    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jnp.median(adv),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
    }


def compute_high_actor_loss(agent, batch, network_params):
    v = sum(agent.network(batch['observations'], batch['high_goals'], method='value')) / 2
    nv = sum(agent.network(batch['next_observations'], batch['high_goals'], method='value')) / 2
    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['high_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)
    dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=network_params)
    if agent.config['use_rep']:
        target = agent.network(targets=batch['high_targets'], bases=batch['observations'], method='value_goal_encoder')
    else:
        target = batch['high_targets'] - batch['observations']
    log_probs = dist.log_prob(target)
    actor_loss = -(exp_a * log_probs).mean()
    return actor_loss, {
        'high_actor_loss': actor_loss,
        'high_adv': adv.mean(),
        'high_bc_log_probs': log_probs.mean(),
        'high_adv_median': jnp.median(adv),
        'high_mse': jnp.mean((dist.mode() - target)**2),
        'high_scale': dist.scale_diag.mean(),
    }


def compute_value_loss(agent, batch, network_params):
    batch['masks'] = 1.0 - batch['rewards']  # masks are 0 if terminal, 1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0  # rewards are 0 if terminal, -1 otherwise
    (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v
    v_t = sum(agent.network(batch['observations'], batch['goals'], method='target_value')) / 2
    adv = q - v_t
    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network(batch['observations'], batch['goals'], method='value', params=network_params)
    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    value_loss = value_loss1 + value_loss2
    return value_loss, {
        'value_loss': value_loss,
        'v max': v1.max(),
        'v min': v1.min(),
        'v mean': v1.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }


class JointTrainAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def pretrain_update(agent, pretrain_batch, seed=None, value_update=True, actor_update=True, high_actor_update=True):
        
        def loss_fn(network_params):
            info = {}
            value_loss, actor_loss, high_actor_loss = 0., 0., 0.
            if value_update:
                value_loss, value_info = compute_value_loss(agent, pretrain_batch, network_params)
                info.update({f'value/{k}': v for k, v in value_info.items()})
            if actor_update:
                actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch, network_params)
                info.update({f'actor/{k}': v for k, v in actor_info.items()})
            if high_actor_update and agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, network_params)
                info.update({f'high_actor/{k}': v for k, v in high_actor_info.items()})
            return value_loss + actor_loss + high_actor_loss, info

        if value_update:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']),
                agent.network.params['networks_value'], agent.network.params['networks_target_value']
            )
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        if value_update:
            params = unfreeze(new_network.params)
            params['networks_target_value'] = new_target_params
            new_network = new_network.replace(params=freeze(params))
        return agent.replace(network=new_network), info
    pretrain_update = jax.jit(pretrain_update, static_argnames=('value_update', 'actor_update', 'high_actor_update'))

    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, low_dim_goals=low_dim_goals, temperature=temperature, method='actor')
        actions = dist.sample(seed=seed, sample_shape=() if num_samples is None else num_samples)
        if not discrete: actions = jnp.clip(actions, -1, 1)
        return actions
    sample_actions = jax.jit(sample_actions, static_argnames=('num_samples', 'low_dim_goals', 'discrete'))

    def sample_high_actions(agent,
                            observations: np.ndarray,
                            goals: np.ndarray,
                            *,
                            seed: PRNGKey,
                            temperature: float = 1.0,
                            num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, temperature=temperature, method='high_actor')
        return dist.sample(seed=seed, sample_shape=() if num_samples is None else num_samples)
    sample_high_actions = jax.jit(sample_high_actions, static_argnames=('num_samples',))

    @jax.jit
    def get_policy_rep(agent,
                       *,
                       targets: np.ndarray,
                       bases: np.ndarray = None,
                       ) -> jnp.ndarray:
        return agent.network(targets=targets, bases=bases, method='policy_goal_encoder')


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        high_temperature: float = 1,
        pretrain_expectile: float = 0.7,
        way_steps: int = 0,
        rep_dim: int = 10,
        use_rep: int = 0,
        policy_train_rep: float = 0,
        visual: int = 0,
        encoder: str = 'impala',
        discrete: int = 0,
        use_layer_norm: int = 0,
        rep_type: str = 'state',
        use_waypoints: int = 0,
        sorb_hl: bool = False,
        sorb_ll: bool = False,
        **kwargs):

        print('Extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng, 2)

        encoders = {k: None for k in ['value_state', 'value_goal', 'policy_state', 'policy_goal', 'high_policy_state', 'high_policy_goal']}
        if visual:
            assert use_rep
            from jaxrl_m.vision import encoders
            visual_encoder = encoders[encoder]
            def make_encoder(bottleneck):
                _rep_dim = rep_dim if bottleneck else value_hidden_dims[-1]
                return RelativeRepresentation(rep_dim=_rep_dim, hidden_dims=(_rep_dim,), visual=True, module=visual_encoder,
                                              layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=bottleneck)
            encoders = {k: make_encoder(bottleneck=use_waypoints if k=='value_goal' else False) for k in encoders}
        elif use_rep:
            def make_encoder(bottleneck):
                _rep_dim = rep_dim if bottleneck else value_hidden_dims[-1]
                return RelativeRepresentation(rep_dim=_rep_dim, hidden_dims=(*value_hidden_dims, _rep_dim),
                                              layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=bottleneck)
            encoders['value_goal'] = make_encoder(bottleneck=True)

        networks = {}
        networks['value'] = MonolithicVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)
        networks['target_value'] = copy.deepcopy(networks['value'])
        if discrete:
            action_dim = actions[0] + 1
            networks['actor'] = DiscretePolicy(actor_hidden_dims, action_dim=action_dim)
        else:
            action_dim = actions.shape[-1]
            networks['actor'] = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)
        high_action_dim = observations.shape[-1] if not use_rep else rep_dim
        networks['high_actor'] = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        network_def = HierarchicalActorCritic(encoders=encoders, networks=networks, use_waypoints=use_waypoints, sorb_hl=sorb_hl, sorb_ll=sorb_ll)
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(key, observations, observations)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_value'] = params['networks_value']
        network = network.replace(params=freeze(params))

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, high_temperature=high_temperature,
            target_update_rate=tau, pretrain_expectile=pretrain_expectile, way_steps=way_steps, rep_dim=rep_dim,
            policy_train_rep=policy_train_rep,
            use_rep=use_rep, use_waypoints=use_waypoints,
        ))

        return JointTrainAgent(rng, network=network, config=config)
