"""Parameterized integration tests for all supported games."""
import jax
import jax.numpy as jnp
import pytest
from conftest import make_env, ALL_GAMES


@pytest.mark.parametrize('game', ALL_GAMES)
def test_runs_100_steps(game):
    env = make_env(game)
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)

    for i in range(100):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, env.n_actions)
        obs, state, reward, done, info = env.step(state, action)
        if done:
            break

    assert state.step_count > 0



@pytest.mark.parametrize('game', ALL_GAMES)
def test_vmap(game):
    env = make_env(game)
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 4)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    obs2, states2, rewards, dones, infos = jax.vmap(env.step)(state_batch, actions)
    assert obs2.shape[0] == 4
