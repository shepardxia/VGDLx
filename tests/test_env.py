import os
import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLxEnv
from conftest import GAMES_DIR


def test_env_render_jax():
    """env.render() returns an RGB image with correct shape."""
    game_file = os.path.join(GAMES_DIR, 'chase.txt')
    level_file = os.path.join(GAMES_DIR, 'chase_lvl0.txt')
    env = VGDLxEnv(game_file, level_file)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    img = env.render(state, block_size=10)
    H, W = 11, 24  # chase_lvl0 dimensions
    assert img.shape == (H * 10, W * 10, 3)
    assert img.dtype == jnp.uint8

    # Walls should be gray (90, 90, 90), not all white
    assert not jnp.all(img == 255)
