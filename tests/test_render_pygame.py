import os
import pytest
import numpy as np
import jax

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from vgdl_jax.parser import parse_vgdl
from vgdl_jax.compiler import compile_game
from conftest import GAMES_DIR


@pytest.mark.skipif(not HAS_PYGAME, reason="pygame not installed")
def test_render_pygame_shape():
    """Pygame renderer returns correct shape."""
    from vgdl_jax.render import render_pygame

    gd = parse_vgdl(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'),
    )
    compiled = compile_game(gd)
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(0))
    block_size = 10

    img = render_pygame(state, gd, block_size, static_grid_map=compiled.static_grid_map)
    H, W = gd.level.height, gd.level.width
    assert img.shape == (H * block_size, W * block_size, 3)
    assert img.dtype == np.uint8


@pytest.mark.skipif(not HAS_PYGAME, reason="pygame not installed")
def test_render_pygame_has_walls():
    """Pygame renderer draws walls (not all white)."""
    from vgdl_jax.render import render_pygame

    gd = parse_vgdl(
        os.path.join(GAMES_DIR, 'chase.txt'),
        os.path.join(GAMES_DIR, 'chase_lvl0.txt'),
    )
    compiled = compile_game(gd)
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(0))

    img = render_pygame(state, gd, block_size=10, static_grid_map=compiled.static_grid_map)
    assert not np.all(img == 255)


@pytest.mark.skipif(not HAS_PYGAME, reason="pygame not installed")
@pytest.mark.xfail(reason="JAX renderer only does color blocks; pygame renderer now draws sprite images")
def test_render_pygame_matches_jax():
    """Pygame renderer and JAX renderer produce the same output at block_size=1."""
    import jax.numpy as jnp
    from vgdl_jax.render import render_pygame, render_rgb
    from vgdl_jax.env import VGDLxEnv

    game_file = os.path.join(GAMES_DIR, 'chase.txt')
    level_file = os.path.join(GAMES_DIR, 'chase_lvl0.txt')
    env = VGDLxEnv(game_file, level_file)
    gd = env.compiled.game_def

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # Both renderers with same block_size
    jax_img = np.asarray(env.render(state, block_size=10))
    pygame_img = render_pygame(state, gd, block_size=10,
                               static_grid_map=env.compiled.static_grid_map)

    np.testing.assert_array_equal(jax_img, pygame_img)


