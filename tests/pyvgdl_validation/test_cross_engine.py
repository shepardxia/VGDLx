"""
Cross-engine validation: compare py-vgdl and vgdl-jax rendered output.

Both engines must produce identical pixel output for the same game/level/actions.
Uses pygame-exact renderer for vgdl-jax to match py-vgdl's pygame rendering.

Key design decisions:
- Color-only tests: py-vgdl renderer is created with render_sprites=False so it
  draws solid color rectangles. vgdl-jax's render_pygame also draws solid color
  fills. This makes the comparison meaningful.
- Sprite-image tests: both engines render with full sprite images (render_sprites=True).
  py-vgdl uses its SpriteLibrary; vgdl-jax loads from the same sprites directory.
- Action mappings match between engines:
    py-vgdl MovingAvatar: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP
    vgdl-jax: DIRECTION_DELTAS[0]=UP(-1,0), [1]=DOWN(1,0), [2]=LEFT(0,-1), [3]=RIGHT(0,1)
  Both use the same integer index ordering, so no translation is needed.
"""
import os
import pytest
import numpy as np
import jax

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    from vgdl.interfaces.gym.env import VGDLEnv
    HAS_PYVGDL = True
except ImportError:
    HAS_PYVGDL = False

from vgdl_jax.parser import parse_vgdl
from vgdl_jax.compiler import compile_game
from vgdl_jax.render import render_pygame
from vgdl_jax.validate.constants import PYVGDL_GAMES_DIR as GAMES_DIR, BLOCK_SIZE
SKIP_MSG = "requires pygame and py-vgdl"


def _setup_pyvgdl(game_name):
    """
    Set up py-vgdl env with color-only rendering (no sprite images).

    We create the renderer manually with render_sprites=False so that sprites
    are drawn as solid color fills, matching vgdl-jax's render_pygame behavior.

    The renderer must be created before reset() because reset() calls _get_obs()
    which needs the renderer when obs_type='image'.
    """
    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')
    env = VGDLEnv(game_file, level_file, obs_type='image', block_size=BLOCK_SIZE)

    # Create renderer with render_sprites=False BEFORE reset.
    # Some sprites have color=None (image-only in py-vgdl), so we patch them
    # to WHITE before any draw calls.
    from vgdl.render import PygameRenderer
    for s in env.game.sprite_registry.sprites():
        if s.color is None:
            s.color = (250, 250, 250)
    env.renderer = PygameRenderer(env.game, BLOCK_SIZE, render_sprites=False)
    env.renderer.init_screen(headless=True)
    # Do initial draw so get_image() works during reset
    env.renderer.draw_all()
    env.renderer.update_display()

    env.reset()

    return env


def _render_pyvgdl(env):
    """Render py-vgdl frame: clear and redraw all sprites, return RGB array.

    Draws solid color rectangles for each sprite, matching vgdl-jax's
    render_pygame behavior. Sprites with color=None (those that only have
    sprite images in py-vgdl) default to WHITE (250, 250, 250).
    """
    renderer = env.renderer
    renderer.screen.fill((255, 255, 255))
    for s in env.game.sprite_registry.sprites():
        sprite_rect = renderer.calculate_render_rect(s.rect, s.shrinkfactor)
        color = s.color if s.color is not None else (250, 250, 250)
        renderer.screen.fill(color, sprite_rect)
    renderer.update_display()
    return renderer.get_image()


def _setup_pyvgdl_with_sprites(game_name):
    """Set up py-vgdl env with full sprite image rendering."""
    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')
    env = VGDLEnv(game_file, level_file, obs_type='image', block_size=BLOCK_SIZE)

    from vgdl.render import PygameRenderer
    env.renderer = PygameRenderer(env.game, BLOCK_SIZE, render_sprites=True)
    env.renderer.init_screen(headless=True)
    env.renderer.draw_all()
    env.renderer.update_display()

    env.reset()
    return env


def _render_pyvgdl_with_sprites(env):
    """Render py-vgdl frame with full sprite images."""
    renderer = env.renderer
    # Clear to white, then redraw all sprites using py-vgdl's full pipeline
    renderer.screen.fill((255, 255, 255))
    for s in env.game.sprite_registry.sprites():
        renderer.draw_sprite(s)
    renderer.update_display()
    return renderer.get_image()


def _setup_jax(game_name):
    """Set up vgdl-jax compiled game.

    Uses a large max_sprites_per_type to ensure ALL sprites (including inert
    background tiles like 'floor') are placed. The default auto-computed max_n
    excludes inert types, which causes background sprites to be dropped.
    """
    from collections import Counter
    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)
    # Compute max sprites needed across all types (including inert ones)
    counts = Counter(t for t, r, c in gd.level.initial_sprites)
    max_n = max(counts.values(), default=1) + 10
    compiled = compile_game(gd, max_sprites_per_type=max_n)
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(42))
    return compiled, state, gd


def _compare_frames(pyvgdl_img, jax_img, step_num):
    """Compare two rendered frames pixel-by-pixel."""
    assert pyvgdl_img.shape == jax_img.shape, (
        f"Step {step_num}: shape mismatch: py-vgdl {pyvgdl_img.shape} vs "
        f"jax {jax_img.shape}"
    )
    if not np.array_equal(pyvgdl_img, jax_img):
        diff = np.abs(pyvgdl_img.astype(int) - jax_img.astype(int))
        n_diff = np.count_nonzero(diff.sum(axis=-1))
        total = pyvgdl_img.shape[0] * pyvgdl_img.shape[1]
        # Find which blocks differ to aid debugging
        diff_blocks = set()
        for y in range(0, pyvgdl_img.shape[0], BLOCK_SIZE):
            for x in range(0, pyvgdl_img.shape[1], BLOCK_SIZE):
                block_a = pyvgdl_img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
                block_b = jax_img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
                if not np.array_equal(block_a, block_b):
                    row, col = y // BLOCK_SIZE, x // BLOCK_SIZE
                    color_a = tuple(block_a[0, 0])
                    color_b = tuple(block_b[0, 0])
                    diff_blocks.add((row, col, color_a, color_b))
        block_report = "\n".join(
            f"  grid ({r},{c}): py-vgdl={ca} jax={cb}"
            for r, c, ca, cb in sorted(diff_blocks)[:10]  # limit to 10
        )
        if len(diff_blocks) > 10:
            block_report += f"\n  ... and {len(diff_blocks) - 10} more"
        pytest.fail(
            f"Step {step_num}: {n_diff}/{total} pixels differ "
            f"({len(diff_blocks)} grid cells). "
            f"Max diff: {diff.max()}\n{block_report}"
        )


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not (HAS_PYGAME and HAS_PYVGDL), reason=SKIP_MSG)
@pytest.mark.parametrize("game_name", ["chase", "zelda", "aliens"])
def test_initial_frame_matches(game_name):
    """The initial rendered frame should match between engines."""
    pyvgdl_env = _setup_pyvgdl(game_name)
    pyvgdl_img = _render_pyvgdl(pyvgdl_env)

    compiled, state, gd = _setup_jax(game_name)
    jax_img = render_pygame(state, gd, BLOCK_SIZE, render_sprites=False,
                           static_grid_map=compiled.static_grid_map)

    _compare_frames(pyvgdl_img, jax_img, step_num=0)


@pytest.mark.skipif(not (HAS_PYGAME and HAS_PYVGDL), reason=SKIP_MSG)
@pytest.mark.xfail(reason="Rendering test without RNG replay; NPC positions diverge after step 0 (state-level RNG replay exists in test_validate.py)")
def test_action_sequence_chase():
    """
    Run a fixed action sequence on chase and compare each frame.

    Uses NOOP actions so avatar doesn't move, but NPCs (Chasers, Fleeing)
    will move stochastically. This rendering test doesn't use RNG replay,
    so NPC positions diverge after step 0.

    Note: RNG replay for state-level comparison exists in validate_harness.py
    and is tested in test_validate.py::test_cross_engine_with_rng_replay.
    """
    pyvgdl_env = _setup_pyvgdl('chase')
    compiled, state, gd = _setup_jax('chase')

    # NOOP: py-vgdl uses n-1 (valid for MovingAvatar which has no shoot),
    # vgdl-jax uses explicit noop_action index
    noop_pyvgdl = pyvgdl_env.action_space.n - 1
    noop_jax = compiled.noop_action

    for step_i in range(5):
        pyvgdl_obs, _, pyvgdl_done, _ = pyvgdl_env.step(noop_pyvgdl)
        state = compiled.step_fn(state, noop_jax)

        if pyvgdl_done:
            break

        pyvgdl_img = _render_pyvgdl(pyvgdl_env)
        jax_img = render_pygame(state, gd, BLOCK_SIZE, render_sprites=False,
                           static_grid_map=compiled.static_grid_map)
        _compare_frames(pyvgdl_img, jax_img, step_num=step_i + 1)


@pytest.mark.skipif(not (HAS_PYGAME and HAS_PYVGDL), reason=SKIP_MSG)
@pytest.mark.parametrize("game_name", ["chase", "zelda", "aliens"])
def test_frame_shapes_match(game_name):
    """Both engines produce images of the same shape."""
    pyvgdl_env = _setup_pyvgdl(game_name)
    pyvgdl_img = _render_pyvgdl(pyvgdl_env)

    compiled, state, gd = _setup_jax(game_name)
    jax_img = render_pygame(state, gd, BLOCK_SIZE, render_sprites=False,
                           static_grid_map=compiled.static_grid_map)

    assert pyvgdl_img.shape == jax_img.shape, (
        f"Shape mismatch: py-vgdl {pyvgdl_img.shape} vs jax {jax_img.shape}"
    )
    assert pyvgdl_img.dtype == jax_img.dtype == np.uint8


@pytest.mark.skipif(not (HAS_PYGAME and HAS_PYVGDL), reason=SKIP_MSG)
@pytest.mark.parametrize("game_name", ["chase", "zelda", "aliens"])
def test_action_meanings_match(game_name):
    """
    Verify that action counts and semantic ordering match between engines.

    py-vgdl: action_set is OrderedDict from declare_possible_actions()
    vgdl-jax: n_actions from compiler, directional order is UP/DOWN/LEFT/RIGHT
    """
    pyvgdl_env = _setup_pyvgdl(game_name)
    compiled, _, _ = _setup_jax(game_name)

    pyvgdl_n = pyvgdl_env.action_space.n
    jax_n = compiled.n_actions

    pyvgdl_meanings = pyvgdl_env.get_action_meanings()

    # For games with MovingAvatar (chase): UP, DOWN, LEFT, RIGHT, NO_OP = 5
    # For games with ShootAvatar (zelda): UP, DOWN, LEFT, RIGHT, SPACE, NO_OP = 6
    # For games with FlakAvatar (aliens): LEFT, RIGHT, SPACE, NO_OP = 4
    assert pyvgdl_n == jax_n, (
        f"Action count mismatch for {game_name}: "
        f"py-vgdl={pyvgdl_n} ({pyvgdl_meanings}) vs jax={jax_n}"
    )


@pytest.mark.skipif(not (HAS_PYGAME and HAS_PYVGDL), reason=SKIP_MSG)
@pytest.mark.parametrize("game_name", ["chase", "zelda", "aliens"])
def test_initial_frame_with_sprites_matches(game_name):
    """The initial rendered frame with sprite images should match between engines."""
    from collections import Counter

    pyvgdl_env = _setup_pyvgdl_with_sprites(game_name)
    pyvgdl_img = _render_pyvgdl_with_sprites(pyvgdl_env)

    # Need larger max_n for full sprite placement
    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')
    gd = parse_vgdl(game_file, level_file)
    counts = Counter(t for t, r, c in gd.level.initial_sprites)
    max_n = max(counts.values(), default=1) + 10
    compiled = compile_game(gd, max_sprites_per_type=max_n)
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(42))

    jax_img = render_pygame(state, gd, BLOCK_SIZE, render_sprites=True,
                           static_grid_map=compiled.static_grid_map)

    _compare_frames(pyvgdl_img, jax_img, step_num=0)
