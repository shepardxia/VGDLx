#!/usr/bin/env python
"""
Human-playable VGDLx game.

Usage:
    python scripts/play.py zelda
    python scripts/play.py aliens --scale 3
    python scripts/play.py path/to/game.txt path/to/level.txt

Controls:
    Arrow keys / WASD  — move (LEFT, RIGHT, DOWN, UP)
    Space              — USE (shoot/interact)
    N                  — NOOP (do nothing)
    R                  — restart
    Q / ESC            — quit
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pygame

# Add parent to path for vgdl_jax imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgdl_jax.env import VGDLxEnv
from vgdl_jax.validate.discovery import discover_games

# ── Sprite image loading ───────────────────────────────────────────

# Bundled sprites in vgdl_jax/sprites/, fallback to GVGAI/sprites/
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRITE_DIR = os.path.join(_PKG_DIR, 'vgdl_jax', 'sprites')
if not os.path.isdir(SPRITE_DIR):
    SPRITE_DIR = os.path.join(_PKG_DIR, '..', 'GVGAI', 'sprites')


def _load_sprite_image(img_path, block_size):
    """Load and scale a sprite image, or return None."""
    if not img_path:
        return None
    base = os.path.join(SPRITE_DIR, img_path.replace('/', os.sep))
    # Try: exact.png, then _0.png (GVGAI animation frame 0)
    for suffix in ['.png', '_0.png']:
        path = base + suffix
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (block_size, block_size))
    return None


def _build_sprite_surfaces(game_def, block_size):
    """Build per-type pygame surfaces: sprite image if available, else colored block."""
    surfaces = {}
    for sd in game_def.sprites:
        img_surface = _load_sprite_image(sd.img, block_size)
        if img_surface:
            surfaces[sd.type_idx] = img_surface
        else:
            # Solid color fallback
            surf = pygame.Surface((block_size, block_size), pygame.SRCALPHA)
            r, g, b = sd.color
            surf.fill((r, g, b, 255))
            surfaces[sd.type_idx] = surf
    return surfaces


# ── Key mapping ────────────────────────────────────────────────────

def _build_key_map(action_map):
    """Map pygame keys to action indices based on env.action_map."""
    # Reverse map: name → index
    name_to_idx = {v: k for k, v in action_map.items()}

    key_map = {}

    # Arrow keys + WASD for directions
    for pygame_key, action_name in [
        (pygame.K_LEFT, 'LEFT'), (pygame.K_a, 'LEFT'),
        (pygame.K_RIGHT, 'RIGHT'), (pygame.K_d, 'RIGHT'),
        (pygame.K_DOWN, 'DOWN'), (pygame.K_s, 'DOWN'),
        (pygame.K_UP, 'UP'), (pygame.K_w, 'UP'),
    ]:
        if action_name in name_to_idx:
            key_map[pygame_key] = name_to_idx[action_name]

    # Space for USE
    if 'USE' in name_to_idx:
        key_map[pygame.K_SPACE] = name_to_idx['USE']

    # N for NOOP
    if 'NIL' in name_to_idx:
        key_map[pygame.K_n] = name_to_idx['NIL']

    return key_map


# ── Rendering ──────────────────────────────────────────────────────

def render_pygame(state, game_def, surfaces, static_grid_map, block_size, height, width):
    """Render game state to a pygame surface."""
    surface = pygame.Surface((width * block_size, height * block_size))
    surface.fill((0, 0, 0))  # black background

    # Draw static grids first (background layer)
    for type_idx, sg_idx in static_grid_map.items():
        grid = np.array(state.static_grids[sg_idx])
        surf = surfaces.get(type_idx)
        if surf is None:
            continue
        for r in range(height):
            for c in range(width):
                if grid[r, c]:
                    surface.blit(surf, (c * block_size, r * block_size))

    # Draw dynamic sprites on top (reverse type order = higher types on top)
    for sd in reversed(game_def.sprites):
        ti = sd.type_idx
        if ti in static_grid_map:
            continue
        surf = surfaces.get(ti)
        if surf is None:
            continue
        alive = np.array(state.alive[ti])
        positions = np.array(state.positions[ti])
        for slot in range(alive.shape[0]):
            if alive[slot]:
                py, px = positions[slot]
                surface.blit(surf, (int(px), int(py)))

    return surface


# ── Main game loop ─────────────────────────────────────────────────

def play(env, game_def, scale=2, fps=15):
    """Run the interactive game loop."""
    from vgdl_jax.data_model import get_block_size
    block_size = get_block_size(game_def)
    height = game_def.level.height
    width = game_def.level.width

    pygame.init()
    win_w = width * block_size * scale
    win_h = height * block_size * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f'VGDLx — {game_def.sprites[0].key if game_def.sprites else "game"}')
    clock = pygame.time.Clock()

    surfaces = _build_sprite_surfaces(game_def, block_size)
    key_map = _build_key_map(env.action_map)
    sgm = env.compiled.static_grid_map

    # Info bar
    font = pygame.font.SysFont('monospace', 14 * scale)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    running = True
    step_count = 0
    total_reward = 0

    while running:
        # Render
        game_surface = render_pygame(state, game_def, surfaces, sgm, block_size, height, width)
        scaled = pygame.transform.scale(game_surface, (win_w, win_h))
        screen.blit(scaled, (0, 0))

        # HUD
        hud = font.render(
            f'Step: {step_count}  Score: {int(state.score)}  Reward: {int(total_reward)}',
            True, (255, 255, 255))
        screen.blit(hud, (4, 2))

        if state.done:
            result = 'WIN!' if state.win else 'GAME OVER'
            msg = font.render(f'{result} — press R to restart', True, (255, 255, 0))
            screen.blit(msg, (win_w // 2 - msg.get_width() // 2, win_h // 2))

        pygame.display.flip()

        # Handle input — check held keys for continuous movement
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    rng, key = jax.random.split(rng)
                    obs, state = env.reset(key)
                    step_count = 0
                    total_reward = 0

        # Read held keys — game steps every frame (NPCs move even without input)
        action = env.noop_action  # default NOOP
        keys_pressed = pygame.key.get_pressed()
        for pygame_key, act_idx in key_map.items():
            if keys_pressed[pygame_key]:
                action = act_idx
                break  # first held key wins

        # Step every frame (not just on keypress)
        if not state.done:
            obs, state, reward, done, info = env.step(state, action)
            step_count += 1
            total_reward += int(reward)

        clock.tick(fps)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Play a VGDLx game')
    parser.add_argument('game', help='Game name (e.g., zelda) or path to game.txt')
    parser.add_argument('level', nargs='?', default=None, help='Path to level.txt (auto-detected if game name given)')
    parser.add_argument('--scale', type=int, default=2, help='Display scale factor (default: 2)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second (default: 10)')
    args = parser.parse_args()

    # Resolve game file
    if os.path.isfile(args.game):
        game_file = args.game
        level_file = args.level
        if level_file is None:
            # Try to find level file
            base = args.game.replace('.txt', '')
            level_file = f'{base}_lvl0.txt'
            if not os.path.isfile(level_file):
                print(f'ERROR: No level file found. Tried {level_file}')
                sys.exit(1)
    else:
        # Look up by name: bundled games first, then GVGAI
        bundled_dir = os.path.join(_PKG_DIR, 'vgdl_jax', 'games', 'gridphysics')
        gvgai_dir = os.path.join(_PKG_DIR, '..', 'GVGAI', 'examples', 'gridphysics')
        game_map = {}
        for d in [bundled_dir, gvgai_dir]:
            if os.path.isdir(d):
                for e in discover_games(d, source='gvgai'):
                    if e.name not in game_map:
                        game_map[e.name] = e
        if args.game not in game_map:
            available = sorted(game_map.keys())[:20]
            print(f"ERROR: Unknown game '{args.game}'. Available: {', '.join(available)}...")
            sys.exit(1)
        entry = game_map[args.game]
        game_file = entry.game_file
        level_file = entry.level_files[0]

    print(f'Loading {game_file}...')
    env = VGDLxEnv(game_file, level_file)
    game_def = env.compiled.game_def

    print(f'Actions: {env.action_map}')
    print(f'Grid: {game_def.level.height}x{game_def.level.width}')
    print(f'Controls: Arrow/WASD=move, Space=use, N=noop, R=restart, Q=quit')
    print()

    play(env, game_def, scale=args.scale, fps=args.fps)


if __name__ == '__main__':
    main()
