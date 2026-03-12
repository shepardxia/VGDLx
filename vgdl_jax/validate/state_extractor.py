"""
State extraction from py-vgdl games for cross-engine validation.

Extracts a normalized state dict that can be compared against vgdl-jax.
Handles coordinate conversion (py-vgdl pixel coords -> grid row,col)
and sorts positions for order-independent comparison.
"""
def extract_pyvgdl_state(game, sprite_key_order, block_size=10):
    """Extract state from a py-vgdl BasicGameLevel into a normalized dict.

    Args:
        game: BasicGameLevel instance (after build_level / tick)
        sprite_key_order: list of sprite keys in type_idx order
            (matching vgdl-jax parser output, e.g. game_def.sprites[i].key)
        block_size: pixel size of one grid cell (default 10)

    Returns:
        dict with:
            'types': {type_idx: {
                'key': str,
                'alive_count': int,
                'positions': sorted list of (row, col) tuples (grid coords),
                'orientations': sorted list of (row, col) tuples,
            }}
            'score': int
            'done': bool
            'step': int
    """
    registry = game.sprite_registry
    types = {}

    for type_idx, key in enumerate(sprite_key_order):
        live_sprites = registry._live_sprites_by_key.get(key, [])

        positions = []
        orientations = []
        for s in live_sprites:
            # py-vgdl: rect.topleft = (x_pixels, y_pixels)
            # where x = col * block_size, y = row * block_size
            col = s.rect.left / block_size
            row = s.rect.top / block_size
            positions.append((row, col))

            # Orientation: py-vgdl uses (x, y) pixel-space vectors
            # Convert to (row, col): row = y component, col = x component
            ori = getattr(s, 'orientation', None)
            if ori is not None:
                # Vector2 or tuple — (x, y) in pixel space
                ori_row = float(ori[1]) if hasattr(ori, '__getitem__') else 0.0
                ori_col = float(ori[0]) if hasattr(ori, '__getitem__') else 0.0
                orientations.append((ori_row, ori_col))
            else:
                orientations.append((0.0, 0.0))

        # Sort for order-independent comparison
        positions.sort()
        orientations.sort()

        types[type_idx] = {
            'key': key,
            'alive_count': len(live_sprites),
            'positions': positions,
            'orientations': orientations,
        }

    return {
        'types': types,
        'score': game.score,
        'done': game.ended,
        'step': game.time,
    }


def extract_jax_state(state, game_def, static_grid_map=None, block_size=1):
    """Extract state from a vgdl-jax GameState into the same normalized dict.

    Args:
        state: GameState (flax.struct.dataclass)
        game_def: GameDef from parser
        static_grid_map: dict mapping type_idx → static_grid_idx, or None
        block_size: pixels per cell (for pixel→cell conversion)

    Returns:
        Same dict format as extract_pyvgdl_state.
    """
    import numpy as np
    if static_grid_map is None:
        static_grid_map = {}
    static_grids = np.asarray(state.static_grids)
    types = {}

    for sd in game_def.sprites:
        type_idx = sd.type_idx

        if type_idx in static_grid_map:
            # Reconstruct positions from static grid
            sg_idx = static_grid_map[type_idx]
            grid = static_grids[sg_idx]
            coords = np.argwhere(grid)  # [N, 2] of (row, col)
            n_alive = len(coords)
            positions = [(float(r), float(c)) for r, c in coords]
            # Static types have default orientation (from SpriteDef)
            orientations = [tuple(float(x) for x in sd.orientation)] * n_alive
        else:
            alive_mask = state.alive[type_idx]
            n_alive = int(alive_mask.sum())
            positions = []
            orientations = []
            for slot in range(alive_mask.shape[0]):
                if alive_mask[slot]:
                    # Pixel→cell conversion
                    row = float(state.positions[type_idx, slot, 0]) / block_size
                    col = float(state.positions[type_idx, slot, 1]) / block_size
                    positions.append((row, col))
                    ori_row = float(state.orientations[type_idx, slot, 0])
                    ori_col = float(state.orientations[type_idx, slot, 1])
                    orientations.append((ori_row, ori_col))

        positions.sort()
        orientations.sort()

        types[type_idx] = {
            'key': sd.key,
            'alive_count': n_alive,
            'positions': positions,
            'orientations': orientations,
        }

    return {
        'types': types,
        'score': int(state.score),
        'done': bool(state.done),
        'step': int(state.step_count),
    }
