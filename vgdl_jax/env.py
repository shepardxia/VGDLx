"""
VGDLx game environment for RL.

Single-env API. Use jax.vmap for batching.

Observation formats:
    'channels': [n_types, H, W] bool       — one binary channel per sprite type
    'grid':     [H, W, max_occupancy] int32 — per-cell type ID list (-1 = empty)
    'entities': [max_entities, 3] int32     — (type_idx, row, col) per alive sprite (-1 = padding)
"""
from functools import partial

import jax
import jax.numpy as jnp

from vgdl_jax.parser import parse_vgdl, parse_vgdl_text
from vgdl_jax.compiler import compile_game
from vgdl_jax.render import render_rgb


# ── Observation builders ───────────────────────────────────────────
# Pure functions: (state, cfg) → obs array.

def obs_channels(state, cfg):
    """[n_types, H, W] bool — one binary channel per sprite type."""
    grid = jnp.zeros((cfg['n_types'], cfg['height'], cfg['width']), dtype=jnp.bool_)
    if cfg['has_static']:
        grid = grid.at[cfg['static_type_indices']].set(
            state.static_grids[cfg['static_grid_indices']])
    if cfg['has_dynamic']:
        dyn_pos = state.positions[cfg['dynamic_type_indices']]
        dyn_alive = state.alive[cfg['dynamic_type_indices']]
        bs = cfg['block_size']
        rows = jnp.clip(dyn_pos[:, :, 0] // bs, 0, cfg['height'] - 1)
        cols = jnp.clip(dyn_pos[:, :, 1] // bs, 0, cfg['width'] - 1)
        type_idx = jnp.broadcast_to(
            cfg['dynamic_type_indices'][:, None], dyn_alive.shape)
        grid = grid.at[type_idx.ravel(), rows.ravel(), cols.ravel()].max(
            dyn_alive.ravel())
    return grid


def obs_grid(state, cfg):
    """[H, W, max_occupancy] int32 — per-cell type ID list, -1 = empty."""
    H, W, K = cfg['height'], cfg['width'], cfg['max_occupancy']
    bs = cfg['block_size']
    grid = jnp.full((H, W, K), -1, dtype=jnp.int32)
    counts = jnp.zeros((H, W), dtype=jnp.int32)

    if cfg['has_static']:
        for i in range(len(cfg['static_type_indices'])):
            ti = cfg['static_type_indices'][i]
            sg = cfg['static_grid_indices'][i]
            occupied = state.static_grids[sg]
            r, c = jnp.where(occupied, size=H * W, fill_value=0)
            valid = occupied[r, c]
            slot = jnp.clip(counts[r, c], 0, K - 1)
            grid = grid.at[r, c, slot].set(jnp.where(valid, ti, grid[r, c, slot]))
            counts = counts.at[r, c].add(valid.astype(jnp.int32))

    if cfg['has_dynamic']:
        dyn_pos = state.positions[cfg['dynamic_type_indices']]
        dyn_alive = state.alive[cfg['dynamic_type_indices']]
        for i in range(len(cfg['dynamic_type_indices'])):
            ti = cfg['dynamic_type_indices'][i]
            alive_i = dyn_alive[i]
            pos_i = dyn_pos[i]
            rows = jnp.clip(pos_i[:, 0] // bs, 0, H - 1)
            cols = jnp.clip(pos_i[:, 1] // bs, 0, W - 1)
            for j in range(alive_i.shape[0]):
                r, c = rows[j], cols[j]
                slot = jnp.clip(counts[r, c], 0, K - 1)
                grid = grid.at[r, c, slot].set(
                    jnp.where(alive_i[j], ti, grid[r, c, slot]))
                counts = counts.at[r, c].add(alive_i[j].astype(jnp.int32))

    return grid


def obs_entities(state, cfg):
    """[max_entities, 3] int32 — (type_idx, row, col) per alive sprite, -1 = padding."""
    bs = cfg['block_size']
    parts = []

    if cfg['has_static']:
        for i in range(len(cfg['static_type_indices'])):
            ti = cfg['static_type_indices'][i]
            sg = cfg['static_grid_indices'][i]
            occupied = state.static_grids[sg]
            H, W = cfg['height'], cfg['width']
            r, c = jnp.where(occupied, size=H * W, fill_value=0)
            valid = occupied[r, c]
            parts.append(jnp.stack([
                jnp.where(valid, ti, -1),
                jnp.where(valid, r, -1),
                jnp.where(valid, c, -1),
            ], axis=-1))

    if cfg['has_dynamic']:
        for i in range(len(cfg['dynamic_type_indices'])):
            ti = cfg['dynamic_type_indices'][i]
            alive_i = state.alive[ti]
            pos_i = state.positions[ti]
            rows = jnp.clip(pos_i[:, 0] // bs, 0, cfg['height'] - 1)
            cols = jnp.clip(pos_i[:, 1] // bs, 0, cfg['width'] - 1)
            parts.append(jnp.stack([
                jnp.where(alive_i, ti, -1),
                jnp.where(alive_i, rows, -1),
                jnp.where(alive_i, cols, -1),
            ], axis=-1))

    if parts:
        all_ents = jnp.concatenate(parts, axis=0)
    else:
        all_ents = jnp.full((0, 3), -1, dtype=jnp.int32)

    max_e = cfg['max_entities']
    n = all_ents.shape[0]
    if n < max_e:
        all_ents = jnp.concatenate([all_ents, jnp.full((max_e - n, 3), -1, dtype=jnp.int32)])
    return all_ents[:max_e]


_OBS_BUILDERS = {
    'channels': obs_channels,
    'grid': obs_grid,
    'entities': obs_entities,
}


# ── Environment ────────────────────────────────────────────────────

class VGDLxEnv:
    """VGDLx game environment.

    Usage:
        env = VGDLxEnv('game.txt', 'level.txt')
        obs, state = env.reset()
        obs, state, reward, done = env.step(state, action)

    Batched (user vmaps):
        keys = jax.random.split(jax.random.PRNGKey(0), 256)
        obs_b, state_b = jax.vmap(env.reset)(keys)
        obs_b, state_b, r_b, d_b = jax.vmap(env.step)(state_b, actions)
    """

    def __init__(self, game_file, level_file, *,
                 obs_type='channels', max_sprites_per_type=None):
        game_def = parse_vgdl(game_file, level_file)
        self._init(game_def, obs_type, max_sprites_per_type)

    @classmethod
    def from_text(cls, game_text, level_text, *,
                  obs_type='channels', max_sprites_per_type=None):
        """Create from game/level text strings."""
        env = cls.__new__(cls)
        game_def = parse_vgdl_text(game_text, level_text)
        env._init(game_def, obs_type, max_sprites_per_type)
        return env

    def _init(self, game_def, obs_type, max_sprites_per_type):
        assert obs_type in _OBS_BUILDERS, \
            f"obs_type must be one of {list(_OBS_BUILDERS)}, got {obs_type!r}"

        self.compiled = compile_game(game_def, max_sprites_per_type)
        self.n_actions = self.compiled.n_actions
        self.noop_action = self.compiled.noop_action

        n_types = len(game_def.sprites)
        height = game_def.level.height
        width = game_def.level.width
        from vgdl_jax.data_model import get_block_size
        block_size = get_block_size(game_def)

        # Index arrays for obs builders
        sgm = self.compiled.static_grid_map
        static_ti, static_gi, dynamic_ti = [], [], []
        for t in range(n_types):
            if t in sgm:
                static_ti.append(t)
                static_gi.append(sgm[t])
            else:
                dynamic_ti.append(t)

        # Obs config — passed to obs builder functions
        max_n = self.compiled.init_state.alive.shape[1]
        n_static_cells = sum(
            int(self.compiled.init_state.static_grids[sg].sum())
            for sg in sgm.values())

        self._obs_cfg = {
            'n_types': n_types,
            'height': height,
            'width': width,
            'block_size': block_size,
            'static_type_indices': jnp.array(static_ti, dtype=jnp.int32),
            'static_grid_indices': jnp.array(static_gi, dtype=jnp.int32),
            'dynamic_type_indices': jnp.array(dynamic_ti, dtype=jnp.int32),
            'has_static': len(static_ti) > 0,
            'has_dynamic': len(dynamic_ti) > 0,
            'max_occupancy': min(n_types, 4),
            'max_entities': len(dynamic_ti) * max_n + n_static_cells,
        }

        # Obs shape
        if obs_type == 'channels':
            self.obs_shape = (n_types, height, width)
        elif obs_type == 'grid':
            self.obs_shape = (height, width, self._obs_cfg['max_occupancy'])
        else:
            self.obs_shape = (self._obs_cfg['max_entities'], 3)

        self._obs_fn = _OBS_BUILDERS[obs_type]
        self._colors = jnp.array([sd.color for sd in game_def.sprites], dtype=jnp.uint8)
        self._default_key = jax.random.PRNGKey(0)

        # Action map: int → human-readable name
        self.action_map = {i: n.replace('ACTION_', '')
                           for i, n in enumerate(self.compiled.action_names)}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None):
        """Reset and return (obs, state).

        Args:
            key: JAX PRNGKey. If None, uses default key.
        """
        if key is None:
            key = self._default_key
        state = self.compiled.init_state.replace(rng=key)
        return self._obs_fn(state, self._obs_cfg), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """Step and return (obs, state, reward, done, info)."""
        prev_score = state.score
        state = self.compiled.step_fn(state, action)
        obs = self._obs_fn(state, self._obs_cfg)
        reward = state.score - prev_score
        return obs, state, reward, state.done, {}

    def render(self, state, block_size=10):
        """Render to [H*bs, W*bs, 3] uint8 RGB image."""
        obs = obs_channels(state, self._obs_cfg)
        return render_rgb(obs, self._colors, block_size)
