"""
Gymnax-style JAX environment wrapper for VGDL games.
Supports jit, vmap for batched RL training.
"""
from functools import partial

import jax
import jax.numpy as jnp

from vgdl_jax.parser import parse_vgdl, parse_vgdl_text
from vgdl_jax.compiler import compile_game
from vgdl_jax.render import render_rgb


class VGDLJaxEnv:
    """
    A VGDL game environment compatible with JAX transformations.

    Usage:
        env = VGDLJaxEnv('game.txt', 'game_lvl0.txt')
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)
        obs, state, reward, done, info = env.step(state, action)

        # Batched:
        obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    """

    def __init__(self, game_file, level_file, max_sprites_per_type=None):
        game_def = parse_vgdl(game_file, level_file)
        self._init_from_game_def(game_def, max_sprites_per_type)

    @classmethod
    def from_text(cls, game_text, level_text, max_sprites_per_type=None):
        """Create env from game/level text strings (no files needed)."""
        env = cls.__new__(cls)
        game_def = parse_vgdl_text(game_text, level_text)
        env._init_from_game_def(game_def, max_sprites_per_type)
        return env

    def _init_from_game_def(self, game_def, max_sprites_per_type):
        self.compiled = compile_game(game_def, max_sprites_per_type)
        self.n_actions = self.compiled.n_actions
        self.noop_action = self.compiled.noop_action
        n_types = len(game_def.sprites)
        self._height = game_def.level.height
        self._width = game_def.level.width
        self._n_types = n_types
        # block_size for pixel→cell conversion in _get_obs
        from vgdl_jax.data_model import get_block_size
        self._block_size = get_block_size(game_def)
        self.obs_shape = (n_types, self._height, self._width)

        # Static grid map: type_idx → static_grid_idx
        self._static_grid_map = self.compiled.static_grid_map

        # Pre-compute observation index arrays for vectorized _get_obs
        static_type_indices = []
        static_grid_indices = []
        dynamic_type_indices = []
        for t in range(n_types):
            if t in self._static_grid_map:
                static_type_indices.append(t)
                static_grid_indices.append(self._static_grid_map[t])
            else:
                dynamic_type_indices.append(t)
        self._static_type_indices = jnp.array(static_type_indices, dtype=jnp.int32)
        self._static_grid_indices = jnp.array(static_grid_indices, dtype=jnp.int32)
        self._dynamic_type_indices = jnp.array(dynamic_type_indices, dtype=jnp.int32)
        self._has_static = len(static_type_indices) > 0
        self._has_dynamic = len(dynamic_type_indices) > 0

        # Build color table from sprite definitions
        self._colors = jnp.array(
            [sd.color for sd in game_def.sprites], dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        """Reset the environment and return (obs, state)."""
        state = self.compiled.init_state.replace(rng=rng)
        return self._get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """Take a step and return (obs, state, reward, done, info)."""
        prev_score = state.score
        state = self.compiled.step_fn(state, action)
        obs = self._get_obs(state)
        reward = state.score - prev_score
        return obs, state, reward, state.done, {}

    def _get_obs(self, state):
        """Render state as a [n_types, height, width] binary grid."""
        grid = jnp.zeros(self.obs_shape, dtype=jnp.bool_)

        # Static types: batch copy from static_grids via index arrays
        if self._has_static:
            grid = grid.at[self._static_type_indices].set(
                state.static_grids[self._static_grid_indices])

        # Dynamic types: single vectorized scatter (pixel→cell)
        if self._has_dynamic:
            dyn_pos = state.positions[self._dynamic_type_indices]
            dyn_alive = state.alive[self._dynamic_type_indices]
            rows = jnp.clip(dyn_pos[:, :, 0] // self._block_size,
                            0, self._height - 1)
            cols = jnp.clip(dyn_pos[:, :, 1] // self._block_size,
                            0, self._width - 1)
            type_idx = jnp.broadcast_to(
                self._dynamic_type_indices[:, None], dyn_alive.shape)
            grid = grid.at[type_idx.ravel(), rows.ravel(), cols.ravel()].max(
                dyn_alive.ravel())

        return grid

    def render(self, state, block_size=10):
        """Render game state to RGB image.

        Args:
            state: GameState
            block_size: pixels per grid cell

        Returns:
            [H*block_size, W*block_size, 3] uint8 RGB image
        """
        obs = self._get_obs(state)
        return render_rgb(obs, self._colors, block_size)
