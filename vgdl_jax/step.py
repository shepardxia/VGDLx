"""
The jittable step function: combines sprite updates, collision detection,
effects, and termination checks into a single state → state transition.

Collision detection uses occupancy grids [height, width] for O(max_n) checks
instead of O(max_n²) pairwise comparisons. Effects are applied via boolean
masks over the sprite arrays (see effects.py for all 37 effect handlers).
"""
import jax
import jax.numpy as jnp
from vgdl_jax.state import GameState
from vgdl_jax.collision import detect_eos, in_bounds
from vgdl_jax.data_model import SpriteClass, STATIC_CLASSES, AVATAR_CLASSES, AABB_THRESHOLD, PHYSICS_CONTINUOUS, PHYSICS_GRAVITY
from vgdl_jax.effects import apply_masked_effect, apply_static_a_effect, POSITION_MODIFYING_EFFECTS, PARTNER_IDX_EFFECTS
from vgdl_jax.sprites import (
    DIRECTION_DELTAS, spawn_sprite, snap_to_pixel_grid,
    update_inertial_avatar, update_mario_avatar,
    update_missile, update_erratic_missile, update_random_npc,
    update_random_inertial, update_spreader, update_chaser,
    update_spawn_point, update_walk_jumper,
    _manhattan_distance_field,
)
from vgdl_jax.terminations import check_all_terminations


def _resolve_slices(state, max_a=None, max_b=None):
    """Resolve per-effect slice sizes from optional max_a/max_b."""
    global_max_n = state.alive.shape[1]
    eff_a = max_a if max_a is not None else global_max_n
    eff_b = max_b if max_b is not None else global_max_n
    return global_max_n, eff_a, eff_b


def _pad_to_global(mask, eff_size, global_max_n):
    """Pad a sliced mask back to [global_max_n]."""
    return jnp.pad(mask, (0, global_max_n - eff_size))


def _pad_partner_to_global(partner_idx, eff_a, global_max_n):
    """Pad a sliced partner_idx array back to [global_max_n], filling with -1."""
    if eff_a < global_max_n:
        return jnp.concatenate([partner_idx, jnp.full(global_max_n - eff_a, -1, dtype=jnp.int32)])
    return partner_idx


def _no_partner(global_max_n):
    """Return an all-(-1) partner_idx array (no partner information)."""
    return jnp.full(global_max_n, -1, dtype=jnp.int32)


def _fractional_neighbor_cells(pos, height, width):
    """Compute clipped floor/ceil cell indices for fractional positions.

    Returns (fr_c, cr_c, fc_c, cc_c) — row-floor, row-ceil, col-floor, col-ceil,
    all clipped to valid grid bounds.
    """
    fr = jnp.floor(pos[:, 0]).astype(jnp.int32)
    fc = jnp.floor(pos[:, 1]).astype(jnp.int32)
    fr_c = jnp.clip(fr, 0, height - 1)
    cr_c = jnp.clip(fr + 1, 0, height - 1)
    fc_c = jnp.clip(fc, 0, width - 1)
    cc_c = jnp.clip(fc + 1, 0, width - 1)
    return fr_c, cr_c, fc_c, cc_c


def _sweep_directions(ipos, iprev):
    """Compute sweep direction and max cells from integer current/previous positions."""
    dir_row = jnp.sign(ipos[:, 0] - iprev[:, 0])
    dir_col = jnp.sign(ipos[:, 1] - iprev[:, 1])
    max_cells = jnp.maximum(
        jnp.abs(ipos[:, 0] - iprev[:, 0]),
        jnp.abs(ipos[:, 1] - iprev[:, 1]))
    return dir_row, dir_col, max_cells


def build_step_fn(effects, terminations, sprite_configs, avatar_config, params,
                  chaser_target_set=frozenset(),
                  static_distance_fields=None,
                  action_map=None):
    """
    Build a jit-compiled step function from compiled game configuration.

    Args:
        effects: list of dicts with keys:
            - type_a, type_b: int type indices
            - is_eos: bool (if True, type_b is ignored, check bounds instead)
            - effect_type: str
            - score_change: int
            - kwargs: dict
        terminations: list of (check_fn, score_change) tuples
        sprite_configs: list of dicts per type with keys:
            - sprite_class: int (SpriteClass enum)
            - cooldown: int
            - flicker_limit: int
            - (class-specific keys: target_type_idx, prob, total, etc.)
        avatar_config: dict with keys:
            - avatar_type_idx, n_move_actions, cooldown, can_shoot,
              shoot_action_idx, projectile_type_idx,
              projectile_orientation_from_avatar, projectile_default_orientation,
              projectile_speed, direction_offset, physics_type
        params: dict with keys:
            - n_types, max_n, height, width
        chaser_target_set: frozenset of target type indices used by chasers/fleeing
        static_distance_fields: dict of target_idx → precomputed [H,W] distance field
            for targets whose positions never change at runtime

    Returns:
        A function step(state, action) → state
    """
    n_types = params['n_types']
    max_n = params['max_n']
    height = params['height']
    width = params['width']
    block_size = params.get('block_size', 0)

    # Distance field caching: separate static (precomputed) from dynamic (per-step)
    _static_distance_fields = static_distance_fields or {}
    _dynamic_chaser_targets = sorted(chaser_target_set - set(_static_distance_fields))

    # Pre-compute cache-safe types for occupancy grid caching.
    # A type_b grid can be cached if its positions never change during effect
    # dispatch (i.e., the type is never type_a in a position-modifying effect).
    _pos_modified_types = set()
    _has_undo_all = False
    for eff in effects:
        if eff.effect_type == 'undo_all':
            _has_undo_all = True
        if eff.effect_type in POSITION_MODIFYING_EFFECTS and not eff.is_eos:
            _pos_modified_types.add(eff.type_a)
    if _has_undo_all:
        _cache_safe_type_b = frozenset()
    else:
        _cache_safe_type_b = frozenset(range(n_types)) - _pos_modified_types

    def _step_inner(state: GameState, action: int) -> GameState:
        # Remap GVGAI action index → internal action index
        if action_map is not None:
            action = action_map[action]

        # Save previous positions for stepBack / undoAll / bounceForward
        prev_positions = state.positions

        # 1. Increment cooldown timers for alive sprites
        state = state.replace(
            cooldown_timers=jnp.where(
                state.alive, state.cooldown_timers + 1,
                state.cooldown_timers))

        # 2. Update avatar (dispatch based on compile-time physics_type)
        if avatar_config.physics_type == PHYSICS_CONTINUOUS:
            state = update_inertial_avatar(
                state, action,
                avatar_type=avatar_config.avatar_type_idx,
                n_move=avatar_config.n_move_actions,
                mass=avatar_config.mass,
                strength=avatar_config.strength)
        elif avatar_config.physics_type == PHYSICS_GRAVITY:
            state = update_mario_avatar(
                state, action,
                avatar_type=avatar_config.avatar_type_idx,
                mass=avatar_config.mass,
                strength=avatar_config.strength,
                jump_strength=avatar_config.jump_strength,
                gravity=avatar_config.gravity,
                airsteering=avatar_config.airsteering)
        elif avatar_config.is_aimed:
            state = _update_aimed_avatar(state, action, avatar_config, height, width,
                                         block_size=block_size)
        elif avatar_config.is_rotating:
            state = _update_rotating_avatar(state, action, avatar_config, height, width,
                                             block_size=block_size)
        else:
            state = _update_avatar(state, action, avatar_config, height, width,
                                   block_size=block_size)

        # 3. Update NPC sprites
        # Precompute distance fields — one per unique target, not per chaser type
        dist_fields = dict(_static_distance_fields)
        for target_idx in _dynamic_chaser_targets:
            dist_fields[target_idx] = _manhattan_distance_field(
                state.positions[target_idx], state.alive[target_idx],
                height, width)

        for type_idx, cfg in enumerate(sprite_configs):
            if cfg.sprite_class in AVATAR_CLASSES:
                continue
            if cfg.sprite_class in STATIC_CLASSES:
                continue
            if cfg.sprite_class in (SpriteClass.CHASER, SpriteClass.FLEEING):
                state = update_chaser(
                    state, type_idx, cfg.target_type_idx, cfg.cooldown,
                    fleeing=(cfg.sprite_class == SpriteClass.FLEEING),
                    height=height, width=width,
                    dist_field=dist_fields.get(cfg.target_type_idx),
                    block_size=block_size)
            else:
                state = _update_npc(state, type_idx, cfg, height, width,
                                    block_size=block_size)


        # 4. Age sprites and kill expired flickers
        new_ages = jnp.where(state.alive, state.ages + 1, state.ages)
        flicker_limits = jnp.array(
            [cfg.flicker_limit for cfg in sprite_configs],
            dtype=jnp.int32)[:, None]  # [n_types, 1]
        flicker_expired = (flicker_limits > 0) & (new_ages >= flicker_limits)
        state = state.replace(
            ages=new_ages,
            alive=state.alive & ~flicker_expired,
        )

        # 5. Apply effects via collision detection + masks
        state = _apply_all_effects(state, prev_positions, effects,
                                   height, width, max_n,
                                   cache_safe_type_b=_cache_safe_type_b)

        # 6. Check terminations (increment step_count first, matching py-vgdl)
        state = state.replace(step_count=state.step_count + 1)
        state, done, win = check_all_terminations(state, terminations)

        state = state.replace(done=done, win=win)
        return state

    def step(state: GameState, action: int) -> GameState:
        # Done-state guard: if already done, return state unchanged
        return jax.lax.cond(
            state.done,
            lambda s, a: s,
            _step_inner,
            state, action)

    return jax.jit(step)


# ── Grid-based collision detection ────────────────────────────────────


def _build_occupancy_grid(positions, alive, height, width):
    """Build a [height, width] boolean grid from sprite positions."""
    # Truncate float32 → int32 (floors toward zero). Correct for grid-physics
    # sprites with integer positions. Fractional-speed pairs use AABB collision
    # instead (see _collision_mask_aabb).
    ipos = positions.astype(jnp.int32)
    ib = in_bounds(ipos, height, width)
    effective = alive & ib
    grid = jnp.zeros((height, width), dtype=jnp.bool_)
    grid = grid.at[ipos[:, 0], ipos[:, 1]].max(effective)
    return grid


def _collision_mask(state, type_a, type_b, height, width,
                    max_a=None, max_b=None, prebuilt_grid_b=None,
                    need_partner=True):
    """Which type_a sprites overlap with any type_b sprite?

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — slot index of colliding type_b sprite (-1 if none)
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    pos_a = state.positions[type_a, :eff_a].astype(jnp.int32)
    alive_a = state.alive[type_a, :eff_a]
    in_bounds_a = in_bounds(pos_a, height, width)
    r = jnp.clip(pos_a[:, 0], 0, height - 1)
    c = jnp.clip(pos_a[:, 1], 0, width - 1)

    if type_a != type_b:
        if prebuilt_grid_b is not None:
            grid_b = prebuilt_grid_b
        else:
            grid_b = _build_occupancy_grid(
                state.positions[type_b, :eff_b], state.alive[type_b, :eff_b],
                height, width)
        mask = grid_b[r, c] & alive_a & in_bounds_a

        if need_partner:
            # Build slot grid: [H, W] int32, last-write-wins for slot index
            pos_b = state.positions[type_b, :eff_b].astype(jnp.int32)
            alive_b = state.alive[type_b, :eff_b]
            ib_b = in_bounds(pos_b, height, width)
            effective_b = alive_b & ib_b
            r_b = jnp.clip(pos_b[:, 0], 0, height - 1)
            c_b = jnp.clip(pos_b[:, 1], 0, width - 1)
            slot_grid = jnp.full((height, width), -1, dtype=jnp.int32)
            slot_indices = jnp.where(effective_b, jnp.arange(eff_b, dtype=jnp.int32), -1)
            slot_grid = slot_grid.at[r_b, c_b].set(slot_indices)
            pidx = jnp.where(mask, slot_grid[r, c], -1)
        else:
            pidx = jnp.full(eff_a, -1, dtype=jnp.int32)
    else:
        counts = jnp.zeros((height, width), dtype=jnp.int32)
        effective = alive_a & in_bounds_a
        counts = counts.at[r, c].add(effective.astype(jnp.int32))
        mask = (counts[r, c] > 1) & effective
        # Self-collision: partner_idx is -1 (no specific partner)
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)

    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


# ── AABB collision detection (for continuous physics) ─────────────────


def _collision_mask_aabb(state, type_a, type_b, height, width,
                         max_a=None, max_b=None, need_partner=True):
    """AABB overlap: two 1x1 sprites overlap when |pos_a - pos_b| < 1.0 on both axes.

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — slot index of first overlapping type_b sprite (-1 if none)
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    pos_a = state.positions[type_a, :eff_a]    # [eff_a, 2] float32
    alive_a = state.alive[type_a, :eff_a]      # [eff_a]
    pos_b = state.positions[type_b, :eff_b]    # [eff_b, 2] float32
    alive_b = state.alive[type_b, :eff_b]      # [eff_b]

    # [eff_a, eff_b, 2]
    diff = jnp.abs(pos_a[:, None, :] - pos_b[None, :, :])
    overlap = jnp.all(diff < AABB_THRESHOLD, axis=-1) & alive_a[:, None] & alive_b[None, :]

    if type_a == type_b:
        overlap = overlap & ~jnp.eye(eff_a, dtype=jnp.bool_)

    mask = jnp.any(overlap, axis=1) & alive_a
    if need_partner:
        # Partner index: argmax of overlap gives first True along axis=1
        # When no overlap, argmax returns 0 — use jnp.where to set to -1
        pidx = jnp.where(mask, jnp.argmax(overlap.astype(jnp.int32), axis=1).astype(jnp.int32), -1)
    else:
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)
    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


# ── Sweep collision detection (for speed > 1) ─────────────────────────


def _build_swept_occupancy_grid(positions, prev_positions, alive,
                                 height, width, max_speed_cells):
    """Build a [H, W] boolean grid marking all cells along each sprite's path."""
    grid = jnp.zeros((height, width), dtype=jnp.bool_)
    iprev = prev_positions.astype(jnp.int32)
    ipos = positions.astype(jnp.int32)
    ib = in_bounds(ipos, height, width)
    effective = alive & ib

    dir_row, dir_col, max_cells = _sweep_directions(ipos, iprev)

    def mark_step(step_i, g):
        r = jnp.clip(iprev[:, 0] + dir_row * step_i, 0, height - 1)
        c = jnp.clip(iprev[:, 1] + dir_col * step_i, 0, width - 1)
        valid = effective & (step_i <= max_cells)
        return g.at[r, c].max(valid)

    return jax.lax.fori_loop(0, max_speed_cells + 1, mark_step, grid)


def _collision_mask_sweep(state, prev_positions, type_a, type_b,
                           height, width, max_speed_cells,
                           max_a=None, max_b=None, need_partner=True):
    """Sweep collision: checks if type_a's path overlaps with type_b's path.

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — slot index of colliding type_b sprite at
        type_a's current position (-1 if none)
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    grid_b = _build_swept_occupancy_grid(
        state.positions[type_b, :eff_b], prev_positions[type_b, :eff_b],
        state.alive[type_b, :eff_b], height, width, max_speed_cells)

    prev_a = prev_positions[type_a, :eff_a]
    alive_a = state.alive[type_a, :eff_a]
    iprev_a = prev_a.astype(jnp.int32)
    ipos_a = state.positions[type_a, :eff_a].astype(jnp.int32)
    in_bounds_a = in_bounds(ipos_a, height, width)

    dir_row, dir_col, max_cells = _sweep_directions(ipos_a, iprev_a)

    def check_step(step_i, hit):
        r = jnp.clip(iprev_a[:, 0] + dir_row * step_i, 0, height - 1)
        c = jnp.clip(iprev_a[:, 1] + dir_col * step_i, 0, width - 1)
        valid = alive_a & in_bounds_a & (step_i <= max_cells)
        return hit | (valid & grid_b[r, c])

    init_hit = jnp.zeros_like(alive_a)
    mask = jax.lax.fori_loop(0, max_speed_cells + 1, check_step, init_hit)

    if need_partner:
        # Build slot grid from type_b's current positions for partner lookup
        pos_b = state.positions[type_b, :eff_b].astype(jnp.int32)
        alive_b = state.alive[type_b, :eff_b]
        ib_b = in_bounds(pos_b, height, width)
        effective_b = alive_b & ib_b
        r_b = jnp.clip(pos_b[:, 0], 0, height - 1)
        c_b = jnp.clip(pos_b[:, 1], 0, width - 1)
        slot_grid = jnp.full((height, width), -1, dtype=jnp.int32)
        slot_indices = jnp.where(effective_b, jnp.arange(eff_b, dtype=jnp.int32), -1)
        slot_grid = slot_grid.at[r_b, c_b].set(slot_indices)
        r_a_clip = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a_clip = jnp.clip(ipos_a[:, 1], 0, width - 1)
        pidx = jnp.where(mask, slot_grid[r_a_clip, c_a_clip], -1)
    else:
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)

    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


# ── Expanded grid collision (fractional vs integer) ────────────────────


def _collision_mask_expanded_grid_a(state, type_a, type_b, height, width,
                                     max_a, max_b, prebuilt_grid_b=None,
                                     need_partner=True):
    """Collision mask when type_a has fractional positions, type_b integer. O(max_a + max_b).

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — slot index of colliding type_b sprite (-1 if none)
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    pos_a = state.positions[type_a, :eff_a]    # [eff_a, 2] float32
    alive_a = state.alive[type_a, :eff_a]      # [eff_a]

    # Build occupancy grid for type_b (integer positions)
    if prebuilt_grid_b is not None:
        grid_b = prebuilt_grid_b
    else:
        grid_b = _build_occupancy_grid(
            state.positions[type_b, :eff_b], state.alive[type_b, :eff_b],
            height, width)

    # For each type_a sprite, check the 4 surrounding cells
    threshold = AABB_THRESHOLD
    r_a = pos_a[:, 0]
    c_a = pos_a[:, 1]
    fr_c, cr_c, fc_c, cc_c = _fractional_neighbor_cells(pos_a, height, width)

    def check_cell(gr, gc):
        has_sprite = grid_b[gr, gc]
        row_diff = jnp.abs(r_a - gr.astype(jnp.float32))
        col_diff = jnp.abs(c_a - gc.astype(jnp.float32))
        return has_sprite & (row_diff < threshold) & (col_diff < threshold)

    hit = (check_cell(fr_c, fc_c) | check_cell(fr_c, cc_c) |
           check_cell(cr_c, fc_c) | check_cell(cr_c, cc_c))
    mask = hit & alive_a

    if need_partner:
        # Build slot grid from type_b integer positions
        pos_b = state.positions[type_b, :eff_b].astype(jnp.int32)
        alive_b = state.alive[type_b, :eff_b]
        ib_b = in_bounds(pos_b, height, width)
        effective_b = alive_b & ib_b
        r_b = jnp.clip(pos_b[:, 0], 0, height - 1)
        c_b = jnp.clip(pos_b[:, 1], 0, width - 1)
        slot_grid = jnp.full((height, width), -1, dtype=jnp.int32)
        slot_indices = jnp.where(effective_b, jnp.arange(eff_b, dtype=jnp.int32), -1)
        slot_grid = slot_grid.at[r_b, c_b].set(slot_indices)
        # Partner from floor cell (primary cell of fractional position)
        pidx = jnp.where(mask, slot_grid[fr_c, fc_c], -1)
    else:
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)

    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


def _collision_mask_expanded_grid_b(state, type_a, type_b, height, width,
                                     max_a, max_b, need_partner=True):
    """Collision mask when type_a has integer positions, type_b fractional. O(max_a + max_b).

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — slot index of colliding type_b sprite (-1 if none)
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    pos_b = state.positions[type_b, :eff_b]    # [eff_b, 2] float32
    alive_b = state.alive[type_b, :eff_b]      # [eff_b]

    # For each type_b sprite, mark up to 4 cells it overlaps via AABB
    threshold = AABB_THRESHOLD
    r_b = pos_b[:, 0]
    c_b = pos_b[:, 1]
    fr_c, cr_c, fc_c, cc_c = _fractional_neighbor_cells(pos_b, height, width)

    coll_grid = jnp.zeros((height, width), dtype=jnp.bool_)

    def mark_cell(g, gr, gc):
        row_diff = jnp.abs(r_b - gr.astype(jnp.float32))
        col_diff = jnp.abs(c_b - gc.astype(jnp.float32))
        overlap = (row_diff < threshold) & (col_diff < threshold) & alive_b
        return g.at[gr, gc].max(overlap)

    coll_grid = mark_cell(coll_grid, fr_c, fc_c)
    coll_grid = mark_cell(coll_grid, fr_c, cc_c)
    coll_grid = mark_cell(coll_grid, cr_c, fc_c)
    coll_grid = mark_cell(coll_grid, cr_c, cc_c)

    # Type_a looks up its own integer cell
    pos_a = state.positions[type_a, :eff_a]
    alive_a = state.alive[type_a, :eff_a]
    ipos_a = pos_a.astype(jnp.int32)
    r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
    c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
    ib_a = in_bounds(ipos_a, height, width)
    mask = coll_grid[r_a, c_a] & alive_a & ib_a

    if need_partner:
        # Build slot grid from type_b's floor positions
        slot_grid = jnp.full((height, width), -1, dtype=jnp.int32)
        slot_indices = jnp.where(alive_b, jnp.arange(eff_b, dtype=jnp.int32), -1)
        slot_grid = slot_grid.at[fr_c, fc_c].set(slot_indices)
        pidx = jnp.where(mask, slot_grid[r_a, c_a], -1)
    else:
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)

    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


# ── Static grid collision detection ──────────────────────────────────


def _collision_mask_static_b_grid(state, type_a, static_grid, height, width,
                                   max_a=None):
    """Collision when type_b is a static grid, type_a has integer positions.

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — always -1 (no slot info for static grids)
    """
    global_max_n, eff_a, _ = _resolve_slices(state, max_a)
    pos_a = state.positions[type_a, :eff_a].astype(jnp.int32)
    alive_a = state.alive[type_a, :eff_a]
    ib_a = in_bounds(pos_a, height, width)
    r = jnp.clip(pos_a[:, 0], 0, height - 1)
    c = jnp.clip(pos_a[:, 1], 0, width - 1)
    mask = static_grid[r, c] & alive_a & ib_a
    return (_pad_to_global(mask, eff_a, global_max_n),
            _no_partner(global_max_n))


def _collision_mask_static_b_expanded(state, type_a, static_grid,
                                       height, width, max_a=None):
    """Collision when type_b is a static grid, type_a has fractional positions.

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — always -1 (no slot info for static grids)
    """
    global_max_n, eff_a, _ = _resolve_slices(state, max_a)
    pos_a = state.positions[type_a, :eff_a]
    alive_a = state.alive[type_a, :eff_a]

    threshold = AABB_THRESHOLD
    r_a = pos_a[:, 0]
    c_a = pos_a[:, 1]
    fr_c, cr_c, fc_c, cc_c = _fractional_neighbor_cells(pos_a, height, width)

    def check_cell(gr, gc):
        has_sprite = static_grid[gr, gc]
        row_diff = jnp.abs(r_a - gr.astype(jnp.float32))
        col_diff = jnp.abs(c_a - gc.astype(jnp.float32))
        return has_sprite & (row_diff < threshold) & (col_diff < threshold)

    hit = (check_cell(fr_c, fc_c) | check_cell(fr_c, cc_c) |
           check_cell(cr_c, fc_c) | check_cell(cr_c, cc_c))
    mask = hit & alive_a
    return (_pad_to_global(mask, eff_a, global_max_n),
            _no_partner(global_max_n))


def _collision_grid_mask_static_a(state, static_grid, type_b,
                                   height, width, max_b=None):
    """Returns [H, W] bool grid of static cells overlapping any type_b sprite."""
    _, _, eff_b = _resolve_slices(state, max_b=max_b)
    pos_b = state.positions[type_b, :eff_b].astype(jnp.int32)
    alive_b = state.alive[type_b, :eff_b]
    ib_b = in_bounds(pos_b, height, width)
    r_b = jnp.clip(pos_b[:, 0], 0, height - 1)
    c_b = jnp.clip(pos_b[:, 1], 0, width - 1)
    effective_b = alive_b & ib_b
    # Build occupancy grid for type_b
    occ_b = jnp.zeros((height, width), dtype=jnp.bool_)
    occ_b = occ_b.at[r_b, c_b].max(effective_b)
    # AND with static grid → cells where static type overlaps type_b
    return static_grid & occ_b


# ── Effect application ────────────────────────────────────────────────


def _apply_all_effects(state, prev_positions, effects, height, width, max_n,
                       cache_safe_type_b=frozenset()):
    """Apply all effects using collision detection and mask operations."""
    once_guard = {}  # (effect_type, type_idx) → [max_n] bool — once-per-step
    grid_cache = {}  # type_b → [H, W] bool occupancy grid (reused across effects)
    for eff in effects:
        type_a = eff.type_a
        effect_type = eff.effect_type
        score_change = eff.score_change
        kwargs = eff.kwargs
        eff_max_a = eff.max_a
        eff_max_b = eff.max_b
        static_a_idx = eff.static_a_grid_idx
        static_b_idx = eff.static_b_grid_idx

        if eff.is_eos:
            eos_mask = detect_eos(
                state.positions[type_a], state.alive[type_a], height, width)
            state = apply_masked_effect(
                state, prev_positions, type_a, -1, eos_mask,
                effect_type, score_change, kwargs, height, width, max_n)
        else:
            type_b = eff.type_b
            collision_mode = eff.collision_mode
            max_speed_cells = eff.max_speed_cells

            # ── Static type_a: collision produces [H,W] grid mask ──
            if static_a_idx is not None and static_b_idx is not None:
                # Both static — skip (self-collision of immovables is a no-op)
                continue
            elif static_a_idx is not None:
                grid_mask = _collision_grid_mask_static_a(
                    state, state.static_grids[static_a_idx], type_b,
                    height, width, max_b=eff_max_b)
                state = apply_static_a_effect(
                    state, static_a_idx, type_b, grid_mask,
                    effect_type, score_change, kwargs, height, width)
                continue

            # ── Collision detection (single dispatch chain) ──
            need_partner = effect_type in PARTNER_IDX_EFFECTS

            # For grid/expanded_grid_a: reuse cached occupancy grid
            cached_grid_b = None
            if collision_mode in ('grid', 'expanded_grid_a') and type_a != type_b:
                if type_b in grid_cache:
                    cached_grid_b = grid_cache[type_b]
                elif type_b in cache_safe_type_b:
                    cached_grid_b = _build_occupancy_grid(
                        state.positions[type_b, :eff_max_b],
                        state.alive[type_b, :eff_max_b], height, width)
                    grid_cache[type_b] = cached_grid_b

            if collision_mode == 'static_b_grid':
                coll_mask, partner_idx = _collision_mask_static_b_grid(
                    state, type_a, state.static_grids[static_b_idx],
                    height, width, max_a=eff_max_a)
            elif collision_mode == 'static_b_expanded':
                coll_mask, partner_idx = _collision_mask_static_b_expanded(
                    state, type_a, state.static_grids[static_b_idx],
                    height, width, max_a=eff_max_a)
            elif collision_mode == 'sweep':
                coll_mask, partner_idx = _collision_mask_sweep(
                    state, prev_positions, type_a, type_b,
                    height, width, max_speed_cells,
                    max_a=eff_max_a, max_b=eff_max_b,
                    need_partner=need_partner)
            elif collision_mode == 'expanded_grid_a':
                coll_mask, partner_idx = _collision_mask_expanded_grid_a(
                    state, type_a, type_b, height, width,
                    eff_max_a, eff_max_b,
                    prebuilt_grid_b=cached_grid_b,
                    need_partner=need_partner)
            elif collision_mode == 'expanded_grid_b':
                coll_mask, partner_idx = _collision_mask_expanded_grid_b(
                    state, type_a, type_b, height, width,
                    eff_max_a, eff_max_b,
                    need_partner=need_partner)
            elif collision_mode == 'aabb':
                coll_mask, partner_idx = _collision_mask_aabb(
                    state, type_a, type_b, height, width,
                    max_a=eff_max_a, max_b=eff_max_b,
                    need_partner=need_partner)
            else:
                coll_mask, partner_idx = _collision_mask(
                    state, type_a, type_b, height, width,
                    max_a=eff_max_a, max_b=eff_max_b,
                    prebuilt_grid_b=cached_grid_b,
                    need_partner=need_partner)

            # once-per-step guards: wallStop, wallBounce, pullWithIt
            if effect_type in ('wall_stop', 'wall_bounce', 'partner_delta'):
                key = (effect_type, type_a)
                already = once_guard.get(key, jnp.zeros(max_n, dtype=jnp.bool_))
                coll_mask = coll_mask & ~already
                once_guard[key] = already | coll_mask

            # For effects that need to know about static type_b, pass grid idx
            eff_kwargs = kwargs
            if static_b_idx is not None and effect_type in (
                    'kill_both', 'wall_stop', 'wall_bounce', 'bounce_direction'):
                eff_kwargs = dict(kwargs, static_b_grid_idx=static_b_idx)

            state = apply_masked_effect(
                state, prev_positions, type_a, type_b, coll_mask,
                effect_type, score_change, eff_kwargs, height, width, max_n,
                max_a=eff_max_a, max_b=eff_max_b,
                partner_idx=partner_idx)

    return state


# ── Avatar and NPC update ─────────────────────────────────────────────


def _update_avatar_single(state, action, cfg, avatar_type, height, width, block_size=0):
    """Update a single avatar type's position and optionally shoot."""
    n_move = cfg.n_move_actions
    cooldown = cfg.cooldown
    direction_offset = cfg.direction_offset

    # Movement
    is_move = action < n_move
    # Apply direction_offset so HorizontalAvatar maps actions 0,1 to LEFT,RIGHT
    move_idx = jnp.clip(action + direction_offset, 0, 3)
    delta = jax.lax.cond(
        is_move,
        lambda: DIRECTION_DELTAS[move_idx],
        lambda: jnp.array([0.0, 0.0], dtype=jnp.float32))

    can_move = state.cooldown_timers[avatar_type, 0] >= cooldown
    # Gate on alive: only move if this avatar type is alive
    is_alive = state.alive[avatar_type, 0]
    should_move = is_move & can_move & is_alive
    speed = state.speeds[avatar_type, 0]
    new_pos = state.positions[avatar_type, 0] + delta * speed * should_move
    if block_size > 0:
        new_pos = snap_to_pixel_grid(new_pos, block_size)

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        cooldown_timers=jnp.where(
            should_move,
            state.cooldown_timers.at[avatar_type, 0].set(0),
            state.cooldown_timers),
    )

    # Update orientation only if actually moved (not just is_move)
    new_ori = jax.lax.cond(
        should_move,
        lambda: DIRECTION_DELTAS[move_idx].astype(jnp.float32),
        lambda: state.orientations[avatar_type, 0])
    state = state.replace(
        orientations=state.orientations.at[avatar_type, 0].set(new_ori))

    # Shoot
    if cfg.can_shoot:
        is_shoot = (action == cfg.shoot_action_idx) & is_alive
        proj_type = cfg.projectile_type_idx
        proj_speed = cfg.projectile_speed

        if cfg.shoot_everywhere:
            # ShootEverywhereAvatar: fire in all 4 cardinal directions
            def _shoot_everywhere(s):
                for i in range(4):
                    s = spawn_sprite(s, avatar_type, 0, proj_type,
                                     DIRECTION_DELTAS[i], proj_speed)
                return s
            state = jax.lax.cond(
                is_shoot, _shoot_everywhere, lambda s: s, state)
        else:
            if cfg.projectile_orientation_from_avatar:
                proj_ori = state.orientations[avatar_type, 0]
            else:
                proj_ori = jnp.array(cfg.projectile_default_orientation,
                                      dtype=jnp.float32)

            state = jax.lax.cond(
                is_shoot,
                lambda s: spawn_sprite(s, avatar_type, 0, proj_type,
                                        proj_ori, proj_speed),
                lambda s: s,
                state,
            )

    return state


def _update_avatar(state, action, cfg, height, width, block_size=0):
    """Update avatar position and optionally shoot.

    When there are multiple avatar subtypes (cfg.avatar_type_indices has >1
    entry), iterates over all at compile time. Only the alive subtype moves.
    """
    avatar_types = cfg.avatar_type_indices
    if len(avatar_types) <= 1:
        # Single avatar type (common case)
        return _update_avatar_single(
            state, action, cfg, cfg.avatar_type_idx, height, width, block_size)
    else:
        # Multiple avatar subtypes — update whichever is alive
        for at in avatar_types:
            state = _update_avatar_single(
                state, action, cfg, at, height, width, block_size)
        return state


def _maybe_shoot(state, action, cfg, avatar_type):
    """Conditionally spawn projectile using avatar's current orientation."""
    is_shoot = (action == cfg.shoot_action_idx)
    proj_type = cfg.projectile_type_idx
    proj_speed = cfg.projectile_speed
    proj_ori = state.orientations[avatar_type, 0]
    return jax.lax.cond(
        is_shoot,
        lambda s: spawn_sprite(s, avatar_type, 0, proj_type, proj_ori, proj_speed),
        lambda s: s,
        state,
    )


def _update_aimed_avatar(state, action, cfg, height, width, block_size=0):
    """Update AimedAvatar / AimedFlakAvatar: continuous-angle aiming + optional horizontal movement.

    AimedAvatar: AIM_UP=0, AIM_DOWN=1, SHOOT=n_move, NOOP=n_move+1
    AimedFlakAvatar: LEFT=0, RIGHT=1, AIM_UP=2, AIM_DOWN=3, SHOOT=4, NOOP=5
    """
    avatar_type = cfg.avatar_type_idx
    angle_diff = cfg.angle_diff
    can_move = cfg.can_move_aimed
    n_move = cfg.n_move_actions

    ori = state.orientations[avatar_type, 0]
    pos = state.positions[avatar_type, 0]

    if can_move:
        # AimedFlakAvatar: actions 0,1 = LEFT,RIGHT; 2,3 = AIM_UP,AIM_DOWN
        is_left = (action == 0)
        is_right = (action == 1)
        is_aim_up = (action == 2)
        is_aim_down = (action == 3)
        h_delta = jnp.where(is_left, -1.0, jnp.where(is_right, 1.0, 0.0))
        new_pos = pos.at[1].add(h_delta)
        if block_size > 0:
            new_pos = snap_to_pixel_grid(new_pos, block_size)
    else:
        # AimedAvatar: actions 0,1 = AIM_UP,AIM_DOWN
        is_aim_up = (action == 0)
        is_aim_down = (action == 1)
        new_pos = pos

    # Apply 2D rotation to orientation
    # AIM_UP: rotate by -angle_diff (CCW), AIM_DOWN: rotate by +angle_diff (CW)
    # Rotation matrix: [[cos, -sin], [sin, cos]]
    # Our orientation is (row, col) = (-sin(theta), cos(theta)) for rightward = (0, 1)
    theta = jnp.where(is_aim_up, -angle_diff,
                      jnp.where(is_aim_down, angle_diff, 0.0))
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    new_ori_r = cos_t * ori[0] - sin_t * ori[1]
    new_ori_c = sin_t * ori[0] + cos_t * ori[1]
    new_ori = jnp.array([new_ori_r, new_ori_c])

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )

    if cfg.can_shoot:
        state = _maybe_shoot(state, action, cfg, avatar_type)

    return state


def _update_rotating_avatar(state, action, cfg, height, width, block_size=0):
    """Update rotating avatar: ego-centric forward/backward + rotation.

    Actions:
        0: thrust forward (move 1 step in current orientation)
        1: thrust backward (RotatingAvatar) or flip 180 (Flipping variants)
        2: rotate CCW
        3: rotate CW
        4+ or NOOP: no-op
    """
    avatar_type = cfg.avatar_type_idx
    is_flipping = cfg.is_flipping
    noise_level = cfg.noise_level

    # Optionally apply noise
    if noise_level > 0:
        rng, key = jax.random.split(state.rng)
        # With probability noise_level, replace action with random
        noisy = jax.random.uniform(key) < noise_level
        rng, key2 = jax.random.split(rng)
        rand_action = jax.random.randint(key2, (), 0, cfg.n_move_actions + 1)
        action = jnp.where(noisy, rand_action, action)
        state = state.replace(rng=rng)

    ori = state.orientations[avatar_type, 0]

    # Find current orientation index in DIRECTION_DELTAS
    # UP=0, DOWN=1, LEFT=2, RIGHT=3
    diffs = jnp.sum(jnp.abs(DIRECTION_DELTAS - ori), axis=-1)
    ori_idx = jnp.argmin(diffs)

    # Action 0: forward — move in current orientation
    is_forward = (action == 0)
    fwd_pos = state.positions[avatar_type, 0] + ori * is_forward

    # Action 1: backward (non-flipping) or flip (flipping)
    is_action1 = (action == 1)
    if is_flipping:
        # Flip: rotate 180 degrees, no movement
        # BASEDIRS cycle: UP(0), DOWN(1), LEFT(2), RIGHT(3)
        # 180 flip: UP↔DOWN, LEFT↔RIGHT → index XOR with specific mapping
        # Actually use: (ori_idx + 2) wouldn't work with our BASEDIRS order
        # UP=0→DOWN=1, DOWN=1→UP=0, LEFT=2→RIGHT=3, RIGHT=3→LEFT=2
        flipped_idx = jnp.array([1, 0, 3, 2])[ori_idx]
        new_ori_flip = DIRECTION_DELTAS[flipped_idx]
        new_ori = jnp.where(is_action1, new_ori_flip, ori)
        new_pos = jnp.where(is_forward, fwd_pos, state.positions[avatar_type, 0])
    else:
        # Backward: move opposite to current orientation
        bwd_pos = state.positions[avatar_type, 0] - ori * is_action1
        new_pos = jnp.where(is_forward, fwd_pos,
                           jnp.where(is_action1, bwd_pos,
                                    state.positions[avatar_type, 0]))
        new_ori = ori

    if block_size > 0:
        new_pos = snap_to_pixel_grid(new_pos, block_size)

    # Action 2: CCW rotation
    # In our BASEDIRS: UP=0,DOWN=1,LEFT=2,RIGHT=3
    # CCW: UP→LEFT→DOWN→RIGHT→UP
    ccw_map = jnp.array([2, 3, 1, 0])  # UP→LEFT, DOWN→RIGHT, LEFT→DOWN, RIGHT→UP
    is_ccw = (action == 2)
    ccw_idx = ccw_map[ori_idx]
    new_ori = jnp.where(is_ccw, DIRECTION_DELTAS[ccw_idx], new_ori)

    # Action 3: CW rotation
    # CW: UP→RIGHT→DOWN→LEFT→UP
    cw_map = jnp.array([3, 2, 0, 1])  # UP→RIGHT, DOWN→LEFT, LEFT→UP, RIGHT→DOWN
    is_cw = (action == 3)
    cw_idx = cw_map[ori_idx]
    new_ori = jnp.where(is_cw, DIRECTION_DELTAS[cw_idx], new_ori)

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )

    if cfg.can_shoot:
        state = _maybe_shoot(state, action, cfg, avatar_type)

    return state


def _npc_spawn_point(state, type_idx, cfg, height, width, block_size):
    return update_spawn_point(
        state, type_idx, cfg.cooldown, prob=cfg.prob, total=cfg.total,
        target_type=cfg.target_type_idx,
        target_orientation=jnp.array(cfg.target_orientation, dtype=jnp.float32),
        target_speed=cfg.target_speed)


def _npc_bomber(state, type_idx, cfg, height, width, block_size):
    state = update_missile(state, type_idx, cfg.cooldown, block_size=block_size)
    return update_spawn_point(
        state, type_idx, cooldown=cfg.spawn_cooldown, prob=cfg.prob,
        total=cfg.total, target_type=cfg.target_type_idx,
        target_orientation=jnp.array(cfg.target_orientation, dtype=jnp.float32),
        target_speed=cfg.target_speed)


_NPC_UPDATERS = {
    SpriteClass.MISSILE: lambda s, ti, cfg, h, w, bs: update_missile(s, ti, cfg.cooldown, block_size=bs),
    SpriteClass.ERRATIC_MISSILE: lambda s, ti, cfg, h, w, bs: update_erratic_missile(s, ti, cfg.cooldown, prob=cfg.prob, block_size=bs),
    SpriteClass.RANDOM_NPC: lambda s, ti, cfg, h, w, bs: update_random_npc(s, ti, cfg.cooldown, cons=cfg.cons, block_size=bs),
    # CHASER and FLEEING handled inline in _step_inner with precomputed dist_fields
    SpriteClass.FLICKER: lambda s, ti, cfg, h, w, bs: s,
    SpriteClass.ORIENTED_FLICKER: lambda s, ti, cfg, h, w, bs: s,
    SpriteClass.SPAWN_POINT: _npc_spawn_point,
    SpriteClass.BOMBER: _npc_bomber,
    SpriteClass.WALKER: lambda s, ti, cfg, h, w, bs: update_missile(s, ti, cfg.cooldown, block_size=bs),
    SpriteClass.SPREADER: lambda s, ti, cfg, h, w, bs: update_spreader(s, ti, spreadprob=cfg.spreadprob),
    SpriteClass.RANDOM_INERTIAL: lambda s, ti, cfg, h, w, bs: update_random_inertial(s, ti, mass=cfg.mass, strength=cfg.strength),
    SpriteClass.RANDOM_MISSILE: lambda s, ti, cfg, h, w, bs: update_missile(s, ti, cfg.cooldown, block_size=bs),
    SpriteClass.WALK_JUMPER: lambda s, ti, cfg, h, w, bs: update_walk_jumper(s, ti, prob=cfg.prob, strength=cfg.strength, gravity=cfg.gravity, mass=cfg.mass),
}


def _update_npc(state, type_idx, cfg, height, width, block_size=0):
    """Update a single NPC type based on its sprite class."""
    updater = _NPC_UPDATERS.get(cfg.sprite_class)
    if updater is not None:
        return updater(state, type_idx, cfg, height, width, block_size)
    return state
