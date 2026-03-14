"""
The jittable step function: combines sprite updates, collision detection,
effects, and termination checks into a single state → state transition.

Collision detection uses occupancy grids [height, width] for O(max_n) checks
instead of O(max_n²) pairwise comparisons. Effects are applied via boolean
masks over the sprite arrays (see effects.py for all 37 effect handlers).

All positions are int32 pixel coordinates. Collision grids are at cell
resolution (pos // block_size). For sprites that can be mid-cell (fractional
speed), pixel_aabb collision checks |pos_a - pos_b| < block_size on each axis.
"""
import jax
import jax.numpy as jnp
from vgdl_jax.state import GameState
from vgdl_jax.collision import detect_eos, in_bounds
from vgdl_jax.data_model import SpriteClass, STATIC_CLASSES, AVATAR_CLASSES, PHYSICS_CONTINUOUS, PHYSICS_GRAVITY
from vgdl_jax.effects import apply_masked_effect, apply_static_a_effect, POSITION_MODIFYING_EFFECTS, PARTNER_IDX_EFFECTS
from vgdl_jax.sprites import (
    DIRECTION_DELTAS, DIRECTION_DELTAS_F32, spawn_sprite,
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


def _sweep_directions(ipos, iprev):
    """Compute sweep direction and max cells from integer current/previous cell positions."""
    dir_row = jnp.sign(ipos[:, 0] - iprev[:, 0])
    dir_col = jnp.sign(ipos[:, 1] - iprev[:, 1])
    max_cells = jnp.maximum(
        jnp.abs(ipos[:, 0] - iprev[:, 0]),
        jnp.abs(ipos[:, 1] - iprev[:, 1]))
    return dir_row, dir_col, max_cells


def _prepare_type_b_cells(state, type_b, eff_b, height, width, block_size):
    """Convert type_b pixel positions to clipped cell coordinates and effective mask.

    Returns (r_b, c_b, effective_b) where:
      r_b, c_b: [eff_b] int32 clipped cell coordinates
      effective_b: [eff_b] bool (alive & in bounds)
    """
    cell_b = state.positions[type_b, :eff_b] // block_size
    alive_b = state.alive[type_b, :eff_b]
    ib_b = in_bounds(cell_b, height, width)
    effective_b = alive_b & ib_b
    r_b = jnp.clip(cell_b[:, 0], 0, height - 1)
    c_b = jnp.clip(cell_b[:, 1], 0, width - 1)
    return r_b, c_b, effective_b


def _build_slot_grid(effective_b, eff_b, r_b, c_b, height, width):
    """Build [H, W] int32 grid mapping each cell to the slot index of the sprite there.

    Returns slot_grid where slot_grid[r, c] is the slot index of the type_b sprite
    at that cell (-1 if empty).
    """
    slot_grid = jnp.full((height, width), -1, dtype=jnp.int32)
    slot_indices = jnp.where(effective_b, jnp.arange(eff_b, dtype=jnp.int32), -1)
    return slot_grid.at[r_b, c_b].set(slot_indices)


def _pre_movement(state, type_idx):
    """GVGAI VGDLSprite.preMovement(): increment lastmove (cooldown_timers) for alive sprites.

    Called per-type, right before that type's update(), matching GVGAI's per-sprite
    preMovement() → update() pattern.
    """
    return state.replace(
        cooldown_timers=state.cooldown_timers.at[type_idx].set(
            jnp.where(state.alive[type_idx],
                      state.cooldown_timers[type_idx] + 1,
                      state.cooldown_timers[type_idx])))


def _pre_spawn(state, type_idx):
    """Increment spawn_timers for SpawnPoint/Bomber types.

    Spawn timing is independent of movement timing (RC3). In GVGAI, spawn
    readiness uses (start+gameTick)%cooldown while movement uses lastmove —
    two independent systems.
    """
    return state.replace(
        spawn_timers=state.spawn_timers.at[type_idx].set(
            jnp.where(state.alive[type_idx],
                      state.spawn_timers[type_idx] + 1,
                      state.spawn_timers[type_idx])))


def build_step_fn(effects, terminations, sprite_configs, avatar_config, params,
                  chaser_target_set=frozenset(),
                  static_distance_fields=None,
                  action_map=None):
    """
    Build a jit-compiled step function from compiled game configuration.

    Args:
        effects: list of CompiledEffect dataclasses
        terminations: list of (check_fn, score_change) tuples
        sprite_configs: list of SpriteConfig per type
        avatar_config: AvatarConfig dataclass
        params: dict with n_types, max_n, height, width, block_size
        chaser_target_set: frozenset of target type indices used by chasers/fleeing
        static_distance_fields: dict of target_idx → precomputed [H,W] distance field
        action_map: JAX int32 array mapping GVGAI action → internal action

    Returns:
        A function step(state, action) → state
    """
    n_types = params['n_types']
    max_n = params['max_n']
    height = params['height']
    width = params['width']
    block_size = params.get('block_size', 1)

    # Distance field caching
    _static_distance_fields = static_distance_fields or {}
    _dynamic_chaser_targets = sorted(chaser_target_set - set(_static_distance_fields))

    # Pre-compute cache-safe types for occupancy grid caching.
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

        # 1. Avatar: preMovement then update (matching GVGAI tick() lines 1365-1371)
        for at in avatar_config.avatar_type_indices:
            state = _pre_movement(state, at)

        if avatar_config.physics_type == PHYSICS_CONTINUOUS:
            state = update_inertial_avatar(
                state, action,
                avatar_type=avatar_config.avatar_type_indices[0],
                n_move=avatar_config.n_move_actions,
                mass=avatar_config.mass,
                strength=avatar_config.strength)
        elif avatar_config.physics_type == PHYSICS_GRAVITY:
            state = update_mario_avatar(
                state, action,
                avatar_type=avatar_config.avatar_type_indices[0],
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

        # 2. NPC sprites: preMovement then update, in REVERSE order
        # (matching GVGAI tick() lines 1377-1387: reverse spriteOrder ensures
        # producers process before children, enabling same-tick processing)

        # Precompute distance fields — one per unique target, not per chaser type
        dist_fields = dict(_static_distance_fields)
        for target_idx in _dynamic_chaser_targets:
            dist_fields[target_idx] = _manhattan_distance_field(
                state.positions[target_idx], state.alive[target_idx],
                height, width, block_size)

        for type_idx in range(len(sprite_configs) - 1, -1, -1):
            cfg = sprite_configs[type_idx]
            if cfg.sprite_class in AVATAR_CLASSES:
                continue
            if cfg.sprite_class in STATIC_CLASSES:
                continue
            state = _pre_movement(state, type_idx)
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
            dtype=jnp.int32)[:, None]
        flicker_expired = (flicker_limits > 0) & (new_ages >= flicker_limits)
        state = state.replace(
            ages=new_ages,
            alive=state.alive & ~flicker_expired,
        )

        # 5. Apply effects via collision detection + masks
        state = _apply_all_effects(state, prev_positions, effects,
                                   height, width, max_n, block_size,
                                   cache_safe_type_b=_cache_safe_type_b)

        # 6. Check terminations (increment step_count first, matching GVGAI)
        state = state.replace(step_count=state.step_count + 1)
        state, done, win = check_all_terminations(state, terminations)

        state = state.replace(done=done, win=win)
        return state

    def step(state: GameState, action: int) -> GameState:
        return jax.lax.cond(
            state.done,
            lambda s, a: s,
            _step_inner,
            state, action)

    return jax.jit(step)


# ── Grid-based collision detection ────────────────────────────────────


def _build_occupancy_grid(positions, alive, height, width, block_size):
    """Build a [height, width] boolean grid from sprite pixel positions.

    Converts pixel positions to cells via integer division by block_size.
    """
    cells = positions // block_size  # int32
    ib = in_bounds(cells, height, width)
    effective = alive & ib
    r = jnp.clip(cells[:, 0], 0, height - 1)
    c = jnp.clip(cells[:, 1], 0, width - 1)
    grid = jnp.zeros((height, width), dtype=jnp.bool_)
    grid = grid.at[r, c].max(effective)
    return grid


def _collision_mask(state, type_a, type_b, height, width, block_size,
                    max_a=None, max_b=None, prebuilt_grid_b=None,
                    need_partner=True):
    """Which type_a sprites overlap with any type_b sprite (same cell)?

    For sprites that are always cell-aligned (speed_px % block_size == 0),
    same-cell check is exact. For mid-cell sprites, use pixel_aabb instead.

    Returns (mask, partner_idx) where:
      mask: [global_max_n] bool
      partner_idx: [global_max_n] int32 — slot index of colliding type_b sprite (-1 if none)
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    cell_a = state.positions[type_a, :eff_a] // block_size  # [eff_a, 2] int32
    alive_a = state.alive[type_a, :eff_a]
    in_bounds_a = in_bounds(cell_a, height, width)
    r = jnp.clip(cell_a[:, 0], 0, height - 1)
    c = jnp.clip(cell_a[:, 1], 0, width - 1)

    if type_a != type_b:
        if prebuilt_grid_b is not None:
            grid_b = prebuilt_grid_b
        else:
            grid_b = _build_occupancy_grid(
                state.positions[type_b, :eff_b], state.alive[type_b, :eff_b],
                height, width, block_size)
        mask = grid_b[r, c] & alive_a & in_bounds_a

        if need_partner:
            r_b, c_b, effective_b = _prepare_type_b_cells(
                state, type_b, eff_b, height, width, block_size)
            slot_grid = _build_slot_grid(effective_b, eff_b, r_b, c_b, height, width)
            pidx = jnp.where(mask, slot_grid[r, c], -1)
        else:
            pidx = jnp.full(eff_a, -1, dtype=jnp.int32)
    else:
        counts = jnp.zeros((height, width), dtype=jnp.int32)
        effective = alive_a & in_bounds_a
        counts = counts.at[r, c].add(effective.astype(jnp.int32))
        mask = (counts[r, c] > 1) & effective
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)

    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


# ── Pixel AABB collision (for mid-cell sprites) ──────────────────────


def _collision_mask_pixel_aabb(state, type_a, type_b, height, width, block_size,
                                max_a=None, max_b=None, need_partner=True):
    """Pixel AABB overlap: two block_size×block_size sprites overlap when
    |pos_a - pos_b| < block_size on both axes.

    Used for sprite pairs where at least one has non-cell-aligned speed
    (speed_px % block_size != 0), producing mid-cell pixel positions.

    Returns (mask, partner_idx).
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    pos_a = state.positions[type_a, :eff_a]    # [eff_a, 2] int32
    alive_a = state.alive[type_a, :eff_a]
    pos_b = state.positions[type_b, :eff_b]    # [eff_b, 2] int32
    alive_b = state.alive[type_b, :eff_b]

    # [eff_a, eff_b, 2]
    diff = jnp.abs(pos_a[:, None, :] - pos_b[None, :, :])
    overlap = jnp.all(diff < block_size, axis=-1) & alive_a[:, None] & alive_b[None, :]

    if type_a == type_b:
        overlap = overlap & ~jnp.eye(eff_a, dtype=jnp.bool_)

    mask = jnp.any(overlap, axis=1) & alive_a
    if need_partner:
        pidx = jnp.where(mask, jnp.argmax(overlap.astype(jnp.int32), axis=1).astype(jnp.int32), -1)
    else:
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)
    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


# ── Sweep collision detection (for speed > block_size) ─────────────────


def _build_swept_occupancy_grid(positions, prev_positions, alive,
                                 height, width, block_size, max_speed_cells):
    """Build a [H, W] boolean grid marking all cells along each sprite's path."""
    grid = jnp.zeros((height, width), dtype=jnp.bool_)
    iprev = prev_positions // block_size
    ipos = positions // block_size
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
                           height, width, block_size, max_speed_cells,
                           max_a=None, max_b=None, need_partner=True):
    """Sweep collision: checks if type_a's cell path overlaps with type_b's cell path.

    Returns (mask, partner_idx).
    """
    global_max_n, eff_a, eff_b = _resolve_slices(state, max_a, max_b)

    grid_b = _build_swept_occupancy_grid(
        state.positions[type_b, :eff_b], prev_positions[type_b, :eff_b],
        state.alive[type_b, :eff_b], height, width, block_size, max_speed_cells)

    prev_a = prev_positions[type_a, :eff_a]
    alive_a = state.alive[type_a, :eff_a]
    iprev_a = prev_a // block_size
    ipos_a = state.positions[type_a, :eff_a] // block_size
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
        r_b, c_b, effective_b = _prepare_type_b_cells(
            state, type_b, eff_b, height, width, block_size)
        slot_grid = _build_slot_grid(effective_b, eff_b, r_b, c_b, height, width)
        r_a_clip = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a_clip = jnp.clip(ipos_a[:, 1], 0, width - 1)
        pidx = jnp.where(mask, slot_grid[r_a_clip, c_a_clip], -1)
    else:
        pidx = jnp.full(eff_a, -1, dtype=jnp.int32)

    return (_pad_to_global(mask, eff_a, global_max_n),
            _pad_partner_to_global(pidx, eff_a, global_max_n))


# ── Static grid collision detection ──────────────────────────────────


def _collision_mask_static_b_grid(state, type_a, static_grid, height, width,
                                   block_size, max_a=None):
    """Collision when type_b is a static grid, type_a has pixel positions.

    For cell-aligned sprites: simple cell lookup.
    For mid-cell sprites: check the 2×2 cell region the bounding box covers,
    with pixel AABB verification to avoid false positives.

    Returns (mask, partner_idx) — partner_idx always -1 for static grids.
    """
    global_max_n, eff_a, _ = _resolve_slices(state, max_a)
    pos_a = state.positions[type_a, :eff_a]  # [eff_a, 2] int32
    alive_a = state.alive[type_a, :eff_a]

    # 2×2 cell region that the block_size×block_size bounding box covers
    min_r = pos_a[:, 0] // block_size
    max_r = (pos_a[:, 0] + block_size - 1) // block_size
    min_c = pos_a[:, 1] // block_size
    max_c = (pos_a[:, 1] + block_size - 1) // block_size

    def check_cell(gr, gc):
        gr_c = jnp.clip(gr, 0, height - 1)
        gc_c = jnp.clip(gc, 0, width - 1)
        has_sprite = static_grid[gr_c, gc_c]
        # Pixel AABB verification: |pos_a - cell_origin| < block_size
        row_diff = jnp.abs(pos_a[:, 0] - gr * block_size)
        col_diff = jnp.abs(pos_a[:, 1] - gc * block_size)
        return has_sprite & (row_diff < block_size) & (col_diff < block_size)

    hit = (check_cell(min_r, min_c) | check_cell(min_r, max_c) |
           check_cell(max_r, min_c) | check_cell(max_r, max_c))

    # Bounds check: at least one cell must be valid
    any_in_bounds = ((min_r >= 0) & (min_r < height) & (min_c >= 0) & (min_c < width)) | \
                    ((max_r >= 0) & (max_r < height) & (max_c >= 0) & (max_c < width))
    mask = hit & alive_a & any_in_bounds

    return (_pad_to_global(mask, eff_a, global_max_n),
            _no_partner(global_max_n))


def _collision_grid_mask_static_a(state, static_grid, type_b,
                                   height, width, block_size, max_b=None):
    """Returns [H, W] bool grid of static cells overlapping any type_b sprite."""
    _, _, eff_b = _resolve_slices(state, max_b=max_b)
    r_b, c_b, effective_b = _prepare_type_b_cells(
        state, type_b, eff_b, height, width, block_size)
    occ_b = jnp.zeros((height, width), dtype=jnp.bool_)
    occ_b = occ_b.at[r_b, c_b].max(effective_b)
    return static_grid & occ_b


# ── Effect application ────────────────────────────────────────────────


def _apply_all_effects(state, prev_positions, effects, height, width, max_n,
                       block_size, cache_safe_type_b=frozenset()):
    """Apply all effects using collision detection and mask operations."""
    once_guard = {}
    grid_cache = {}
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
                state.positions[type_a], state.alive[type_a],
                height, width, block_size)
            state = apply_masked_effect(
                state, prev_positions, type_a, -1, eos_mask,
                effect_type, score_change, kwargs, height, width, max_n,
                block_size=block_size)
        else:
            type_b = eff.type_b
            collision_mode = eff.collision_mode
            max_speed_cells = eff.max_speed_cells

            # ── Static type_a: collision produces [H,W] grid mask ──
            if static_a_idx is not None and static_b_idx is not None:
                continue
            elif static_a_idx is not None:
                grid_mask = _collision_grid_mask_static_a(
                    state, state.static_grids[static_a_idx], type_b,
                    height, width, block_size, max_b=eff_max_b)
                state = apply_static_a_effect(
                    state, static_a_idx, type_b, grid_mask,
                    effect_type, score_change, kwargs, height, width,
                    block_size=block_size)
                continue

            # ── Collision detection ──
            need_partner = effect_type in PARTNER_IDX_EFFECTS

            # Occupancy grid cache for grid/pixel_aabb modes
            cached_grid_b = None
            if collision_mode == 'grid' and type_a != type_b:
                if type_b in grid_cache:
                    cached_grid_b = grid_cache[type_b]
                elif type_b in cache_safe_type_b:
                    cached_grid_b = _build_occupancy_grid(
                        state.positions[type_b, :eff_max_b],
                        state.alive[type_b, :eff_max_b],
                        height, width, block_size)
                    grid_cache[type_b] = cached_grid_b

            if collision_mode == 'static_b_grid':
                coll_mask, partner_idx = _collision_mask_static_b_grid(
                    state, type_a, state.static_grids[static_b_idx],
                    height, width, block_size, max_a=eff_max_a)
            elif collision_mode == 'sweep':
                coll_mask, partner_idx = _collision_mask_sweep(
                    state, prev_positions, type_a, type_b,
                    height, width, block_size, max_speed_cells,
                    max_a=eff_max_a, max_b=eff_max_b,
                    need_partner=need_partner)
            elif collision_mode == 'pixel_aabb':
                coll_mask, partner_idx = _collision_mask_pixel_aabb(
                    state, type_a, type_b, height, width, block_size,
                    max_a=eff_max_a, max_b=eff_max_b,
                    need_partner=need_partner)
            else:
                coll_mask, partner_idx = _collision_mask(
                    state, type_a, type_b, height, width, block_size,
                    max_a=eff_max_a, max_b=eff_max_b,
                    prebuilt_grid_b=cached_grid_b,
                    need_partner=need_partner)

            # once-per-step guards
            if effect_type in ('wall_stop', 'wall_bounce', 'partner_delta'):
                key = (effect_type, type_a)
                already = once_guard.get(key, jnp.zeros(max_n, dtype=jnp.bool_))
                coll_mask = coll_mask & ~already
                once_guard[key] = already | coll_mask

            eff_kwargs = kwargs
            if static_b_idx is not None and effect_type in (
                    'kill_both', 'wall_stop', 'wall_bounce', 'bounce_direction'):
                eff_kwargs = dict(kwargs, static_b_grid_idx=static_b_idx)

            state = apply_masked_effect(
                state, prev_positions, type_a, type_b, coll_mask,
                effect_type, score_change, eff_kwargs, height, width, max_n,
                max_a=eff_max_a, max_b=eff_max_b,
                partner_idx=partner_idx, block_size=block_size)

    return state


# ── Avatar and NPC update ─────────────────────────────────────────────


def _update_avatar_single(state, action, cfg, avatar_type, height, width, block_size=1):
    """Update a single avatar type's position and optionally shoot."""
    n_move = cfg.n_move_actions
    cooldown = cfg.cooldown
    direction_offset = cfg.direction_offset

    # Movement
    is_move = action < n_move
    move_idx = jnp.clip(action + direction_offset, 0, 3)
    delta = jax.lax.cond(
        is_move,
        lambda: DIRECTION_DELTAS[move_idx],
        lambda: jnp.array([0, 0], dtype=jnp.int32))

    can_move = state.cooldown_timers[avatar_type, 0] >= cooldown
    is_alive = state.alive[avatar_type, 0]

    if cfg.rotate_in_place:
        # RC2: GVGAI OrientedAvatar rotateInPlace semantics.
        # If action direction differs from current orientation: rotate only (no move).
        # If action direction matches current orientation: move normally.
        cur_ori_int = state.orientations[avatar_type, 0].astype(jnp.int32)
        ori_matches = jnp.all(DIRECTION_DELTAS[move_idx] == cur_ori_int)
        should_move = is_move & can_move & is_alive & ori_matches
        # Always update orientation on any directional input (even without moving)
        should_rotate = is_move & is_alive
    else:
        should_move = is_move & can_move & is_alive
        should_rotate = should_move

    speed = state.speeds[avatar_type, 0]  # int32 pixel displacement
    new_pos = state.positions[avatar_type, 0] + delta * speed * should_move.astype(jnp.int32)

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        cooldown_timers=jnp.where(
            should_move,
            state.cooldown_timers.at[avatar_type, 0].set(0),
            state.cooldown_timers),
    )

    # Update orientation on rotate (RC2: any directional input; non-RC2: only on move)
    new_ori = jax.lax.cond(
        should_rotate,
        lambda: DIRECTION_DELTAS_F32[move_idx],
        lambda: state.orientations[avatar_type, 0])
    state = state.replace(
        orientations=state.orientations.at[avatar_type, 0].set(new_ori))

    # Shoot
    if cfg.can_shoot:
        is_shoot = (action == cfg.shoot_action_idx) & is_alive
        proj_type = cfg.projectile_type_idx
        proj_speed = cfg.projectile_speed
        if cfg.projectile_singleton:
            is_shoot = is_shoot & ~jnp.any(state.alive[proj_type])

        # RC8: ammo check — FlakAvatar/AimedFlakAvatar need ammo to shoot
        if cfg.ammo_resource_idx >= 0:
            ammo_count = state.resources[avatar_type, 0, cfg.ammo_resource_idx]
            if cfg.min_ammo >= 0:
                has_ammo = ammo_count > cfg.min_ammo
            else:
                has_ammo = ammo_count > 0
            is_shoot = is_shoot & has_ammo

        if cfg.shoot_everywhere:
            if cfg.projectile_offset:
                def _shoot_everywhere(s):
                    for i in range(4):
                        proj_pos = s.positions[avatar_type, 0] + DIRECTION_DELTAS[i] * block_size
                        s = spawn_sprite(s, avatar_type, 0, proj_type,
                                         DIRECTION_DELTAS_F32[i], proj_speed,
                                         pos_override=proj_pos)
                    return s
            else:
                def _shoot_everywhere(s):
                    for i in range(4):
                        s = spawn_sprite(s, avatar_type, 0, proj_type,
                                         DIRECTION_DELTAS_F32[i], proj_speed)
                    return s
            state = jax.lax.cond(
                is_shoot, _shoot_everywhere, lambda s: s, state)
        else:
            if cfg.projectile_orientation_from_avatar:
                proj_ori = state.orientations[avatar_type, 0]
            else:
                proj_ori = jnp.array(cfg.projectile_default_orientation,
                                      dtype=jnp.float32)

            if cfg.projectile_offset:
                # RC4: ShootAvatar spawns projectile one cell ahead
                proj_pos = state.positions[avatar_type, 0] + \
                    state.orientations[avatar_type, 0].astype(jnp.int32) * block_size
                state = jax.lax.cond(
                    is_shoot,
                    lambda s: spawn_sprite(s, avatar_type, 0, proj_type,
                                            proj_ori, proj_speed,
                                            pos_override=proj_pos),
                    lambda s: s,
                    state,
                )
            else:
                state = jax.lax.cond(
                    is_shoot,
                    lambda s: spawn_sprite(s, avatar_type, 0, proj_type,
                                            proj_ori, proj_speed),
                    lambda s: s,
                    state,
                )

        # RC8: subtract ammo cost after shooting
        if cfg.ammo_resource_idx >= 0:
            new_ammo = state.resources[avatar_type, 0, cfg.ammo_resource_idx] - cfg.ammo_cost
            state = state.replace(
                resources=jnp.where(
                    is_shoot,
                    state.resources.at[avatar_type, 0, cfg.ammo_resource_idx].set(new_ammo),
                    state.resources))

    return state


def _update_avatar(state, action, cfg, height, width, block_size=1):
    """Update avatar position and optionally shoot."""
    for at in cfg.avatar_type_indices:
        state = _update_avatar_single(
            state, action, cfg, at, height, width, block_size)
    return state


def _maybe_shoot(state, action, cfg, avatar_type):
    """Conditionally spawn projectile using avatar's current orientation."""
    is_shoot = (action == cfg.shoot_action_idx)
    proj_type = cfg.projectile_type_idx
    proj_speed = cfg.projectile_speed
    proj_ori = state.orientations[avatar_type, 0]
    if cfg.projectile_singleton:
        # Singleton: only spawn if no projectile of this type is alive
        is_shoot = is_shoot & ~jnp.any(state.alive[proj_type])
    # RC8: ammo check
    if cfg.ammo_resource_idx >= 0:
        ammo_count = state.resources[avatar_type, 0, cfg.ammo_resource_idx]
        if cfg.min_ammo >= 0:
            has_ammo = ammo_count > cfg.min_ammo
        else:
            has_ammo = ammo_count > 0
        is_shoot = is_shoot & has_ammo
    state = jax.lax.cond(
        is_shoot,
        lambda s: spawn_sprite(s, avatar_type, 0, proj_type, proj_ori, proj_speed),
        lambda s: s,
        state,
    )
    # RC8: subtract ammo cost after shooting
    if cfg.ammo_resource_idx >= 0:
        new_ammo = state.resources[avatar_type, 0, cfg.ammo_resource_idx] - cfg.ammo_cost
        state = state.replace(
            resources=jnp.where(
                is_shoot,
                state.resources.at[avatar_type, 0, cfg.ammo_resource_idx].set(new_ammo),
                state.resources))
    return state


def _update_aimed_avatar(state, action, cfg, height, width, block_size=1):
    """Update AimedAvatar / AimedFlakAvatar."""
    avatar_type = cfg.avatar_type_indices[0]
    angle_diff = cfg.angle_diff
    can_move = cfg.can_move_aimed
    n_move = cfg.n_move_actions

    ori = state.orientations[avatar_type, 0]
    pos = state.positions[avatar_type, 0]

    if can_move:
        is_left = (action == 0)
        is_right = (action == 1)
        is_aim_up = (action == 2)
        is_aim_down = (action == 3)
        # Move by block_size pixels (one cell) left/right
        h_delta = jnp.where(is_left, -block_size, jnp.where(is_right, block_size, 0))
        new_pos = pos.at[1].add(h_delta)
    else:
        is_aim_up = (action == 0)
        is_aim_down = (action == 1)
        new_pos = pos

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


def _update_rotating_avatar(state, action, cfg, height, width, block_size=1):
    """Update rotating avatar: ego-centric forward/backward + rotation."""
    avatar_type = cfg.avatar_type_indices[0]
    is_flipping = cfg.is_flipping
    noise_level = cfg.noise_level

    if noise_level > 0:
        rng, key = jax.random.split(state.rng)
        noisy = jax.random.uniform(key) < noise_level
        rng, key2 = jax.random.split(rng)
        rand_action = jax.random.randint(key2, (), 0, cfg.n_move_actions + 1)
        action = jnp.where(noisy, rand_action, action)
        state = state.replace(rng=rng)

    ori = state.orientations[avatar_type, 0]

    # Find current orientation index in DIRECTION_DELTAS (float→int matching)
    diffs = jnp.sum(jnp.abs(DIRECTION_DELTAS_F32 - ori), axis=-1)
    ori_idx = jnp.argmin(diffs)

    speed = state.speeds[avatar_type, 0]  # int32 pixel displacement

    # Action 0: forward
    is_forward = (action == 0)
    ori_int = ori.astype(jnp.int32)
    fwd_pos = state.positions[avatar_type, 0] + ori_int * speed * is_forward.astype(jnp.int32)

    # Action 1: backward (non-flipping) or flip (flipping)
    is_action1 = (action == 1)
    if is_flipping:
        flipped_idx = jnp.array([1, 0, 3, 2])[ori_idx]
        new_ori_flip = DIRECTION_DELTAS_F32[flipped_idx]
        new_ori = jnp.where(is_action1, new_ori_flip, ori)
        new_pos = jnp.where(is_forward, fwd_pos, state.positions[avatar_type, 0])
    else:
        bwd_pos = state.positions[avatar_type, 0] - ori_int * speed * is_action1.astype(jnp.int32)
        new_pos = jnp.where(is_forward, fwd_pos,
                           jnp.where(is_action1, bwd_pos,
                                    state.positions[avatar_type, 0]))
        new_ori = ori

    # Action 2: CCW rotation
    ccw_map = jnp.array([2, 3, 1, 0])
    is_ccw = (action == 2)
    ccw_idx = ccw_map[ori_idx]
    new_ori = jnp.where(is_ccw, DIRECTION_DELTAS_F32[ccw_idx], new_ori)

    # Action 3: CW rotation
    cw_map = jnp.array([3, 2, 0, 1])
    is_cw = (action == 3)
    cw_idx = cw_map[ori_idx]
    new_ori = jnp.where(is_cw, DIRECTION_DELTAS_F32[cw_idx], new_ori)

    state = state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )

    if cfg.can_shoot:
        state = _maybe_shoot(state, action, cfg, avatar_type)

    return state


def _npc_spawn_point(state, type_idx, cfg, height, width, block_size):
    state = _pre_spawn(state, type_idx)
    return update_spawn_point(
        state, type_idx, cfg.spawn_cooldown, prob=cfg.prob, total=cfg.total,
        target_type=cfg.target_type_idx,
        target_orientation=jnp.array(cfg.target_orientation, dtype=jnp.float32),
        target_speed=cfg.target_speed,
        target_singleton=cfg.target_singleton)


def _npc_bomber(state, type_idx, cfg, height, width, block_size):
    # GVGAI SpawnPoint.update(): spawn first, then super.update() does movement
    # spawn_timers and cooldown_timers are independent (RC3)
    state = _pre_spawn(state, type_idx)
    state = update_spawn_point(
        state, type_idx, cooldown=cfg.spawn_cooldown, prob=cfg.prob,
        total=cfg.total, target_type=cfg.target_type_idx,
        target_orientation=jnp.array(cfg.target_orientation, dtype=jnp.float32),
        target_speed=cfg.target_speed,
        target_singleton=cfg.target_singleton)
    return update_missile(state, type_idx, cfg.cooldown)


_NPC_UPDATERS = {
    SpriteClass.MISSILE: lambda s, ti, cfg, h, w, bs: update_missile(s, ti, cfg.cooldown),
    SpriteClass.ERRATIC_MISSILE: lambda s, ti, cfg, h, w, bs: update_erratic_missile(s, ti, cfg.cooldown, prob=cfg.prob),
    SpriteClass.RANDOM_NPC: lambda s, ti, cfg, h, w, bs: update_random_npc(s, ti, cfg.cooldown, cons=cfg.cons),
    SpriteClass.FLICKER: lambda s, ti, cfg, h, w, bs: s,
    SpriteClass.ORIENTED_FLICKER: lambda s, ti, cfg, h, w, bs: s,
    SpriteClass.SPAWN_POINT: _npc_spawn_point,
    SpriteClass.BOMBER: _npc_bomber,
    SpriteClass.WALKER: lambda s, ti, cfg, h, w, bs: update_missile(s, ti, cfg.cooldown),
    SpriteClass.SPREADER: lambda s, ti, cfg, h, w, bs: update_spreader(s, ti, spreadprob=cfg.spreadprob, block_size=bs),
    SpriteClass.RANDOM_INERTIAL: lambda s, ti, cfg, h, w, bs: update_random_inertial(s, ti, mass=cfg.mass, strength=cfg.strength),
    SpriteClass.RANDOM_MISSILE: lambda s, ti, cfg, h, w, bs: update_missile(s, ti, cfg.cooldown),
    SpriteClass.WALK_JUMPER: lambda s, ti, cfg, h, w, bs: update_walk_jumper(s, ti, prob=cfg.prob, strength=cfg.strength, gravity=cfg.gravity, mass=cfg.mass),
}


def _update_npc(state, type_idx, cfg, height, width, block_size=1):
    """Update a single NPC type based on its sprite class."""
    updater = _NPC_UPDATERS.get(cfg.sprite_class)
    if updater is not None:
        return updater(state, type_idx, cfg, height, width, block_size)
    return state
