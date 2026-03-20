"""Effect handlers for the VGDL-JAX step function.

Each handler implements one collision effect (kill, transform, resource change,
etc.). Handlers share a uniform keyword-argument interface — each declares the
params it needs and absorbs the rest via **_.

Sections:
    - JAX Primitives (prim_*)
    - Shared helpers
    - Kill / removal
    - Position and orientation
    - Resources
    - Spawn and transform
    - Movement and conveying
    - Physics / wall interactions
    - Dispatch (EFFECT_DISPATCH dict + apply_masked_effect)
"""
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional, Callable
from vgdl_jax.collision import in_bounds
from vgdl_jax.data_model import (
    DEFAULT_RESOURCE_LIMIT, MOVING_NPC_CLASSES,
    SpriteClass, speed_to_pixels,
)
from vgdl_jax.sprites import DIRECTION_DELTAS, prefix_sum_allocate


@dataclass(frozen=True)
class EffectEntry:
    handler: Callable
    key: str
    needs_partner: bool = False
    modifies_position: bool = False
    modifies_alive: bool = False
    static_a_handler: Optional[Callable] = None
    compile_kwargs: Optional[Callable] = None  # fn(ed, ctx) → kwargs dict

@dataclass(frozen=True)
class CompileContext:
    """Bundle of compile-time context passed to per-effect compile functions."""
    game_def: object
    resource_name_to_idx: object
    resource_limits: object
    avatar_type_idx: int
    concrete_actor_idx: Optional[int] = None
    concrete_actee_idx: Optional[int] = None
    resolve_first: Callable = field(default=None)
    block_size: int = 1

    def __post_init__(self):
        if self.resolve_first is None:
            object.__setattr__(self, 'resolve_first', lambda gd, st, d=None: d)


# ── JAX Primitives ───────────────────────────────────────────────────

def prim_kill(state, type_idx, mask):
    """Kill sprites where mask is True."""
    return state.replace(alive=state.alive.at[type_idx].set(state.alive[type_idx] & ~mask))


def prim_kill_partner(state, type_b, partner_idx, actor_mask):
    """Scatter-kill: kill type_b sprites that are partners of masked type_a sprites."""
    max_b = state.alive.shape[1]
    safe_idx = jnp.clip(partner_idx, 0, max_b - 1)
    b_kill = jnp.zeros(max_b, dtype=jnp.bool_).at[safe_idx].max(actor_mask & (partner_idx >= 0))
    return state.replace(alive=state.alive.at[type_b].set(state.alive[type_b] & ~b_kill))


def prim_restore_pos(state, type_idx, mask, prev_positions):
    """Restore positions from prev_positions where mask is True."""
    new = jnp.where(mask[:, None], prev_positions[type_idx], state.positions[type_idx])
    return state.replace(positions=state.positions.at[type_idx].set(new))


def prim_move(state, type_idx, mask, delta):
    """Add delta to positions where mask is True."""
    new = state.positions[type_idx] + mask[:, None] * delta
    return state.replace(positions=state.positions.at[type_idx].set(new))


def prim_set_orientation(state, type_idx, mask, new_ori):
    """Set orientations where mask is True."""
    cur = state.orientations[type_idx]
    new = jnp.where(mask[:, None], new_ori, cur)
    return state.replace(orientations=state.orientations.at[type_idx].set(new))


def prim_negate_orientation(state, type_idx, mask):
    """Negate orientations where mask is True."""
    cur = state.orientations[type_idx]
    new = jnp.where(mask[:, None], -cur, cur)
    return state.replace(orientations=state.orientations.at[type_idx].set(new))


def prim_clear_static(state, sg_idx, clear_mask):
    """Clear cells from a static grid."""
    return state.replace(static_grids=state.static_grids.at[sg_idx].set(
        state.static_grids[sg_idx] & ~clear_mask))


def _with_score(state, score_delta):
    """Add score_delta to state score."""
    return state.replace(score=state.score + score_delta)


def _partner_vals(field_slice, partner_idx):
    """Safely index field_slice[partner_idx] with clipping. field_slice: [max_n, ...]"""
    safe = jnp.clip(partner_idx, 0, field_slice.shape[0] - 1)
    return field_slice[safe]


def _partner_scatter_mask(actor_mask, partner_idx, max_b):
    """Build [max_b] bool mask: True for type_b slots that are partners of masked actors."""
    return jnp.zeros(max_b, dtype=jnp.bool_).at[jnp.clip(partner_idx, 0, max_b - 1)].max(
        actor_mask & (partner_idx >= 0))


# ── Shared helpers ──────────────────────────────────────────────────────

def _nearest_partner(pos_a, state, type_b, eff_b):
    """Find nearest alive type_b sprite for each pos_a sprite. Returns [eff_a, 2] positions."""
    pos_b = state.positions[type_b, :eff_b]
    alive_b = state.alive[type_b, :eff_b]
    diff = pos_a[:, None, :] - pos_b[None, :, :]
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    dist_sq = jnp.where(alive_b[None, :], dist_sq, 1e10)
    nearest_b = jnp.argmin(dist_sq, axis=1)
    return pos_b[nearest_b]


def _fill_slots(state, target_type, source_mask, src_positions,
                src_orientations=None, target_speed=None, reset_cooldown=False,
                src_resources=None, target_cons=0, target_spawn_cd=0):
    """Allocate dead slots in target_type and fill with source data.

    Effect-spawned sprites get is_first_tick=True and cooldown_timers=0.
    In GVGAI, effects happen after the tick loop, so spawned sprites don't
    get same-tick processing — their isFirstTick is consumed on the next tick.

    target_cons: if > 0, set direction_ticks=cons and orientation=(0,0) for
        newly filled slots. Matches GVGAI's addSprite() which creates from
        template with counter=0, prevAction=DNONE — the first `cons` calls
        to getRandomMove() produce DNONE (no movement).

    target_spawn_cd: if > 0, the target is a SpawnPoint/Bomber. Initialize
        spawn_timers based on step_count to match GVGAI's (start+gameTick)%cd
        formula. Effects run after the NPC loop, so the spawned sprite's first
        update is on the NEXT tick (step_count+1). start = step_count+1 in
        GVGAI terms, so first fire iff (2*(step_count+1))%cd == 0.
    """
    should_fill, src_idx = prefix_sum_allocate(state.alive[target_type], source_mask)
    src_pos = src_positions[src_idx]
    state = state.replace(
        alive=state.alive.at[target_type].set(state.alive[target_type] | should_fill),
        positions=state.positions.at[target_type].set(
            jnp.where(should_fill[:, None], src_pos, state.positions[target_type])),
        ages=state.ages.at[target_type].set(
            jnp.where(should_fill, 0, state.ages[target_type])),
        is_first_tick=state.is_first_tick.at[target_type].set(
            state.is_first_tick[target_type] | should_fill),
    )
    if target_cons > 0:
        # RandomNPC cons init: overrides src_orientations — GVGAI addSprite()
        # sets prevAction=DNONE regardless of source orientation.
        state = state.replace(
            direction_ticks=state.direction_ticks.at[target_type].set(
                jnp.where(should_fill, target_cons,
                           state.direction_ticks[target_type])),
            orientations=state.orientations.at[target_type].set(
                jnp.where(should_fill[:, None],
                           jnp.zeros(2, dtype=jnp.float32),
                           state.orientations[target_type])))
    elif src_orientations is not None:
        src_ori = src_orientations[src_idx]
        state = state.replace(
            orientations=state.orientations.at[target_type].set(
                jnp.where(should_fill[:, None], src_ori, state.orientations[target_type])))
    if target_speed is not None:
        state = state.replace(
            speeds=state.speeds.at[target_type].set(
                jnp.where(should_fill, target_speed, state.speeds[target_type])))
    if reset_cooldown:
        state = state.replace(
            cooldown_timers=state.cooldown_timers.at[target_type].set(
                jnp.where(should_fill, 0, state.cooldown_timers[target_type])))
    if src_resources is not None:
        src_res = src_resources[src_idx]
        state = state.replace(
            resources=state.resources.at[target_type].set(
                jnp.where(should_fill[:, None], src_res, state.resources[target_type])))
    else:
        # Reset resources to 0 for newly spawned sprites. GVGAI addSprite()
        # creates sprites with empty resource maps — stale resources in a
        # reused dead slot must not carry over.
        state = state.replace(
            resources=state.resources.at[target_type].set(
                jnp.where(should_fill[:, None], 0, state.resources[target_type])))
    if target_spawn_cd > 0:
        # GVGAI: effect-spawned Bomber/SpawnPoint gets start=-1 (loadDefaults).
        # First update at step_count+1 sets start=step_count+1.
        # Fire when (2*(step_count+1))%cd == 0.
        # Init: cd - 1 - ((2*(step_count+1)) % cd)
        next_tick = state.step_count + 1
        init_val = target_spawn_cd - 1 - ((2 * next_tick) % target_spawn_cd)
        state = state.replace(
            spawn_timers=state.spawn_timers.at[target_type].set(
                jnp.where(should_fill, init_val,
                           state.spawn_timers[target_type])))
    return state


# ── Kill / removal ─────────────────────────────────────────────────────

def kill_sprite(state, type_a, mask, score_delta, **_):
    return _with_score(prim_kill(state, type_a, mask), score_delta)


def kill_both(state, type_a, type_b, mask, score_delta, height, width,
              kwargs=None, partner_idx=None, **_):
    state = prim_kill(state, type_a, mask)
    if kwargs is None:
        kwargs = {}
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    if static_b_grid_idx is not None:
        ipos_a = state.positions[type_a].astype(jnp.int32)
        r_a = jnp.clip(ipos_a[:, 0], 0, height - 1)
        c_a = jnp.clip(ipos_a[:, 1], 0, width - 1)
        grid_coll = jnp.zeros((height, width), dtype=jnp.bool_).at[r_a, c_a].max(mask)
        state = prim_clear_static(state, static_b_grid_idx, grid_coll)
    elif type_b >= 0:
        state = prim_kill_partner(state, type_b, partner_idx, mask)
    return _with_score(state, score_delta)


def kill_if_slow(state, type_a, mask, score_change, kwargs, **_):
    limitspeed = kwargs.get('limitspeed', 0.0)
    should_kill = mask & (state.speeds[type_a] < limitspeed)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_others(state, type_a, mask, score_delta, kwargs, **_):
    kill_type = kwargs.get('kill_type_idx', -1)
    if kill_type >= 0:
        kill_all = jnp.broadcast_to(mask.any(), state.alive[kill_type].shape)
        state = prim_kill(state, kill_type, kill_all)
    return _with_score(state, score_delta)


def kill_if_from_above(state, prev_positions, type_a, type_b, mask,
                       score_change, partner_idx=None, **_):
    if type_b >= 0:
        icurr_b = _partner_vals(state.positions[type_b], partner_idx).astype(jnp.int32)
        iprev_b = _partner_vals(prev_positions[type_b], partner_idx).astype(jnp.int32)
        should_kill = mask & (partner_idx >= 0) & (icurr_b[:, 0] > iprev_b[:, 0])
    else:
        should_kill = jnp.zeros_like(mask)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_has_less(state, type_a, mask, score_change, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    should_kill = mask & (state.resources[type_a, :, r_idx] <= limit)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_has_more(state, type_a, mask, score_change, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    should_kill = mask & (state.resources[type_a, :, r_idx] >= limit)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_other_has_more(state, type_a, type_b, mask, score_change,
                           kwargs, partner_idx=None, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        partner_res = _partner_vals(state.resources[type_b, :, r_idx], partner_idx)
        partner_res = jnp.where(partner_idx >= 0, partner_res, 0)
        should_kill = mask & jnp.greater_equal(partner_res, limit)
    else:
        should_kill = jnp.zeros_like(mask)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_other_has_less(state, type_a, type_b, mask, score_change,
                           kwargs, partner_idx=None, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        partner_res = _partner_vals(state.resources[type_b, :, r_idx], partner_idx)
        partner_res = jnp.where(partner_idx >= 0, partner_res, -1)
        should_kill = mask & (partner_res >= 0) & (partner_res <= limit)
    else:
        should_kill = jnp.zeros_like(mask)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_avatar_without_resource(state, type_a, mask, score_change, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    should_kill = mask & ~(state.resources[ati, 0, r_idx] > 0)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_all(state, type_a, mask, score_delta, kwargs, **_):
    """Kill all sprites of the target type when any collision occurs."""
    kill_type = kwargs.get('kill_type_idx', -1)
    if kill_type >= 0:
        all_mask = jnp.broadcast_to(mask.any(), state.alive[kill_type].shape)
        state = prim_kill(state, kill_type, all_mask)
    return _with_score(state, score_delta)


def _kill_if_frontal_core(state, prev_positions, type_a, type_b, mask,
                          score_change, partner_idx, frontal):
    """Shared logic for killIfFrontal / killIfNotFrontal.

    frontal=True: kill when sprites move in opposite directions (or sprite1 is static).
    frontal=False: kill when sprites do NOT move in opposite directions (or sprite1 is static).
    """
    if type_b >= 0 and partner_idx is not None:
        delta_a = (state.positions[type_a] - prev_positions[type_a]).astype(jnp.float32)
        norm_a = jnp.sqrt(jnp.sum(delta_a ** 2, axis=-1, keepdims=True))
        dir_a = jnp.where(norm_a > 0.5, delta_a / norm_a, 0.0)
        partner_delta = _partner_vals(
            (state.positions[type_b] - prev_positions[type_b]).astype(jnp.float32),
            partner_idx)
        norm_b = jnp.sqrt(jnp.sum(partner_delta ** 2, axis=-1, keepdims=True))
        dir_b = jnp.where(norm_b > 0.5, partner_delta / norm_b, 0.0)
        # Sum of normalized directions — if (0,0) they are opposite
        sum_dir = dir_a + dir_b
        sum_mag = jnp.sqrt(jnp.sum(sum_dir ** 2, axis=-1))
        a_is_static = (norm_a[:, 0] < 0.5)
        is_opposite = sum_mag < 0.5
        direction_match = is_opposite if frontal else ~is_opposite
        should_kill = mask & (partner_idx >= 0) & (a_is_static | direction_match)
    else:
        should_kill = jnp.zeros_like(mask)
    return _with_score(prim_kill(state, type_a, should_kill),
                       should_kill.sum() * jnp.int32(score_change))


def kill_if_frontal(state, prev_positions, type_a, type_b, mask, score_change,
                    partner_idx=None, **_):
    """Kill sprite1 if sprites are moving in opposite directions (or sprite1 is static)."""
    return _kill_if_frontal_core(state, prev_positions, type_a, type_b, mask,
                                 score_change, partner_idx, frontal=True)


def kill_if_not_frontal(state, prev_positions, type_a, type_b, mask, score_change,
                        partner_idx=None, **_):
    """Kill sprite1 if sprites are NOT moving in opposite directions (or sprite1 is static)."""
    return _kill_if_frontal_core(state, prev_positions, type_a, type_b, mask,
                                 score_change, partner_idx, frontal=False)


def transform_to_singleton(state, type_a, mask, score_delta, kwargs, **_):
    """Like transformTo, but first transforms existing sprites of target type to stype_other,
    then transforms sprite1 to stype. Only executes if collision occurs."""
    new_type = kwargs['new_type_idx']
    other_type = kwargs.get('other_type_idx', -1)
    target_speed = kwargs.get('target_speed', None)
    target_cons = kwargs.get('target_cons', 0)
    target_spawn_cd = kwargs.get('target_spawn_cd', 0)
    copy_ori = kwargs.get('copy_orientation', True)
    other_speed = kwargs.get('other_speed', None)
    other_cons = kwargs.get('other_cons', 0)

    any_collision = mask.any()

    # Step 1: Transform existing sprites of new_type back to other_type
    if other_type >= 0:
        existing_alive = state.alive[new_type] & any_collision
        kill_all = jnp.broadcast_to(any_collision, state.alive[new_type].shape)
        state = prim_kill(state, new_type, kill_all)
        state = _fill_slots(state, other_type, existing_alive,
                            state.positions[new_type],
                            state.orientations[new_type],
                            target_speed=other_speed, reset_cooldown=True,
                            src_resources=state.resources[new_type],
                            target_cons=other_cons)

    # Step 2: Transform sprite1 to new_type (same as normal transformTo)
    if copy_ori:
        orientations = state.orientations[type_a]
    else:
        default_ori = kwargs.get('default_orientation', (0.0, 1.0))
        orientations = jnp.broadcast_to(
            jnp.array(default_ori, dtype=jnp.float32),
            state.orientations[type_a].shape)
    state = prim_kill(state, type_a, mask)
    state = _fill_slots(state, new_type, mask,
                        state.positions[type_a], orientations,
                        target_speed=target_speed, reset_cooldown=True,
                        src_resources=state.resources[type_a],
                        target_cons=target_cons,
                        target_spawn_cd=target_spawn_cd)
    return _with_score(state, score_delta)


def kill_if_alive(state, type_a, type_b, mask, score_change,
                  partner_idx=None, **_):
    """Kill sprite1 only if partner sprite2 is still alive."""
    if partner_idx is not None and type_b >= 0:
        partner_alive = state.alive[type_b][jnp.clip(partner_idx, 0, state.alive.shape[1] - 1)]
        valid = partner_idx >= 0
        mask = mask & partner_alive & valid
    return _with_score(prim_kill(state, type_a, mask),
                       mask.sum() * jnp.int32(score_change))


# ── Position and orientation ───────────────────────────────────────────

def step_back(state, prev_positions, type_a, mask, score_delta, **_):
    return _with_score(prim_restore_pos(state, type_a, mask, prev_positions), score_delta)


def reverse_direction(state, type_a, mask, score_delta, **_):
    return _with_score(prim_negate_orientation(state, type_a, mask), score_delta)


def turn_around(state, prev_positions, type_a, mask, score_delta,
                height, width, block_size=1, **_):
    # GVGAI TurnAround: restore position, call activeMovement(DDOWN) twice
    # (which adds speed in the DOWN direction each time), then negate orientation.
    # GVGAI does NOT clip the resulting position — the sprite can overshoot
    # the screen boundary (EOS detection handles it later if needed).
    state = prim_restore_pos(state, type_a, mask, prev_positions)
    pos = state.positions[type_a]
    speed_px = state.speeds[type_a]  # [max_n] int32
    # 2 * speed_px downward (row += 2*speed, col unchanged)
    displacement = jnp.stack([2 * speed_px, jnp.zeros_like(speed_px)], axis=-1)
    displaced = pos + mask[:, None].astype(jnp.int32) * displacement
    state = state.replace(positions=state.positions.at[type_a].set(
        jnp.where(mask[:, None], displaced, pos)))
    state = prim_negate_orientation(state, type_a, mask)
    return _with_score(state, score_delta)


def flip_direction(state, type_a, mask, score_delta, max_n, **_):
    rng, key = jax.random.split(state.rng)
    dir_indices = jax.random.randint(key, (max_n,), 0, 4)
    random_ori = DIRECTION_DELTAS[dir_indices]
    return _with_score(
        prim_set_orientation(state, type_a, mask, random_ori).replace(rng=rng),
        score_delta)


def undo_all(state, prev_positions, mask, score_delta, **_):
    new_positions = jnp.where(mask.any(), prev_positions, state.positions)
    return _with_score(state.replace(positions=new_positions), score_delta)


def wrap_around(state, type_a, mask, score_delta, height, width,
                kwargs=None, block_size=1, **_):
    offset = (kwargs or {}).get('offset', 0)
    ori = state.orientations[type_a]
    pos = state.positions[type_a]
    row_axis = ori[:, 0] != 0
    # Boundaries in pixel space: (height-1-offset)*block_size and offset*block_size
    new_row = jnp.where(
        mask & row_axis & (ori[:, 0] < 0), (height - 1 - offset) * block_size,
        jnp.where(mask & row_axis & (ori[:, 0] > 0), offset * block_size, pos[:, 0]))
    new_col = jnp.where(
        mask & ~row_axis & (ori[:, 1] < 0), (width - 1 - offset) * block_size,
        jnp.where(mask & ~row_axis & (ori[:, 1] > 0), offset * block_size, pos[:, 1]))
    return _with_score(state.replace(
        positions=state.positions.at[type_a].set(
            jnp.stack([new_row, new_col], axis=-1))), score_delta)


def attract_gaze(state, type_a, type_b, mask, score_delta, kwargs, max_n,
                 partner_idx=None, **_):
    prob = kwargs.get('prob', 0.5)
    if type_b >= 0:
        rng, key = jax.random.split(state.rng)
        rolls = jax.random.uniform(key, (max_n,))
        should_attract = mask & (rolls < prob)
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        return _with_score(
            prim_set_orientation(state, type_a, should_attract, partner_ori).replace(rng=rng),
            score_delta)
    return _with_score(state, score_delta)


# ── Resources ──────────────────────────────────────────────────────────

def change_resource(state, type_a, type_b, mask, score_delta, kwargs,
                    partner_idx=None, **_):
    r_idx = kwargs.get('resource_idx', 0)
    value = kwargs.get('value', 0)
    limit = kwargs.get('limit', 100)
    kill_resource = kwargs.get('kill_resource', False)
    cur = state.resources[type_a, :, r_idx]
    new_val = jnp.where(mask, jnp.clip(cur + value, 0, limit), cur)
    state = state.replace(resources=state.resources.at[type_a, :, r_idx].set(new_val))
    if kill_resource and type_b >= 0:
        b_affected = _partner_scatter_mask(mask, partner_idx, state.alive.shape[1])
        state = prim_kill(state, type_b, b_affected)
    return _with_score(state, score_delta)


def collect_resource(state, type_a, type_b, mask, score_delta,
                     kwargs, partner_idx=None, **_):
    r_idx = kwargs.get('resource_idx', 0)
    r_value = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    kill_resource = kwargs.get('kill_resource', False)
    if type_b >= 0:
        b_affected = _partner_scatter_mask(mask, partner_idx, state.alive.shape[1])
        cur = state.resources[type_b, :, r_idx]
        new_val = jnp.where(b_affected, jnp.clip(cur + r_value, 0, limit), cur)
        state = state.replace(
            resources=state.resources.at[type_b, :, r_idx].set(new_val))
    if kill_resource:
        state = prim_kill(state, type_a, mask)
    return _with_score(state, score_delta)


def avatar_collect_resource(state, mask, score_delta, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    r_val = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    cur = state.resources[ati, 0, r_idx]
    new_val = jnp.minimum(cur + mask.sum() * r_val, limit)
    return _with_score(state.replace(
        resources=state.resources.at[ati, 0, r_idx].set(new_val)), score_delta)


def spend_resource(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    amount = kwargs.get('amount', 1)
    cur = state.resources[type_a, :, r_idx]
    spend = jnp.where(mask, jnp.minimum(cur, amount), 0)
    return _with_score(state.replace(
        resources=state.resources.at[type_a, :, r_idx].add(-spend)), score_delta)


def spend_avatar_resource(state, mask, score_delta, kwargs, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    amount = kwargs.get('amount', 1)
    cur = state.resources[ati, 0, r_idx]
    spend = jnp.minimum(cur, mask.sum() * amount)
    return _with_score(state.replace(
        resources=state.resources.at[ati, 0, r_idx].add(-spend)), score_delta)


# ── Spawn and transform ───────────────────────────────────────────────

def transform_to(state, type_a, type_b, mask, score_delta, kwargs,
                  partner_idx=None, **_):
    new_type = kwargs['new_type_idx']
    target_speed = kwargs.get('target_speed', None)
    target_cons = kwargs.get('target_cons', 0)
    target_spawn_cd = kwargs.get('target_spawn_cd', 0)
    kill_second = kwargs.get('kill_second', False)
    copy_ori = kwargs.get('copy_orientation', True)
    if copy_ori:
        orientations = state.orientations[type_a]
    else:
        default_ori = kwargs.get('default_orientation', (0.0, 1.0))
        orientations = jnp.broadcast_to(
            jnp.array(default_ori, dtype=jnp.float32),
            state.orientations[type_a].shape)
    state = _with_score(prim_kill(state, type_a, mask), score_delta)
    state = _fill_slots(state, new_type, mask,
                       state.positions[type_a], orientations,
                       target_speed=target_speed, reset_cooldown=True,
                       src_resources=state.resources[type_a],
                       target_cons=target_cons,
                       target_spawn_cd=target_spawn_cd)
    if kill_second:
        static_b_sg = kwargs.get('static_b_grid_idx', None)
        if static_b_sg is not None:
            # type_b is a static grid — clear cells at type_a positions
            pos_a = state.positions[type_a]
            cells = pos_a // kwargs.get('block_size', 1)
            r = jnp.clip(cells[:, 0], 0, kwargs.get('height', 1) - 1)
            c = jnp.clip(cells[:, 1], 0, kwargs.get('width', 1) - 1)
            clear = jnp.zeros_like(state.static_grids[static_b_sg])
            clear = clear.at[r, c].max(mask)
            state = prim_clear_static(state, static_b_sg, clear)
        elif type_b >= 0 and partner_idx is not None:
            state = prim_kill_partner(state, type_b, partner_idx, mask)
    return state


def clone_sprite(state, type_a, mask, score_delta, kwargs=None, **_):
    target_speed = kwargs.get('target_speed', None) if kwargs else None
    target_cons = kwargs.get('target_cons', 0) if kwargs else 0
    target_spawn_cd = kwargs.get('target_spawn_cd', 0) if kwargs else 0
    state = _fill_slots(state, type_a, mask,
                        state.positions[type_a], state.orientations[type_a],
                        target_speed=target_speed, reset_cooldown=True,
                        target_cons=target_cons,
                        target_spawn_cd=target_spawn_cd)
    return _with_score(state, score_delta)


def _spawn_core(state, type_a, mask, kwargs, max_n=None):
    """Shared spawn logic: apply prob gate, then fill slots in spawn target type."""
    spawn_type = kwargs.get('spawn_type_idx', -1)
    if spawn_type < 0:
        return state, mask
    prob = kwargs.get('prob', 1.0)
    if prob < 1.0 and max_n is not None:
        rng, key = jax.random.split(state.rng)
        state = state.replace(rng=rng)
        rolls = jax.random.uniform(key, (max_n,))
        mask = mask & (rolls < prob)
    state = _fill_slots(state, spawn_type, mask,
                        state.positions[type_a],
                        target_speed=kwargs.get('target_speed', None),
                        reset_cooldown=True,
                        target_cons=kwargs.get('target_cons', 0),
                        target_spawn_cd=kwargs.get('target_spawn_cd', 0))
    return state, mask


def spawn(state, type_a, mask, score_delta, kwargs, max_n, **_):
    state, _ = _spawn_core(state, type_a, mask, kwargs, max_n=max_n)
    return _with_score(state, score_delta)


def spawn_if_has_more(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    mask = mask & (state.resources[type_a, :, r_idx] >= limit)
    state, _ = _spawn_core(state, type_a, mask, kwargs)
    return _with_score(state, score_delta)


def spawn_if_has_less(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    mask = mask & (state.resources[type_a, :, r_idx] <= limit)
    state, _ = _spawn_core(state, type_a, mask, kwargs)
    return _with_score(state, score_delta)


def transform_others_to(state, type_a, mask, score_delta, kwargs, **_):
    target_type = kwargs.get('target_type_idx', -1)
    new_type = kwargs.get('new_type_idx', -1)
    target_speed = kwargs.get('target_speed', None)
    target_cons = kwargs.get('target_cons', 0)
    if target_type >= 0 and new_type >= 0:
        any_collision = mask.any()
        target_alive = state.alive[target_type] & any_collision
        kill_all = jnp.broadcast_to(any_collision, state.alive[target_type].shape)
        state = prim_kill(state, target_type, kill_all)
        state = _fill_slots(state, new_type, target_alive,
                            state.positions[target_type],
                            state.orientations[target_type],
                            target_speed=target_speed, reset_cooldown=True,
                            target_cons=target_cons)
    return _with_score(state, score_delta)


# ── Movement and conveying ─────────────────────────────────────────────

def teleport_to_exit(state, type_a, mask, score_delta, kwargs, **_):
    exit_type = kwargs.get('exit_type_idx', -1)
    if exit_type >= 0:
        rng, key = jax.random.split(state.rng)
        exit_pos = state.positions[exit_type]
        exit_alive = state.alive[exit_type]
        n_exits = exit_alive.sum()
        exit_rank = jnp.cumsum(exit_alive)         # [n_exit_slots]
        max_n = mask.shape[0]

        # Per-sprite independent random exit, fully vectorized (no vmap).
        # rand_indices: [max_n] ints in [0, n_exits)
        rand_indices = jax.random.randint(key, (max_n,), 0, jnp.maximum(n_exits, 1))
        # matches[i, j] = exit_alive[j] & (exit_rank[j] == rand_indices[i] + 1)
        matches = exit_alive[None, :] & (exit_rank[None, :] == rand_indices[:, None] + 1)
        chosen_slots = jnp.argmax(matches, axis=1)  # [max_n]
        targets = exit_pos[chosen_slots]             # [max_n, 2]

        # Copy exit portal orientation to teleported sprite (GVGAI behavior)
        exit_ori = state.orientations[exit_type]
        target_oris = exit_ori[chosen_slots]          # [max_n, 2]

        pos_a = state.positions[type_a]
        active_mask = mask & (n_exits > 0)
        active = active_mask[:, None]
        new_pos = jnp.where(active, targets, pos_a)
        ori_a = state.orientations[type_a]
        new_ori = jnp.where(active, target_oris, ori_a)
        cooldown = state.cooldown_timers[type_a]
        new_cooldown = jnp.where(active_mask, 0, cooldown)
        return _with_score(state.replace(
            positions=state.positions.at[type_a].set(new_pos),
            orientations=state.orientations.at[type_a].set(new_ori),
            cooldown_timers=state.cooldown_timers.at[type_a].set(new_cooldown),
            rng=rng), score_delta)
    return _with_score(state, score_delta)


def convey_sprite(state, type_a, type_b, mask, score_delta,
                  kwargs, partner_idx=None, block_size=1, **_):
    strength = kwargs.get('strength', 1.0)
    if type_b >= 0:
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        valid = (partner_idx >= 0) & mask
        # Movement in pixel space: orientation * strength (cast to int32)
        delta = (partner_ori * strength).astype(jnp.int32)
        state = prim_move(state, type_a, valid, delta)
    return _with_score(state, score_delta)


def wind_gust(state, type_a, type_b, mask, score_delta, max_n,
              partner_idx=None, kwargs=None, block_size=1, **_):
    if kwargs is None:
        kwargs = {}
    if type_b >= 0:
        rng, key = jax.random.split(state.rng)
        strength = kwargs.get('strength', 1.0)
        offsets = jax.random.randint(key, (max_n,), -1, 2)
        per_sprite = strength + offsets.astype(jnp.float32)
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        delta = (partner_ori * per_sprite[:, None]).astype(jnp.int32)
        state = prim_move(state, type_a, mask, delta)
        return _with_score(state.replace(rng=rng), score_delta)
    return _with_score(state, score_delta)


def slip_forward(state, type_a, mask, score_delta, kwargs, max_n,
                 block_size=1, **_):
    prob = kwargs.get('prob', 0.5)
    rng, key = jax.random.split(state.rng)
    rolls = jax.random.uniform(key, (max_n,))
    should_slip = mask & (rolls < prob)
    # Move one cell = block_size pixels in orientation direction
    delta = (state.orientations[type_a] * block_size).astype(jnp.int32)
    state = prim_move(state, type_a, should_slip, delta)
    return _with_score(state.replace(rng=rng), score_delta)


# ── Physics / wall interactions ────────────────────────────────────────

def partner_delta(state, prev_positions, type_a, type_b, mask,
                  height, width, score_delta, partner_idx=None, **_):
    """Apply type_b's movement delta to type_a (bounceForward / pullWithIt)."""
    if type_b >= 0:
        b_curr = _partner_vals(state.positions[type_b], partner_idx)
        b_prev = _partner_vals(prev_positions[type_b], partner_idx)
        b_delta = (b_curr - b_prev).astype(jnp.int32)
        valid = (partner_idx >= 0) & mask
        new_pos = jnp.where(valid[:, None],
                            state.positions[type_a] + b_delta,
                            state.positions[type_a])
        state = state.replace(
            positions=state.positions.at[type_a].set(new_pos))
    return _with_score(state, score_delta)


def wall_stop(state, prev_positions, type_a, type_b, mask,
              score_delta, kwargs, height, width,
              max_a=None, max_b=None, block_size=1, **_):
    global_max_n = state.alive.shape[1]
    eff_a = max_a if max_a is not None else global_max_n
    eff_b = max_b if max_b is not None else global_max_n

    friction = kwargs.get('friction', 0.0)
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    pos = state.positions[type_a, :eff_a]    # int32 pixels
    prev = prev_positions[type_a, :eff_a]    # int32 pixels
    vel = state.velocities[type_a, :eff_a]   # float32
    pf = state.passive_forces[type_a, :eff_a]
    m = mask[:eff_a]

    if static_b_grid_idx is not None:
        grid_b = state.static_grids[static_b_grid_idx]
        delta_d = pos - prev
        going_down = jnp.where(delta_d[:, 0] != 0, delta_d[:, 0] > 0, vel[:, 0] >= 0)
        going_right = jnp.where(delta_d[:, 1] != 0, delta_d[:, 1] > 0, vel[:, 1] >= 0)
        # Wall cell in movement direction: pixel→cell
        r_wall = jnp.where(going_down,
                           (pos[:, 0] + block_size - 1) // block_size,
                           pos[:, 0] // block_size)
        c_wall = jnp.where(going_right,
                           (pos[:, 1] + block_size - 1) // block_size,
                           pos[:, 1] // block_size)
        c_at_prev = prev[:, 1] // block_size
        r_at_prev = prev[:, 0] // block_size
        r_wall_c = jnp.clip(r_wall, 0, height - 1)
        c_wall_c = jnp.clip(c_wall, 0, width - 1)
        c_prev_c = jnp.clip(c_at_prev, 0, width - 1)
        r_prev_c = jnp.clip(r_at_prev, 0, height - 1)
        has_row_cross = grid_b[r_wall_c, c_prev_c]
        has_col_cross = grid_b[r_prev_c, c_wall_c]
    elif type_b >= 0:
        pos_b = state.positions[type_b, :eff_b]  # int32 pixels
        alive_b = state.alive[type_b, :eff_b]
        # Pixel AABB: overlap when |diff| < block_size on both axes
        v_rdiff = jnp.abs(pos[:, None, 0] - pos_b[None, :, 0])
        v_cdiff = jnp.abs(prev[:, None, 1] - pos_b[None, :, 1])
        check_v = (v_rdiff < block_size) & (v_cdiff < block_size) & alive_b[None, :]
        h_rdiff = jnp.abs(prev[:, None, 0] - pos_b[None, :, 0])
        h_cdiff = jnp.abs(pos[:, None, 1] - pos_b[None, :, 1])
        check_h = (h_rdiff < block_size) & (h_cdiff < block_size) & alive_b[None, :]
        has_row_cross = jnp.any(check_v, axis=1)
        has_col_cross = jnp.any(check_h, axis=1)
    else:
        has_row_cross = jnp.zeros_like(m)
        has_col_cross = jnp.zeros_like(m)

    neither = m & ~has_row_cross & ~has_col_cross
    delta = pos - prev
    is_vert_fb = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])
    has_row_cross = has_row_cross | (neither & is_vert_fb)
    has_col_cross = has_col_cross | (neither & ~is_vert_fb)

    vert_mask = m & has_row_cross
    if static_b_grid_idx is not None:
        # Flush to cell boundary in pixel space
        # Going down: flush to pixel just before wall cell → (r_wall - 1) * block_size
        # Going up: flush to pixel just after wall cell → (r_wall + 1) * block_size
        flush_row = jnp.where(going_down,
                              (r_wall - 1) * block_size,
                              (r_wall + 1) * block_size)
        new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
    elif type_b >= 0:
        v_dist = jnp.where(check_v, v_rdiff, jnp.int32(1000000))
        nearest_v = jnp.argmin(v_dist, axis=1)
        wall_row = pos_b[nearest_v, 0]
        flush_row = jnp.where(vel[:, 0] > 0, wall_row - block_size, wall_row + block_size)
        new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
    else:
        new_pos_row = jnp.where(vert_mask, prev[:, 0], pos[:, 0])
    new_vel_row = jnp.where(vert_mask, 0.0, vel[:, 0])
    new_pf_row = jnp.where(vert_mask, 0.0, pf[:, 0])
    new_vel_col_v = jnp.where(
        vert_mask & (friction > 0), vel[:, 1] * (1.0 - friction), vel[:, 1])

    horiz_mask = m & has_col_cross
    if static_b_grid_idx is not None:
        flush_col = jnp.where(going_right,
                              (c_wall - 1) * block_size,
                              (c_wall + 1) * block_size)
        new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
    elif type_b >= 0:
        h_dist = jnp.where(check_h, h_cdiff, jnp.int32(1000000))
        nearest_h = jnp.argmin(h_dist, axis=1)
        wall_col = pos_b[nearest_h, 1]
        flush_col = jnp.where(vel[:, 1] > 0, wall_col - block_size, wall_col + block_size)
        new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
    else:
        new_pos_col = jnp.where(horiz_mask, prev[:, 1], pos[:, 1])
    new_vel_col = jnp.where(horiz_mask, 0.0, new_vel_col_v)
    new_pf_col = jnp.where(horiz_mask, 0.0, pf[:, 1])
    new_vel_row2 = jnp.where(
        horiz_mask & (friction > 0), new_vel_row * (1.0 - friction), new_vel_row)

    new_pos_s = jnp.stack([new_pos_row, new_pos_col], axis=-1)
    new_vel_s = jnp.stack([new_vel_row2, new_vel_col], axis=-1)
    new_pf_s = jnp.stack([new_pf_row, new_pf_col], axis=-1)
    return state.replace(
        positions=state.positions.at[type_a, :eff_a].set(new_pos_s),
        velocities=state.velocities.at[type_a, :eff_a].set(new_vel_s),
        passive_forces=state.passive_forces.at[type_a, :eff_a].set(new_pf_s),
        score=state.score + score_delta,
    )


def wall_bounce(state, prev_positions, type_a, type_b, mask,
                score_delta, kwargs, max_a=None, max_b=None,
                height=0, width=0, block_size=1, **_):
    global_max_n = state.alive.shape[1]
    eff_a = max_a if max_a is not None else global_max_n
    eff_b = max_b if max_b is not None else global_max_n

    friction = kwargs.get('friction', 0.0)
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    pos = state.positions[type_a, :eff_a]    # int32 pixels
    prev = prev_positions[type_a, :eff_a]    # int32 pixels
    vel = state.velocities[type_a, :eff_a]   # float32
    m = mask[:eff_a]

    if static_b_grid_idx is not None:
        grid_b = state.static_grids[static_b_grid_idx]
        # Pixel→cell: find nearest cell center offset
        r_cell = pos[:, 0] // block_size
        c_cell = pos[:, 1] // block_size
        r_center = r_cell * block_size + block_size // 2
        c_center = c_cell * block_size + block_size // 2
        row_diff = jnp.abs(pos[:, 0] - r_center)
        col_diff = jnp.abs(pos[:, 1] - c_center)
        is_vertical = row_diff >= col_diff
    elif type_b >= 0:
        nb_pos = _nearest_partner(pos, state, type_b, eff_b)
        row_diff = jnp.abs(pos[:, 0] - nb_pos[:, 0])
        col_diff = jnp.abs(pos[:, 1] - nb_pos[:, 1])
        is_vertical = row_diff >= col_diff
    else:
        delta = pos - prev
        is_vertical = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])

    new_vel_row = jnp.where(m & is_vertical, -vel[:, 0], vel[:, 0])
    new_vel_col = jnp.where(m & ~is_vertical, -vel[:, 1], vel[:, 1])

    speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
    fric_scale = jnp.where(
        m & (friction > 0) & (speed > 1e-6),
        jnp.maximum(1.0 - friction, 0.0), 1.0)
    new_vel_row = new_vel_row * fric_scale
    new_vel_col = new_vel_col * fric_scale

    new_pos = jnp.where(m[:, None], prev, pos)

    new_speed = jnp.sqrt(new_vel_row ** 2 + new_vel_col ** 2)
    ori_a = state.orientations[type_a, :eff_a]
    new_ori_row = jnp.where(new_speed > 1e-6, new_vel_row / new_speed, ori_a[:, 0])
    new_ori_col = jnp.where(new_speed > 1e-6, new_vel_col / new_speed, ori_a[:, 1])

    return state.replace(
        positions=state.positions.at[type_a, :eff_a].set(new_pos),
        velocities=state.velocities.at[type_a, :eff_a].set(
            jnp.stack([new_vel_row, new_vel_col], axis=-1)),
        orientations=state.orientations.at[type_a, :eff_a].set(
            jnp.stack([new_ori_row, new_ori_col], axis=-1)),
        score=state.score + score_delta,
    )


def bounce_direction(state, prev_positions, type_a, type_b, mask,
                     score_delta, kwargs, max_a=None, max_b=None,
                     height=0, width=0, block_size=1, **_):
    static_b_grid_idx = kwargs.get('static_b_grid_idx') if kwargs else None

    if static_b_grid_idx is not None or type_b >= 0:
        global_max_n = state.alive.shape[1]
        eff_a = max_a if max_a is not None else global_max_n
        eff_b = max_b if max_b is not None else global_max_n

        pos_a = state.positions[type_a, :eff_a]  # int32 pixels
        vel = state.velocities[type_a, :eff_a]   # float32
        prev = prev_positions[type_a, :eff_a]
        m = mask[:eff_a]

        if static_b_grid_idx is not None:
            # Pixel→cell center: use as the "wall position" for reflection normal
            r_cell = jnp.clip(pos_a[:, 0] // block_size, 0, height - 1)
            c_cell = jnp.clip(pos_a[:, 1] // block_size, 0, width - 1)
            nb_pos = jnp.stack([r_cell * block_size + block_size // 2,
                                c_cell * block_size + block_size // 2], axis=-1).astype(jnp.float32)
        else:
            nb_pos = _nearest_partner(pos_a, state, type_b, eff_b)

        n = pos_a - nb_pos
        n_len = jnp.sqrt(jnp.sum(n ** 2, axis=-1, keepdims=True))
        n = jnp.where(n_len > 1e-6, n / n_len, jnp.array([0.0, 0.0]))

        v_dot_n = jnp.sum(vel * n, axis=-1, keepdims=True)
        reflected = vel - 2.0 * v_dot_n * n

        friction = kwargs.get('friction', 0.0)
        fric_scale = jnp.where(
            m & (friction > 0), jnp.maximum(1.0 - friction, 0.0), 1.0)
        reflected = reflected * fric_scale[:, None]

        new_vel = jnp.where(m[:, None], reflected, vel)
        new_pos = jnp.where(m[:, None], prev, pos_a)

        new_speed = jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1, keepdims=True))
        ori_a = state.orientations[type_a, :eff_a]
        new_ori = jnp.where(
            (new_speed > 1e-6) & m[:, None],
            new_vel / new_speed, ori_a)

        return state.replace(
            positions=state.positions.at[type_a, :eff_a].set(new_pos),
            velocities=state.velocities.at[type_a, :eff_a].set(new_vel),
            orientations=state.orientations.at[type_a, :eff_a].set(new_ori),
            score=state.score + score_delta,
        )
    return _with_score(state, score_delta)


# ── Null (unknown effect) ─────────────────────────────────────────────

def null(state, score_delta, **_):
    return _with_score(state, score_delta)


def _add_spawn_kwargs(kwargs, sd, block_size):
    """Add target_speed, target_cons, and target_spawn_cd to effect kwargs for a spawn target."""
    kwargs['target_speed'] = speed_to_pixels(sd.speed, block_size, sd.physics_type)
    if sd.sprite_class == SpriteClass.RANDOM_NPC and sd.cons > 0:
        kwargs['target_cons'] = sd.cons
    if sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
        kwargs['target_spawn_cd'] = max(sd.cooldown, 1)


# ── Compile-time kwargs ────────────────────────────────────────────

def _ckw_transform_to(ed, ctx):
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ''))
    if idx is not None:
        from vgdl_jax.data_model import SPRITE_REGISTRY
        src_def = ctx.game_def.sprites[ctx.concrete_actor_idx] if ctx.concrete_actor_idx is not None else None
        dst_def = ctx.game_def.sprites[idx]
        # GVGAI TransformTo.java line 64: copy orientation only when
        # forceOrientation=true, OR (both src and dst are is_oriented AND
        # the destination sprite's default orientation is DNONE).
        # This prevents overwriting an explicit orientation on the destination type.
        src_oriented = SPRITE_REGISTRY.get(src_def.sprite_class, None) if src_def else None
        dst_oriented = SPRITE_REGISTRY.get(dst_def.sprite_class, None)
        force_ori = ed.kwargs.get('forceOrientation', 'false').lower() == 'true' if isinstance(ed.kwargs.get('forceOrientation'), str) else bool(ed.kwargs.get('forceOrientation', False))
        dst_ori_is_dnone = (dst_def.orientation == (0.0, 0.0))
        copy_ori = force_ori or (
            (src_oriented is not None and src_oriented.is_oriented) and
            (dst_oriented is not None and dst_oriented.is_oriented) and
            dst_ori_is_dnone)
        result = {'new_type_idx': idx, 'copy_orientation': copy_ori}
        _add_spawn_kwargs(result, dst_def, ctx.block_size)
        if not copy_ori:
            result['default_orientation'] = dst_def.orientation
        # GVGAI TransformTo.killSecond: also kill the collision partner
        kill_second = ed.kwargs.get('killSecond', 'false')
        if isinstance(kill_second, str):
            kill_second = kill_second.lower() == 'true'
        if kill_second:
            result['kill_second'] = True
            result['block_size'] = ctx.block_size
        return result
    return {}

def _ckw_clone_sprite(ed, ctx):
    if ctx.concrete_actor_idx is not None:
        sd = ctx.game_def.sprites[ctx.concrete_actor_idx]
        result = {}
        _add_spawn_kwargs(result, sd, ctx.block_size)
        return result
    return {}

def _ckw_change_resource(ed, ctx):
    res_name = ed.kwargs.get('resource', '')
    r_idx = ctx.resource_name_to_idx.get(res_name, 0)
    value = ed.kwargs.get('value', 0)
    limit = ctx.resource_limits[r_idx] if ctx.resource_limits else DEFAULT_RESOURCE_LIMIT
    kill_resource = str(ed.kwargs.get('killResource', 'false')).lower() == 'true'
    result = {'resource_idx': r_idx, 'value': value, 'limit': limit}
    if kill_resource:
        result['kill_resource'] = True
    return result

def _ckw_collect_resource(ed, ctx):
    """Shared by collect_resource and avatar_collect_resource."""
    if ctx.concrete_actor_idx is not None:
        res_sd = ctx.game_def.sprites[ctx.concrete_actor_idx]
    else:
        actor_indices = ctx.game_def.resolve_stype(ed.actor_stype)
        res_sd = ctx.game_def.sprites[actor_indices[0]] if actor_indices else None
    kwargs = {}
    if res_sd is not None:
        res_name = res_sd.resource_name or res_sd.key
        kwargs['resource_idx'] = ctx.resource_name_to_idx.get(res_name, 0)
        kwargs['resource_value'] = res_sd.resource_value
        kwargs['limit'] = ctx.resource_limits[kwargs['resource_idx']] if ctx.resource_limits else DEFAULT_RESOURCE_LIMIT
    kill_resource = ed.kwargs.get('killResource', 'false')
    if str(kill_resource).lower() == 'true':
        kwargs['kill_resource'] = True
    return kwargs

def _ckw_avatar_collect_resource(ed, ctx):
    kwargs = _ckw_collect_resource(ed, ctx)
    kwargs['avatar_type_idx'] = ctx.avatar_type_idx
    return kwargs

def _ckw_kill_if_resource(ed, ctx):
    """Shared by kill_if_has_less/more, kill_if_other_has_more/less."""
    res_name = ed.kwargs.get('resource', '')
    return {
        'resource_idx': ctx.resource_name_to_idx.get(res_name, 0),
        'limit': ed.kwargs.get('limit', 0),
    }

def _ckw_kill_if_slow(ed, ctx):
    return {'limitspeed': ed.kwargs.get('limitspeed', 0.0)}

def _ckw_actee_strength(ed, ctx):
    if ctx.concrete_actee_idx is not None:
        return {'strength': float(ctx.game_def.sprites[ctx.concrete_actee_idx].strength)}
    return {'strength': 1.0}


def _ckw_spawn_if_has_more(ed, ctx):
    res_name = ed.kwargs.get('resource', '')
    kwargs = {
        'resource_idx': ctx.resource_name_to_idx.get(res_name, 0),
        'limit': ed.kwargs.get('limit', 0),
    }
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ''))
    if idx is not None:
        sd = ctx.game_def.sprites[idx]
        kwargs['spawn_type_idx'] = idx
        _add_spawn_kwargs(kwargs, sd, ctx.block_size)
    return kwargs

def _ckw_spawn(ed, ctx):
    kwargs = {}
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ''))
    if idx is not None:
        sd = ctx.game_def.sprites[idx]
        kwargs['spawn_type_idx'] = idx
        _add_spawn_kwargs(kwargs, sd, ctx.block_size)
    prob = float(ed.kwargs.get('prob', 1.0))
    if prob < 1.0:
        kwargs['prob'] = prob
    return kwargs

def _ckw_prob_half(ed, ctx):
    return {'prob': float(ed.kwargs.get('prob', 0.5))}

def _ckw_wrap_around(ed, ctx):
    offset = ed.kwargs.get('offset', 0)
    if offset:
        return {'offset': int(offset)}
    return {}


def _ckw_spend_resource(ed, ctx):
    res_name = ed.kwargs.get('resource', ed.kwargs.get('target', ''))
    return {
        'resource_idx': ctx.resource_name_to_idx.get(res_name, 0),
        'amount': int(ed.kwargs.get('amount', 1)),
    }

def _ckw_spend_avatar_resource(ed, ctx):
    kwargs = _ckw_spend_resource(ed, ctx)
    kwargs['avatar_type_idx'] = ctx.avatar_type_idx
    return kwargs

def _ckw_kill_others(ed, ctx):
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ed.kwargs.get('target', '')))
    if idx is not None:
        return {'kill_type_idx': idx}
    return {}

def _ckw_kill_all(ed, ctx):
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ed.kwargs.get('target', '')))
    if idx is not None:
        return {'kill_type_idx': idx}
    return {}

def _ckw_transform_to_singleton(ed, ctx):
    """Compile kwargs for transformToSingleton: resolve stype (new type) and stype_other."""
    from vgdl_jax.data_model import SPRITE_REGISTRY
    kwargs = {}
    # stype: the new type to transform sprite1 into
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ''))
    if idx is not None:
        src_def = ctx.game_def.sprites[ctx.concrete_actor_idx] if ctx.concrete_actor_idx is not None else None
        dst_def = ctx.game_def.sprites[idx]
        src_oriented = SPRITE_REGISTRY.get(src_def.sprite_class, None) if src_def else None
        dst_oriented = SPRITE_REGISTRY.get(dst_def.sprite_class, None)
        dst_ori_is_dnone = (dst_def.orientation == (0.0, 0.0))
        copy_ori = (
            (src_oriented is not None and src_oriented.is_oriented) and
            (dst_oriented is not None and dst_oriented.is_oriented) and
            dst_ori_is_dnone)
        kwargs['new_type_idx'] = idx
        kwargs['copy_orientation'] = copy_ori
        _add_spawn_kwargs(kwargs, dst_def, ctx.block_size)
        if not copy_ori:
            kwargs['default_orientation'] = dst_def.orientation
    # stype_other: existing sprites of stype are transformed to this type
    other_idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype_other', ''))
    if other_idx is not None:
        kwargs['other_type_idx'] = other_idx
        other_def = ctx.game_def.sprites[other_idx]
        kwargs['other_speed'] = speed_to_pixels(other_def.speed, ctx.block_size, other_def.physics_type)
        if other_def.sprite_class == SpriteClass.RANDOM_NPC and other_def.cons > 0:
            kwargs['other_cons'] = other_def.cons
    return kwargs

def _ckw_kill_if_avatar_without_resource(ed, ctx):
    res_name = ed.kwargs.get('resource', ed.kwargs.get('target', ''))
    return {
        'resource_idx': ctx.resource_name_to_idx.get(res_name, 0),
        'avatar_type_idx': ctx.avatar_type_idx,
    }

def _ckw_transform_others_to(ed, ctx):
    kwargs = {}
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('target', ''))
    if idx is not None:
        kwargs['target_type_idx'] = idx
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ''))
    if idx is not None:
        sd = ctx.game_def.sprites[idx]
        kwargs['new_type_idx'] = idx
        _add_spawn_kwargs(kwargs, sd, ctx.block_size)
    return kwargs

def _ckw_wall_physics(ed, ctx):
    if 'friction' in ed.kwargs:
        return {'friction': float(ed.kwargs['friction'])}
    return {}

def _ckw_teleport_to_exit(ed, ctx):
    if ctx.concrete_actee_idx is not None:
        portal_sd = ctx.game_def.sprites[ctx.concrete_actee_idx]
    else:
        actee_idx = ctx.resolve_first(ctx.game_def, ed.actee_stype)
        portal_sd = ctx.game_def.sprites[actee_idx] if actee_idx is not None else None
    if portal_sd is not None and portal_sd.portal_exit_stype:
        exit_idx = ctx.resolve_first(ctx.game_def, portal_sd.portal_exit_stype)
        if exit_idx is not None:
            return {'exit_type_idx': exit_idx}
    return {}


# ── Static-A handlers ─────────────────────────────────────────────────

def _static_kill_if_other_resource(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width,
                                    compare_fn, block_size=1):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        pos_b = state.positions[type_b]
        ib_b = in_bounds(pos_b // block_size, height, width)
        b_matches = compare_fn(state.resources[type_b, :, r_idx], limit) & state.alive[type_b] & ib_b
        # Build resource grid covering full bounding box (2x2 region), with pixel AABB.
        min_r = pos_b[:, 0] // block_size
        max_r = (pos_b[:, 0] + block_size - 1) // block_size
        min_c = pos_b[:, 1] // block_size
        max_c = (pos_b[:, 1] + block_size - 1) // block_size
        res_grid = jnp.zeros((height, width), dtype=jnp.bool_)
        for (gr, gc) in [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]:
            gr_c = jnp.clip(gr, 0, height - 1)
            gc_c = jnp.clip(gc, 0, width - 1)
            row_diff = jnp.abs(pos_b[:, 0] - gr * block_size)
            col_diff = jnp.abs(pos_b[:, 1] - gc * block_size)
            valid = b_matches & (row_diff < block_size) & (col_diff < block_size)
            res_grid = res_grid.at[gr_c, gc_c].max(valid)
        kill_mask = grid_mask & res_grid
    else:
        kill_mask = jnp.zeros_like(grid_mask)
    state = prim_clear_static(state, sg_idx, kill_mask)
    return _with_score(state, kill_mask.sum() * jnp.int32(score_change))


def _static_kill_sprite(state, sg_idx, type_b, grid_mask, score_change, kwargs, height, width, **_):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    return _with_score(state, n_killed * jnp.int32(score_change))


def _static_kill_both(state, sg_idx, type_b, grid_mask, score_change, kwargs,
                      height, width, block_size=1):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    if type_b >= 0:
        pos_b = state.positions[type_b]
        alive_b = state.alive[type_b]
        ib_b = in_bounds(pos_b // block_size, height, width)
        effective_b = alive_b & ib_b
        # Check all cells covered by the sprite's bounding box (2x2 region),
        # with pixel AABB verification — same pattern as _collision_grid_mask_static_a.
        min_r = pos_b[:, 0] // block_size
        max_r = (pos_b[:, 0] + block_size - 1) // block_size
        min_c = pos_b[:, 1] // block_size
        max_c = (pos_b[:, 1] + block_size - 1) // block_size
        mask_b = jnp.zeros_like(alive_b)
        for (gr, gc) in [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]:
            gr_c = jnp.clip(gr, 0, height - 1)
            gc_c = jnp.clip(gc, 0, width - 1)
            row_diff = jnp.abs(pos_b[:, 0] - gr * block_size)
            col_diff = jnp.abs(pos_b[:, 1] - gc * block_size)
            valid = effective_b & (row_diff < block_size) & (col_diff < block_size)
            mask_b = mask_b | (grid_mask[gr_c, gc_c] & valid)
        state = prim_kill(state, type_b, mask_b)
    return _with_score(state, n_killed * jnp.int32(score_change))


def _static_kill_if_other_has_more(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width,
                                    block_size=1):
    return _static_kill_if_other_resource(
        state, sg_idx, type_b, grid_mask, score_change, kwargs,
        height, width, jnp.greater_equal, block_size=block_size)


def _static_kill_if_other_has_less(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width,
                                    block_size=1):
    return _static_kill_if_other_resource(
        state, sg_idx, type_b, grid_mask, score_change, kwargs,
        height, width, jnp.less_equal, block_size=block_size)


def _static_kill_if_avatar_without_resource(state, sg_idx, type_b, grid_mask,
                                             score_change, kwargs, height, width, **_):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    kill_mask = grid_mask & ~(state.resources[ati, 0, r_idx] > 0)
    state = prim_clear_static(state, sg_idx, kill_mask)
    return _with_score(state, kill_mask.sum() * jnp.int32(score_change))


def _static_collect_resource(state, sg_idx, type_b, grid_mask,
                              score_change, kwargs, height, width, **_):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    r_idx = kwargs.get('resource_idx', 0)
    r_val = kwargs.get('resource_value', 1)
    limit = kwargs.get('limit', 100)
    ati = kwargs.get('avatar_type_idx', type_b)
    current = state.resources[ati, 0, r_idx]
    new_res = jnp.minimum(current + n_killed * r_val, limit)
    return _with_score(state.replace(
        resources=state.resources.at[ati, 0, r_idx].set(new_res)),
        n_killed * jnp.int32(score_change))


def _static_spawn(state, sg_idx, type_b, grid_mask, score_change, kwargs,
                   height, width, block_size=1):
    """Static type_a spawn: spawn target sprites at type_b positions that
    overlap with the static_a grid (given by grid_mask [H,W]).

    In GVGAI, Spawn creates a new sprite at sprite1's position. For static
    type_a, sprite1's position is the grid cell. Since grid_mask gives the
    collision cells, we use type_b's positions (which are at those same cells)
    as the spawn locations.
    """
    spawn_type = kwargs.get('spawn_type_idx', -1)
    target_speed = kwargs.get('target_speed', None)
    target_cons = kwargs.get('target_cons', 0)
    target_spawn_cd = kwargs.get('target_spawn_cd', 0)
    prob = kwargs.get('prob', 1.0)
    if spawn_type < 0 or type_b < 0:
        return state
    max_n = state.alive.shape[1]
    # Find type_b sprites that collide with the static grid
    pos_b = state.positions[type_b]
    alive_b = state.alive[type_b]
    ib_b = in_bounds(pos_b // block_size, height, width)
    effective_b = alive_b & ib_b
    r_b = jnp.clip(pos_b[:, 0] // block_size, 0, height - 1)
    c_b = jnp.clip(pos_b[:, 1] // block_size, 0, width - 1)
    # Type_b sprites on colliding cells
    source_mask = effective_b & grid_mask[r_b, c_b]
    if prob < 1.0:
        rng, key = jax.random.split(state.rng)
        state = state.replace(rng=rng)
        rolls = jax.random.uniform(key, (max_n,))
        source_mask = source_mask & (rolls < prob)
    # Spawn at type_b's positions (same cell as static_a)
    state = _fill_slots(state, spawn_type, source_mask, pos_b,
                        target_speed=target_speed, reset_cooldown=True,
                        target_cons=target_cons, target_spawn_cd=target_spawn_cd)
    return _with_score(state, source_mask.sum() * jnp.int32(score_change))


def _static_spawn_if_has_more(state, sg_idx, type_b, grid_mask, score_change,
                               kwargs, height, width, block_size=1):
    """Static type_a spawnIfHasMore: spawn only if type_b's resource >= limit."""
    spawn_type = kwargs.get('spawn_type_idx', -1)
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if spawn_type < 0 or type_b < 0:
        return state
    target_speed = kwargs.get('target_speed', None)
    target_cons = kwargs.get('target_cons', 0)
    target_spawn_cd = kwargs.get('target_spawn_cd', 0)
    # Build resource grid: cells where type_b has resource >= limit
    pos_b = state.positions[type_b]
    alive_b = state.alive[type_b]
    ib_b = in_bounds(pos_b // block_size, height, width)
    has_enough = alive_b & ib_b & (state.resources[type_b, :, r_idx] >= limit)
    res_grid = jnp.zeros((height, width), dtype=jnp.bool_)
    r_b = jnp.clip(pos_b[:, 0] // block_size, 0, height - 1)
    c_b = jnp.clip(pos_b[:, 1] // block_size, 0, width - 1)
    res_grid = res_grid.at[r_b, c_b].max(has_enough)
    effective_mask = grid_mask & res_grid
    # Delegate to _static_spawn logic
    return _static_spawn(state, sg_idx, type_b, effective_mask, score_change,
                         dict(kwargs, prob=1.0), height, width, block_size=block_size)


# ── Dispatch ───────────────────────────────────────────────────────────

EFFECT_REGISTRY = {
    # Kill / removal
    'killSprite':     EffectEntry(kill_sprite, 'kill_sprite',
        modifies_alive=True, static_a_handler=_static_kill_sprite),
    'killBoth':       EffectEntry(kill_both, 'kill_both',
        needs_partner=True, modifies_alive=True, static_a_handler=_static_kill_both),
    'killIfAlive':    EffectEntry(kill_if_alive, 'kill_if_alive',
        needs_partner=True, modifies_alive=True),
    'killIfSlow':     EffectEntry(kill_if_slow, 'kill_if_slow',
        modifies_alive=True, compile_kwargs=_ckw_kill_if_slow),
    'KillOthers':     EffectEntry(kill_others, 'kill_others',
        modifies_alive=True, compile_kwargs=_ckw_kill_others),
    'killIfFromAbove': EffectEntry(kill_if_from_above, 'kill_if_from_above',
        needs_partner=True, modifies_alive=True),
    # Position and orientation
    'stepBack':          EffectEntry(step_back, 'step_back', modifies_position=True),
    'reverseDirection':  EffectEntry(reverse_direction, 'reverse_direction'),
    'turnAround':        EffectEntry(turn_around, 'turn_around', modifies_position=True),
    'flipDirection':     EffectEntry(flip_direction, 'flip_direction'),
    'undoAll':           EffectEntry(undo_all, 'undo_all', modifies_position=True),
    'wrapAround':        EffectEntry(wrap_around, 'wrap_around',
        modifies_position=True, compile_kwargs=_ckw_wrap_around),
    'attractGaze':       EffectEntry(attract_gaze, 'attract_gaze',
        needs_partner=True, compile_kwargs=_ckw_prob_half),
    # Resources
    'changeResource':       EffectEntry(change_resource, 'change_resource',
        needs_partner=True, modifies_alive=True,
        compile_kwargs=_ckw_change_resource),
    'collectResource':      EffectEntry(collect_resource, 'collect_resource',
        needs_partner=True, static_a_handler=_static_collect_resource,
        compile_kwargs=_ckw_collect_resource),
    'AvatarCollectResource': EffectEntry(avatar_collect_resource, 'avatar_collect_resource',
        static_a_handler=_static_collect_resource,
        compile_kwargs=_ckw_avatar_collect_resource),
    'SpendResource':        EffectEntry(spend_resource, 'spend_resource',
        compile_kwargs=_ckw_spend_resource),
    'SpendAvatarResource':  EffectEntry(spend_avatar_resource, 'spend_avatar_resource',
        compile_kwargs=_ckw_spend_avatar_resource),
    'killIfHasLess':        EffectEntry(kill_if_has_less, 'kill_if_has_less',
        modifies_alive=True, compile_kwargs=_ckw_kill_if_resource),
    'killIfHasMore':        EffectEntry(kill_if_has_more, 'kill_if_has_more',
        modifies_alive=True, compile_kwargs=_ckw_kill_if_resource),
    'killIfOtherHasMore':   EffectEntry(kill_if_other_has_more, 'kill_if_other_has_more',
        needs_partner=True, modifies_alive=True,
        static_a_handler=_static_kill_if_other_has_more,
        compile_kwargs=_ckw_kill_if_resource),
    'killIfOtherHasLess':   EffectEntry(kill_if_other_has_less, 'kill_if_other_has_less',
        needs_partner=True, modifies_alive=True,
        static_a_handler=_static_kill_if_other_has_less,
        compile_kwargs=_ckw_kill_if_resource),
    'KillIfAvatarWithoutResource': EffectEntry(kill_if_avatar_without_resource, 'kill_if_avatar_without_resource',
        modifies_alive=True, static_a_handler=_static_kill_if_avatar_without_resource,
        compile_kwargs=_ckw_kill_if_avatar_without_resource),
    'killAll':           EffectEntry(kill_all, 'kill_all',
        modifies_alive=True, compile_kwargs=_ckw_kill_all),
    'killIfFrontal':     EffectEntry(kill_if_frontal, 'kill_if_frontal',
        needs_partner=True, modifies_alive=True),
    'killIfNotFrontal':  EffectEntry(kill_if_not_frontal, 'kill_if_not_frontal',
        needs_partner=True, modifies_alive=True),
    # Spawn and transform
    'transformTo':       EffectEntry(transform_to, 'transform_to',
        modifies_alive=True, compile_kwargs=_ckw_transform_to),
    'transformToSingleton': EffectEntry(transform_to_singleton, 'transform_to_singleton',
        modifies_alive=True, compile_kwargs=_ckw_transform_to_singleton),
    'cloneSprite':       EffectEntry(clone_sprite, 'clone_sprite',
        modifies_alive=True, compile_kwargs=_ckw_clone_sprite),
    'spawn':             EffectEntry(spawn, 'spawn',
        modifies_alive=True, static_a_handler=_static_spawn,
        compile_kwargs=_ckw_spawn),
    'spawnIfHasMore':    EffectEntry(spawn_if_has_more, 'spawn_if_has_more',
        modifies_alive=True, static_a_handler=_static_spawn_if_has_more,
        compile_kwargs=_ckw_spawn_if_has_more),
    'spawnIfHasLess':    EffectEntry(spawn_if_has_less, 'spawn_if_has_less',
        modifies_alive=True, compile_kwargs=_ckw_spawn_if_has_more),  # same kwargs pattern
    'TransformOthersTo': EffectEntry(transform_others_to, 'transform_others_to',
        compile_kwargs=_ckw_transform_others_to),
    # Movement and conveying
    'teleportToExit':   EffectEntry(teleport_to_exit, 'teleport_to_exit',
        modifies_position=True, compile_kwargs=_ckw_teleport_to_exit),
    'conveySprite':     EffectEntry(convey_sprite, 'convey_sprite',
        needs_partner=True, modifies_position=True, compile_kwargs=_ckw_actee_strength),
    'windGust':         EffectEntry(wind_gust, 'wind_gust',
        needs_partner=True, modifies_position=True, compile_kwargs=_ckw_actee_strength),
    'slipForward':      EffectEntry(slip_forward, 'slip_forward',
        modifies_position=True, compile_kwargs=_ckw_prob_half),
    # Physics / wall interactions
    'bounceForward':    EffectEntry(partner_delta, 'bounce_forward',
        needs_partner=True, modifies_position=True),
    'pullWithIt':       EffectEntry(partner_delta, 'pull_with_it',
        needs_partner=True, modifies_position=True),
    'wallStop':         EffectEntry(wall_stop, 'wall_stop',
        modifies_position=True, compile_kwargs=_ckw_wall_physics),
    'wallBounce':       EffectEntry(wall_bounce, 'wall_bounce',
        modifies_position=True, compile_kwargs=_ckw_wall_physics),
    'bounceDirection':  EffectEntry(bounce_direction, 'bounce_direction',
        modifies_position=True, compile_kwargs=_ckw_wall_physics),
    # No-op
    'NullEffect':       EffectEntry(null, 'null'),
}

# Derived from EFFECT_REGISTRY — single source of truth
EFFECT_DISPATCH = {e.key: e.handler for e in EFFECT_REGISTRY.values()}
VGDL_TO_KEY = {name: e.key for name, e in EFFECT_REGISTRY.items()}
_STATIC_A_HANDLERS = {e.key: e.static_a_handler
                      for e in EFFECT_REGISTRY.values() if e.static_a_handler is not None}
_ENTRY_BY_KEY = {e.key: e for e in EFFECT_REGISTRY.values()}

# Effect metadata sets (exported for step.py / compiler.py)
PARTNER_IDX_EFFECTS = frozenset(e.key for e in EFFECT_REGISTRY.values() if e.needs_partner)
POSITION_MODIFYING_EFFECTS = frozenset(e.key for e in EFFECT_REGISTRY.values() if e.modifies_position)
ALIVE_MODIFYING_EFFECTS = frozenset(e.key for e in EFFECT_REGISTRY.values() if e.modifies_alive)


def compile_effect_kwargs(ed, ctx):
    """Dispatch to per-effect compile function from registry."""
    entry = _ENTRY_BY_KEY.get(ed.effect_type)
    if entry is None or entry.compile_kwargs is None:
        return {}
    return entry.compile_kwargs(ed, ctx)


def apply_static_a_effect(state, static_a_grid_idx, type_b, grid_mask,
                          effect_type, score_change, kwargs, height, width,
                          block_size=1):
    """Apply an effect where type_a is stored as a static grid."""
    handler = _STATIC_A_HANDLERS.get(effect_type)
    if handler is not None:
        return handler(state, static_a_grid_idx, type_b, grid_mask,
                       score_change, kwargs, height, width,
                       block_size=block_size)
    return state  # Unsupported effect — no-op


def apply_masked_effect(state, prev_positions, type_a, type_b, mask,
                        effect_type, score_change, kwargs,
                        height, width, max_n,
                        max_a=None, max_b=None,
                        partner_idx=None, block_size=1):
    """Apply a single effect to all type_a sprites indicated by mask [max_n].

    Both score_delta and score_change are passed to handlers:
      - score_delta = mask.sum() * score_change — pre-computed total for effects
        that apply to ALL colliding sprites.
      - score_change — raw per-sprite value for handlers that conditionally kill a
        SUBSET of colliders.

    partner_idx: optional [max_n] int32 array giving the type_b slot index for each
      type_a sprite (-1 if no collision).
    """
    n_affected = mask.sum()
    score_delta = n_affected * jnp.int32(score_change)
    handler = EFFECT_DISPATCH.get(effect_type, null)
    return handler(
        state,
        type_a=type_a, type_b=type_b, mask=mask,
        score_delta=score_delta, score_change=score_change,
        prev_positions=prev_positions, kwargs=kwargs,
        height=height, width=width, max_n=max_n,
        max_a=max_a, max_b=max_b,
        partner_idx=partner_idx,
        block_size=block_size,
    )
