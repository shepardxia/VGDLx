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
from vgdl_jax.collision import in_bounds, AABB_EPS
from vgdl_jax.data_model import DEFAULT_RESOURCE_LIMIT
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
                src_resources=None):
    """Allocate dead slots in target_type and fill with source data."""
    should_fill, src_idx = prefix_sum_allocate(state.alive[target_type], source_mask)
    src_pos = src_positions[src_idx]
    state = state.replace(
        alive=state.alive.at[target_type].set(state.alive[target_type] | should_fill),
        positions=state.positions.at[target_type].set(
            jnp.where(should_fill[:, None], src_pos, state.positions[target_type])),
        ages=state.ages.at[target_type].set(
            jnp.where(should_fill, 0, state.ages[target_type])),
    )
    if src_orientations is not None:
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


def turn_around(state, prev_positions, type_a, mask, score_delta, height, width, **_):
    # py-vgdl: restore position, move DOWN twice, reverse direction
    state = prim_restore_pos(state, type_a, mask, prev_positions)
    pos = state.positions[type_a]
    displaced = pos + mask[:, None] * jnp.array([2.0, 0.0])
    displaced = jnp.stack([
        jnp.clip(displaced[:, 0], 0, height - 1),
        jnp.clip(displaced[:, 1], 0, width - 1),
    ], axis=-1)
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


def wrap_around(state, type_a, mask, score_delta, height, width, kwargs=None, **_):
    offset = (kwargs or {}).get('offset', 0)
    ori = state.orientations[type_a]
    pos = state.positions[type_a]
    row_axis = ori[:, 0] != 0
    new_row = jnp.where(
        mask & row_axis & (ori[:, 0] < 0), height - 1 - offset,
        jnp.where(mask & row_axis & (ori[:, 0] > 0), 0 + offset, pos[:, 0]))
    new_col = jnp.where(
        mask & ~row_axis & (ori[:, 1] < 0), width - 1 - offset,
        jnp.where(mask & ~row_axis & (ori[:, 1] > 0), 0 + offset, pos[:, 1]))
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

def change_resource(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    value = kwargs.get('value', 0)
    limit = kwargs.get('limit', 100)
    cur = state.resources[type_a, :, r_idx]
    new_val = jnp.where(mask, jnp.clip(cur + value, 0, limit), cur)
    return _with_score(state.replace(
        resources=state.resources.at[type_a, :, r_idx].set(new_val)), score_delta)


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

def transform_to(state, type_a, mask, score_delta, kwargs, **_):
    new_type = kwargs['new_type_idx']
    target_speed = kwargs.get('target_speed', None)
    state = _with_score(prim_kill(state, type_a, mask), score_delta)
    return _fill_slots(state, new_type, mask,
                       state.positions[type_a], state.orientations[type_a],
                       target_speed=target_speed, reset_cooldown=True,
                       src_resources=state.resources[type_a])


def clone_sprite(state, type_a, mask, score_delta, kwargs=None, **_):
    target_speed = kwargs.get('target_speed', None) if kwargs else None
    state = _fill_slots(state, type_a, mask,
                        state.positions[type_a], state.orientations[type_a],
                        target_speed=target_speed, reset_cooldown=True)
    return _with_score(state, score_delta)


def spawn_if_has_more(state, type_a, mask, score_delta, kwargs, **_):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    spawn_type = kwargs.get('spawn_type_idx', -1)
    target_speed = kwargs.get('target_speed', None)
    if spawn_type >= 0:
        has_enough = mask & (state.resources[type_a, :, r_idx] >= limit)
        state = _fill_slots(state, spawn_type, has_enough,
                            state.positions[type_a],
                            target_speed=target_speed, reset_cooldown=True)
    return _with_score(state, score_delta)


def transform_others_to(state, type_a, mask, score_delta, kwargs, **_):
    target_type = kwargs.get('target_type_idx', -1)
    new_type = kwargs.get('new_type_idx', -1)
    target_speed = kwargs.get('target_speed', None)
    if target_type >= 0 and new_type >= 0:
        any_collision = mask.any()
        target_alive = state.alive[target_type] & any_collision
        kill_all = jnp.broadcast_to(any_collision, state.alive[target_type].shape)
        state = prim_kill(state, target_type, kill_all)
        state = _fill_slots(state, new_type, target_alive,
                            state.positions[target_type],
                            state.orientations[target_type],
                            target_speed=target_speed, reset_cooldown=True)
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
                  kwargs, partner_idx=None, **_):
    strength = kwargs.get('strength', 1.0)
    if type_b >= 0:
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        valid = (partner_idx >= 0) & mask
        state = prim_move(state, type_a, valid, partner_ori * strength)
    return _with_score(state, score_delta)


def wind_gust(state, type_a, type_b, mask, score_delta, max_n,
              partner_idx=None, kwargs=None, **_):
    if kwargs is None:
        kwargs = {}
    if type_b >= 0:
        rng, key = jax.random.split(state.rng)
        strength = kwargs.get('strength', 1.0)
        offsets = jax.random.randint(key, (max_n,), -1, 2)
        per_sprite = strength + offsets.astype(jnp.float32)
        partner_ori = _partner_vals(state.orientations[type_b], partner_idx)
        state = prim_move(state, type_a, mask, partner_ori * per_sprite[:, None])
        return _with_score(state.replace(rng=rng), score_delta)
    return _with_score(state, score_delta)


def slip_forward(state, type_a, mask, score_delta, kwargs, max_n, **_):
    prob = kwargs.get('prob', 0.5)
    rng, key = jax.random.split(state.rng)
    rolls = jax.random.uniform(key, (max_n,))
    should_slip = mask & (rolls < prob)
    state = prim_move(state, type_a, should_slip, state.orientations[type_a])
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
        new_pos = jnp.clip(new_pos,
                           jnp.array([0, 0]),
                           jnp.array([height - 1, width - 1]))
        state = state.replace(
            positions=state.positions.at[type_a].set(new_pos))
    return _with_score(state, score_delta)


def wall_stop(state, prev_positions, type_a, type_b, mask,
              score_delta, kwargs, height, width,
              max_a=None, max_b=None, **_):
    global_max_n = state.alive.shape[1]
    eff_a = max_a if max_a is not None else global_max_n
    eff_b = max_b if max_b is not None else global_max_n

    friction = kwargs.get('friction', 0.0)
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    pos = state.positions[type_a, :eff_a]
    prev = prev_positions[type_a, :eff_a]
    vel = state.velocities[type_a, :eff_a]
    pf = state.passive_forces[type_a, :eff_a]
    m = mask[:eff_a]

    if static_b_grid_idx is not None:
        grid_b = state.static_grids[static_b_grid_idx]
        delta_d = pos - prev
        going_down = jnp.where(delta_d[:, 0] != 0, delta_d[:, 0] > 0, vel[:, 0] >= 0)
        going_right = jnp.where(delta_d[:, 1] != 0, delta_d[:, 1] > 0, vel[:, 1] >= 0)
        r_wall = jnp.where(going_down,
                           jnp.ceil(pos[:, 0]).astype(jnp.int32),
                           jnp.floor(pos[:, 0]).astype(jnp.int32))
        c_wall = jnp.where(going_right,
                           jnp.ceil(pos[:, 1]).astype(jnp.int32),
                           jnp.floor(pos[:, 1]).astype(jnp.int32))
        c_at_prev = jnp.round(prev[:, 1]).astype(jnp.int32)
        r_at_prev = jnp.round(prev[:, 0]).astype(jnp.int32)
        r_wall_c = jnp.clip(r_wall, 0, height - 1)
        c_wall_c = jnp.clip(c_wall, 0, width - 1)
        c_prev_c = jnp.clip(c_at_prev, 0, width - 1)
        r_prev_c = jnp.clip(r_at_prev, 0, height - 1)
        has_row_cross = grid_b[r_wall_c, c_prev_c]
        has_col_cross = grid_b[r_prev_c, c_wall_c]
        wall_row_v = r_wall.astype(jnp.float32)
        wall_col_h = c_wall.astype(jnp.float32)
    elif type_b >= 0:
        pos_b = state.positions[type_b, :eff_b]
        alive_b = state.alive[type_b, :eff_b]
        threshold = 1.0 - AABB_EPS
        v_rdiff = jnp.abs(pos[:, None, 0] - pos_b[None, :, 0])
        v_cdiff = jnp.abs(prev[:, None, 1] - pos_b[None, :, 1])
        check_v = (v_rdiff < threshold) & (v_cdiff < threshold) & alive_b[None, :]
        h_rdiff = jnp.abs(prev[:, None, 0] - pos_b[None, :, 0])
        h_cdiff = jnp.abs(pos[:, None, 1] - pos_b[None, :, 1])
        check_h = (h_rdiff < threshold) & (h_cdiff < threshold) & alive_b[None, :]
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
        flush_row = jnp.where(going_down, wall_row_v - 1.0, wall_row_v + 1.0)
        new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
    elif type_b >= 0:
        v_dist = jnp.where(check_v, v_rdiff, 1e10)
        nearest_v = jnp.argmin(v_dist, axis=1)
        wall_row = pos_b[nearest_v, 0]
        flush_row = jnp.where(vel[:, 0] > 0, wall_row - 1.0, wall_row + 1.0)
        new_pos_row = jnp.where(vert_mask, flush_row, pos[:, 0])
    else:
        new_pos_row = jnp.where(vert_mask, prev[:, 0], pos[:, 0])
    new_vel_row = jnp.where(vert_mask, 0.0, vel[:, 0])
    new_pf_row = jnp.where(vert_mask, 0.0, pf[:, 0])
    new_vel_col_v = jnp.where(
        vert_mask & (friction > 0), vel[:, 1] * (1.0 - friction), vel[:, 1])

    horiz_mask = m & has_col_cross
    if static_b_grid_idx is not None:
        flush_col = jnp.where(going_right, wall_col_h - 1.0, wall_col_h + 1.0)
        new_pos_col = jnp.where(horiz_mask, flush_col, pos[:, 1])
    elif type_b >= 0:
        h_dist = jnp.where(check_h, h_cdiff, 1e10)
        nearest_h = jnp.argmin(h_dist, axis=1)
        wall_col = pos_b[nearest_h, 1]
        flush_col = jnp.where(vel[:, 1] > 0, wall_col - 1.0, wall_col + 1.0)
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
                score_delta, kwargs, max_a=None, max_b=None, height=0, width=0, **_):
    global_max_n = state.alive.shape[1]
    eff_a = max_a if max_a is not None else global_max_n
    eff_b = max_b if max_b is not None else global_max_n

    friction = kwargs.get('friction', 0.0)
    static_b_grid_idx = kwargs.get('static_b_grid_idx')
    pos = state.positions[type_a, :eff_a]
    prev = prev_positions[type_a, :eff_a]
    vel = state.velocities[type_a, :eff_a]
    m = mask[:eff_a]

    if static_b_grid_idx is not None:
        grid_b = state.static_grids[static_b_grid_idx]
        r_curr = jnp.clip(jnp.round(pos[:, 0]).astype(jnp.int32), 0, height - 1)
        c_curr = jnp.clip(jnp.round(pos[:, 1]).astype(jnp.int32), 0, width - 1)
        row_diff = jnp.abs(pos[:, 0] - r_curr.astype(jnp.float32))
        col_diff = jnp.abs(pos[:, 1] - c_curr.astype(jnp.float32))
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
                     score_delta, kwargs, max_a=None, max_b=None, height=0, width=0, **_):
    static_b_grid_idx = kwargs.get('static_b_grid_idx') if kwargs else None

    if static_b_grid_idx is not None or type_b >= 0:
        global_max_n = state.alive.shape[1]
        eff_a = max_a if max_a is not None else global_max_n
        eff_b = max_b if max_b is not None else global_max_n

        pos_a = state.positions[type_a, :eff_a]
        vel = state.velocities[type_a, :eff_a]
        prev = prev_positions[type_a, :eff_a]
        m = mask[:eff_a]

        if static_b_grid_idx is not None:
            r_curr = jnp.clip(jnp.round(pos_a[:, 0]).astype(jnp.int32), 0, height - 1)
            c_curr = jnp.clip(jnp.round(pos_a[:, 1]).astype(jnp.int32), 0, width - 1)
            nb_pos = jnp.stack([r_curr.astype(jnp.float32),
                                c_curr.astype(jnp.float32)], axis=-1)
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


# ── Compile-time kwargs ────────────────────────────────────────────

def _ckw_transform_to(ed, ctx):
    idx = ctx.resolve_first(ctx.game_def, ed.kwargs.get('stype', ''))
    if idx is not None:
        return {'new_type_idx': idx, 'target_speed': ctx.game_def.sprites[idx].speed}
    return {}

def _ckw_clone_sprite(ed, ctx):
    if ctx.concrete_actor_idx is not None:
        return {'target_speed': ctx.game_def.sprites[ctx.concrete_actor_idx].speed}
    return {}

def _ckw_change_resource(ed, ctx):
    res_name = ed.kwargs.get('resource', '')
    r_idx = ctx.resource_name_to_idx.get(res_name, 0)
    value = ed.kwargs.get('value', 0)
    limit = ctx.resource_limits[r_idx] if ctx.resource_limits else DEFAULT_RESOURCE_LIMIT
    return {'resource_idx': r_idx, 'value': value, 'limit': limit}

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
        kwargs['spawn_type_idx'] = idx
        kwargs['target_speed'] = ctx.game_def.sprites[idx].speed
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
        kwargs['new_type_idx'] = idx
        kwargs['target_speed'] = ctx.game_def.sprites[idx].speed
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
                                    compare_fn):
    r_idx = kwargs.get('resource_idx', 0)
    limit = kwargs.get('limit', 0)
    if type_b >= 0:
        ipos_b = state.positions[type_b].astype(jnp.int32)
        r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
        c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
        ib_b = in_bounds(ipos_b, height, width)
        b_matches = compare_fn(state.resources[type_b, :, r_idx], limit) & state.alive[type_b] & ib_b
        res_grid = jnp.zeros((height, width), dtype=jnp.bool_).at[r_b, c_b].max(b_matches)
        kill_mask = grid_mask & res_grid
    else:
        kill_mask = jnp.zeros_like(grid_mask)
    state = prim_clear_static(state, sg_idx, kill_mask)
    return _with_score(state, kill_mask.sum() * jnp.int32(score_change))


def _static_kill_sprite(state, sg_idx, type_b, grid_mask, score_change, kwargs, height, width):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    return _with_score(state, n_killed * jnp.int32(score_change))


def _static_kill_both(state, sg_idx, type_b, grid_mask, score_change, kwargs, height, width):
    n_killed = grid_mask.sum()
    state = prim_clear_static(state, sg_idx, grid_mask)
    if type_b >= 0:
        ipos_b = state.positions[type_b].astype(jnp.int32)
        r_b = jnp.clip(ipos_b[:, 0], 0, height - 1)
        c_b = jnp.clip(ipos_b[:, 1], 0, width - 1)
        ib_b = in_bounds(ipos_b, height, width)
        mask_b = grid_mask[r_b, c_b] & state.alive[type_b] & ib_b
        state = prim_kill(state, type_b, mask_b)
    return _with_score(state, n_killed * jnp.int32(score_change))


def _static_kill_if_other_has_more(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width):
    return _static_kill_if_other_resource(
        state, sg_idx, type_b, grid_mask, score_change, kwargs,
        height, width, jnp.greater_equal)


def _static_kill_if_other_has_less(state, sg_idx, type_b, grid_mask,
                                    score_change, kwargs, height, width):
    return _static_kill_if_other_resource(
        state, sg_idx, type_b, grid_mask, score_change, kwargs,
        height, width, jnp.less_equal)


def _static_kill_if_avatar_without_resource(state, sg_idx, type_b, grid_mask,
                                             score_change, kwargs, height, width):
    ati = kwargs.get('avatar_type_idx', 0)
    r_idx = kwargs.get('resource_idx', 0)
    kill_mask = grid_mask & ~(state.resources[ati, 0, r_idx] > 0)
    state = prim_clear_static(state, sg_idx, kill_mask)
    return _with_score(state, kill_mask.sum() * jnp.int32(score_change))


def _static_collect_resource(state, sg_idx, type_b, grid_mask,
                              score_change, kwargs, height, width):
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
    # Spawn and transform
    'transformTo':       EffectEntry(transform_to, 'transform_to',
        modifies_alive=True, compile_kwargs=_ckw_transform_to),
    'cloneSprite':       EffectEntry(clone_sprite, 'clone_sprite',
        modifies_alive=True, compile_kwargs=_ckw_clone_sprite),
    'spawnIfHasMore':    EffectEntry(spawn_if_has_more, 'spawn_if_has_more',
        modifies_alive=True, compile_kwargs=_ckw_spawn_if_has_more),
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
                          effect_type, score_change, kwargs, height, width):
    """Apply an effect where type_a is stored as a static grid."""
    handler = _STATIC_A_HANDLERS.get(effect_type)
    if handler is not None:
        return handler(state, static_a_grid_idx, type_b, grid_mask,
                       score_change, kwargs, height, width)
    return state  # Unsupported effect — no-op


def apply_masked_effect(state, prev_positions, type_a, type_b, mask,
                        effect_type, score_change, kwargs,
                        height, width, max_n,
                        max_a=None, max_b=None,
                        partner_idx=None):
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
    )
