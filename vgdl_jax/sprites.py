import jax
import jax.numpy as jnp
from vgdl_jax.state import GameState
from vgdl_jax.collision import in_bounds
from vgdl_jax.data_model import N_DIRECTIONS

# UP, DOWN, LEFT, RIGHT — int32 for pixel-space arithmetic
DIRECTION_DELTAS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)
# Float32 version for orientation comparisons/updates (avoids repeated .astype())
DIRECTION_DELTAS_F32 = DIRECTION_DELTAS.astype(jnp.float32)


# ── Shared helpers ─────────────────────────────────────────────────────


def prefix_sum_allocate(alive_mask, source_mask):
    """Allocate dead slots to sources using prefix-sum.

    Args:
        alive_mask: [max_n] bool — current alive mask of the target type
        source_mask: [N] bool — which sources want to spawn (N may differ from max_n)

    Returns:
        (should_fill, src_indices):
            should_fill: [max_n] bool — which target slots to fill
            src_indices: [max_n] int — which source maps to each target slot
    """
    n_spawns = source_mask.sum()
    # Dead slots in the target type are candidates for filling
    available = ~alive_mask
    # Assign rank 1,2,3... to each dead slot via prefix-sum
    slot_rank = jnp.cumsum(available)
    # Select the first N dead slots (one per source that wants to spawn)
    should_fill = available & (slot_rank <= n_spawns)
    # argsort(~source_mask) puts True-valued sources first (False sorts as 0)
    source_order = jnp.argsort(~source_mask)
    # Map each target slot to its corresponding source by rank lookup:
    # slot with rank k gets source_order[k-1]
    src_indices = source_order[jnp.clip(slot_rank - 1, 0, source_mask.shape[0] - 1)]
    return should_fill, src_indices


def _move_with_cooldown(state, type_idx, cooldown, deltas=None, passive=True,
                        skip_mask=None):
    """Apply cooldown-gated movement. Uses orientations if deltas is None.

    Positions are int32 pixels. Speed is int32 pixel displacement per tick.
    Movement: new_pos = pos + deltas * speed * can_move (pure int32 arithmetic).

    GVGAI has two movement paths:
    - passiveMovement (missiles): isFirstTick blocks movement, then clears flag
    - activeMovement (chasers, random NPCs): isFirstTick does NOT block, just clears

    Args:
        passive: If True, is_first_tick blocks movement (missile-style).
                 If False, is_first_tick is only cleared, not blocking (chaser-style).
        skip_mask: [max_n] bool — if True, skip movement AND cooldown reset for
                   that sprite (e.g. DNONE direction in RandomNPC cons phase).
                   In GVGAI, activeMovement() returns early when action==DNONE,
                   so _updatePos() never runs and lastmove keeps incrementing.

    Returns:
        (new_pos, new_timers, can_move, first_tick_mask)
        first_tick_mask: [max_n] bool — sprites whose is_first_tick should be cleared
    """
    alive = state.alive[type_idx]
    first_tick = state.is_first_tick[type_idx] & alive
    cooldown_ok = (state.cooldown_timers[type_idx] >= cooldown) & alive
    can_move = (~first_tick & cooldown_ok) if passive else cooldown_ok
    if skip_mask is not None:
        can_move = can_move & ~skip_mask
    if deltas is None:
        # Orientations are float32 direction vectors {-1,0,1} — truncate to int32
        deltas = state.orientations[type_idx].astype(jnp.int32)
    speed = state.speeds[type_idx]  # [max_n] int32 pixel displacement
    new_pos = state.positions[type_idx] + deltas * speed[:, None] * can_move[:, None].astype(jnp.int32)
    new_timers = jnp.where(can_move, 0, state.cooldown_timers[type_idx])
    return new_pos, new_timers, can_move, first_tick


def _apply_npc_move(state, type_idx, new_pos, new_timers, new_ori=None, rng=None,
                    first_tick_mask=None):
    """Write back NPC movement results: positions + timers, optionally orientations + rng.

    If first_tick_mask is provided, clears is_first_tick for those sprites.
    """
    updates = dict(
        positions=state.positions.at[type_idx].set(new_pos),
        cooldown_timers=state.cooldown_timers.at[type_idx].set(new_timers),
    )
    if new_ori is not None:
        updates['orientations'] = state.orientations.at[type_idx].set(new_ori)
    if rng is not None:
        updates['rng'] = rng
    if first_tick_mask is not None:
        updates['is_first_tick'] = state.is_first_tick.at[type_idx].set(
            state.is_first_tick[type_idx] & ~first_tick_mask)
    return state.replace(**updates)


# ── NPC movement updates ──────────────────────────────────────────────


def update_missile(state: GameState, type_idx, cooldown):
    """Move along fixed orientation each tick (if cooldown met and alive)."""
    new_pos, new_timers, _, first_tick = _move_with_cooldown(
        state, type_idx, cooldown)
    return _apply_npc_move(state, type_idx, new_pos, new_timers,
                           first_tick_mask=first_tick)


def update_erratic_missile(state: GameState, type_idx, cooldown, prob):
    """Missile that randomly changes direction with probability `prob` each tick."""
    rng, key_move, key_dir = jax.random.split(state.rng, 3)
    max_n = state.alive.shape[1]

    # Move along current orientation (same as missile)
    new_pos, new_timers, _, first_tick = _move_with_cooldown(
        state, type_idx, cooldown)

    # Randomly change direction with probability `prob`
    should_change = (jax.random.uniform(key_move, (max_n,)) < prob) & state.alive[type_idx]
    dir_indices = jax.random.randint(key_dir, (max_n,), 0, N_DIRECTIONS)
    random_ori = DIRECTION_DELTAS_F32[dir_indices]
    new_ori = jnp.where(should_change[:, None], random_ori, state.orientations[type_idx])

    return _apply_npc_move(state, type_idx, new_pos, new_timers, new_ori=new_ori, rng=rng,
                           first_tick_mask=first_tick)


def update_random_npc(state: GameState, type_idx, cooldown, cons=0):
    """Pick a random direction each move. cons>0 repeats direction for N ticks."""
    rng, key = jax.random.split(state.rng)
    max_n = state.alive.shape[1]
    dir_indices = jax.random.randint(key, (max_n,), 0, N_DIRECTIONS)
    random_deltas = DIRECTION_DELTAS[dir_indices]  # int32
    if cons > 0:
        # Use current orientation while direction_ticks > 0, else pick new
        ticks = state.direction_ticks[type_idx]
        keep = ticks > 0
        # Orientations are float32, convert to int32 for delta
        cur_deltas = state.orientations[type_idx].astype(jnp.int32)
        deltas = jnp.where(keep[:, None], cur_deltas, random_deltas)
        new_ticks = jnp.where(keep, ticks - 1, cons)
    else:
        deltas = random_deltas
        new_ticks = None
    # GVGAI: activeMovement() skips _updatePos() when action==DNONE,
    # so lastmove keeps incrementing (cooldown NOT reset). This happens
    # during the cons repeat phase when the initial orientation is zero.
    is_dnone = (deltas[:, 0] == 0) & (deltas[:, 1] == 0)
    new_pos, new_timers, can_move, first_tick = _move_with_cooldown(
        state, type_idx, cooldown, deltas=deltas, passive=False,
        skip_mask=is_dnone)
    new_ori = jnp.where(can_move[:, None], deltas.astype(jnp.float32),
                        state.orientations[type_idx])
    state = _apply_npc_move(state, type_idx, new_pos, new_timers, new_ori=new_ori, rng=rng,
                            first_tick_mask=first_tick)
    if new_ticks is not None:
        state = state.replace(
            direction_ticks=state.direction_ticks.at[type_idx].set(new_ticks))
    return state


def _relax_distance_field(init_grid, height, width):
    """Run Manhattan-distance BFS relaxation on an initial distance grid.

    Iterates (height + width) times, each step propagating the minimum
    distance from the 4 cardinal neighbours + 1.

    Args:
        init_grid: [H, W] int32 — 0 at source cells, INF elsewhere.
        height: grid height (used for iteration count).
        width: grid width (used for iteration count).

    Returns:
        [H, W] int32 distance field.
    """
    INF = jnp.int32(height + width)

    def _relax_step(_, dist):
        padded = jnp.pad(dist, 1, mode='constant', constant_values=INF)
        up    = padded[:-2, 1:-1] + 1
        down  = padded[2:,  1:-1] + 1
        left  = padded[1:-1, :-2] + 1
        right = padded[1:-1, 2:]  + 1
        return jnp.minimum(dist, jnp.minimum(
            jnp.minimum(up, down), jnp.minimum(left, right)))

    return jax.lax.fori_loop(0, height + width, _relax_step, init_grid)


def _manhattan_distance_field(target_pos, target_alive, height, width, block_size):
    """Compute Manhattan distance to nearest alive target for every grid cell.

    Uses iterative relaxation: O(H*W*(H+W)) total work, O(H+W) depth.
    Returns [H, W] int32 distance field.

    Args:
        target_pos: [max_n, 2] int32 pixel positions
        block_size: pixels per cell (for pixel→cell conversion)
    """
    INF = jnp.int32(height + width)
    grid = jnp.full((height, width), INF, dtype=jnp.int32)
    # Convert pixel positions to cells
    itarget_cell = target_pos // block_size
    ib = in_bounds(itarget_cell, height, width)
    effective = target_alive & ib
    r = jnp.clip(itarget_cell[:, 0], 0, height - 1)
    c = jnp.clip(itarget_cell[:, 1], 0, width - 1)
    grid = grid.at[r, c].min(jnp.where(effective, jnp.int32(0), INF))

    return _relax_distance_field(grid, height, width)


def update_chaser(state: GameState, type_idx, target_type_idx, cooldown,
                  fleeing=False, height=0, width=0, dist_field=None,
                  block_size=1, target_always_alive=False):
    """Move toward (or away from) nearest target using grid distance field. O(H*W + N).

    In GVGAI, isFirstTick is cleared in updatePassive() but doesn't block the
    action (activeMovement still runs). We match this: first_tick sprites CAN
    move, but we clear the flag afterwards.

    Args:
        target_always_alive: True when the target is a static grid type whose
            alive mask is empty (positions stored in static_grids, not arrays).
            The precomputed dist_field is valid; skip the any_target_alive check.
    """
    rng, key = jax.random.split(state.rng)

    # Convert pixel positions to cells for distance field lookup
    chaser_cell = state.positions[type_idx] // block_size  # [max_n, 2] int32
    if target_always_alive:
        any_target_alive = True  # Python True: compiled out at trace time
    else:
        target_alive = state.alive[target_type_idx]
        any_target_alive = jnp.any(target_alive)

    # Distance field: [H, W] Manhattan distance to nearest alive target
    if dist_field is None:
        target_pos = state.positions[target_type_idx]
        dist_field = _manhattan_distance_field(target_pos, target_alive,
                                               height, width, block_size)

    # For each chaser, look up distance at each neighbor direction
    r = jnp.clip(chaser_cell[:, 0], 0, height - 1)
    c = jnp.clip(chaser_cell[:, 1], 0, width - 1)
    INF = jnp.int32(height + width)
    d_up    = jnp.where(r > 0,          dist_field[jnp.clip(r - 1, 0, height - 1), c], INF)
    d_down  = jnp.where(r < height - 1, dist_field[jnp.clip(r + 1, 0, height - 1), c], INF)
    d_left  = jnp.where(c > 0,          dist_field[r, jnp.clip(c - 1, 0, width - 1)], INF)
    d_right = jnp.where(c < width - 1,  dist_field[r, jnp.clip(c + 1, 0, width - 1)], INF)

    neighbor_dists = jnp.stack([d_up, d_down, d_left, d_right], axis=-1)  # [max_n, 4]

    if fleeing:
        best_val = jnp.max(neighbor_dists, axis=-1, keepdims=True)
    else:
        best_val = jnp.min(neighbor_dists, axis=-1, keepdims=True)
    is_best = (neighbor_dists == best_val)

    # Gumbel-max trick for random tie-breaking among tied-best directions
    rng, key_tie = jax.random.split(rng)
    gumbel = jax.random.uniform(key_tie, neighbor_dists.shape)
    tie_scores = jnp.where(is_best, gumbel, -1.0)
    best_dir = jnp.argmax(tie_scores, axis=-1)
    delta = DIRECTION_DELTAS[best_dir]  # int32

    # If no targets alive, pick random direction
    rand_dirs = jax.random.randint(key, (chaser_cell.shape[0],), 0, N_DIRECTIONS)
    rand_delta = DIRECTION_DELTAS[rand_dirs]
    delta = jnp.where(any_target_alive, delta, rand_delta)

    new_pos, new_timers, can_move, first_tick = _move_with_cooldown(
        state, type_idx, cooldown, deltas=delta, passive=False)
    new_ori = jnp.where(can_move[:, None], delta.astype(jnp.float32),
                        state.orientations[type_idx])
    return _apply_npc_move(state, type_idx, new_pos, new_timers, new_ori=new_ori, rng=rng,
                           first_tick_mask=first_tick)


def spawn_sprite(state: GameState, pos, target_type, orientation, speed):
    """Create a new sprite of target_type at the given position.

    Tick-spawned (SpawnPoint, Bomber, avatar shoot): cooldown_timers=0,
    is_first_tick=True. The reverse NPC loop in step.py gives these sprites
    per-type preMovement (0→1) then update (isFirstTick blocks passiveMovement,
    clears flag) in the same tick — matching GVGAI's per-sprite processing.

    NOTE: The field writes here (alive, positions, orientations, speeds, ages,
    cooldown_timers, is_first_tick) duplicate _fill_slots() in effects.py.
    Cannot call _fill_slots directly because effects.py imports from sprites.py,
    so importing effects.py here would create a circular import. Keep in sync.
    """
    available = ~state.alive[target_type]
    slot = jnp.argmax(available)
    has_slot = available[slot]
    state = state.replace(
        alive=state.alive.at[target_type, slot].set(
            state.alive[target_type, slot] | has_slot),
        positions=state.positions.at[target_type, slot].set(
            jnp.where(has_slot, pos, state.positions[target_type, slot])),
        orientations=state.orientations.at[target_type, slot].set(
            jnp.where(has_slot, orientation, state.orientations[target_type, slot])),
        speeds=state.speeds.at[target_type, slot].set(
            jnp.where(has_slot, speed, state.speeds[target_type, slot])),
        ages=state.ages.at[target_type, slot].set(
            jnp.where(has_slot, 0, state.ages[target_type, slot])),
        cooldown_timers=state.cooldown_timers.at[target_type, slot].set(
            jnp.where(has_slot, 0, state.cooldown_timers[target_type, slot])),
        is_first_tick=state.is_first_tick.at[target_type, slot].set(
            jnp.where(has_slot, True, state.is_first_tick[target_type, slot])),
    )
    return state


def update_spawn_point(state: GameState, type_idx, cooldown, prob, total,
                       target_type, target_orientation, target_speed,
                       target_singleton=False, target_cons=0,
                       target_spawn_cd=0):
    """Conditionally spawn sprites — fully vectorized via prefix-sum slot allocation.

    Uses spawn_timers (not cooldown_timers) for spawn readiness checks, so
    spawning and movement timing are fully independent (RC3).

    Spawned sprites get cooldown_timers=0, is_first_tick=True. The reverse NPC
    loop in step.py gives them per-type preMovement (0→1) then update
    (isFirstTick blocks passiveMovement, clears flag) in the same tick.

    If target_singleton=True, only spawns if no alive sprites of target_type
    exist (matching GVGAI's singletons[typeInt] check in Game.addSprite).

    If target_cons > 0, set direction_ticks=cons and orientation=(0,0) for
    spawned sprites (RandomNPC cons initialization, matching GVGAI addSprite
    from template with counter=0, prevAction=DNONE).

    If target_spawn_cd > 0, the target is itself a SpawnPoint/Bomber. Its
    spawn_timers are initialized dynamically based on state.step_count to match
    GVGAI's (start+gameTick)%cd formula where start=gameTick at first update.

    NOTE: The slot-filling field writes (alive, positions, orientations, speeds,
    ages, cooldown_timers, is_first_tick) duplicate _fill_slots() in effects.py.
    Cannot call _fill_slots directly because effects.py imports from sprites.py,
    so importing effects.py here would create a circular import. Keep in sync.
    """
    rng, key = jax.random.split(state.rng)
    max_n = state.alive.shape[1]

    # Vectorized spawn decision — uses spawn_timers, not cooldown_timers
    is_alive = state.alive[type_idx]
    timer_ready = state.spawn_timers[type_idx] >= cooldown
    under_total = (total <= 0) | (state.spawn_counts[type_idx] < total)
    rand_ok = jax.random.uniform(key, (max_n,)) < prob
    should_spawn = is_alive & timer_ready & under_total & rand_ok

    # RC7: singleton check — refuse to spawn if target type already has an alive sprite
    if target_singleton:
        target_has_none = ~jnp.any(state.alive[target_type])
        should_spawn = should_spawn & target_has_none

    # Parallel slot allocation in target type
    should_fill, src_idx = prefix_sum_allocate(state.alive[target_type], should_spawn)
    src_pos = state.positions[type_idx][src_idx]

    # Orientation: if target_cons > 0, use (0,0) = DNONE for cons init
    if target_cons > 0:
        spawn_orientation = jnp.zeros(2, dtype=jnp.float32)
    else:
        spawn_orientation = target_orientation

    state = state.replace(
        alive=state.alive.at[target_type].set(
            state.alive[target_type] | should_fill),
        positions=state.positions.at[target_type].set(
            jnp.where(should_fill[:, None], src_pos,
                      state.positions[target_type])),
        orientations=state.orientations.at[target_type].set(
            jnp.where(should_fill[:, None], spawn_orientation,
                      state.orientations[target_type])),
        speeds=state.speeds.at[target_type].set(
            jnp.where(should_fill, target_speed,
                      state.speeds[target_type])),
        ages=state.ages.at[target_type].set(
            jnp.where(should_fill, 0, state.ages[target_type])),
        cooldown_timers=state.cooldown_timers.at[target_type].set(
            jnp.where(should_fill, 0, state.cooldown_timers[target_type])),
        is_first_tick=state.is_first_tick.at[target_type].set(
            jnp.where(should_fill, True, state.is_first_tick[target_type])),
        rng=rng,
    )

    # Set direction_ticks for RandomNPC cons initialization
    if target_cons > 0:
        state = state.replace(
            direction_ticks=state.direction_ticks.at[target_type].set(
                jnp.where(should_fill, target_cons,
                           state.direction_ticks[target_type])))

    # Set spawn_timers for spawned SpawnPoint/Bomber targets.
    # GVGAI: start = gameTick on first update(). Formula: (start+gameTick)%cd.
    # For tick-spawned targets processed in the same tick: start = step_count.
    # First fire check: (2*step_count) % cd == 0.
    # VGDLx: _pre_spawn increments spawn_timers, fires when >= cd.
    # Set init = cd - 1 - ((2*step_count) % cd) so first _pre_spawn brings it
    # to cd iff (2*step_count)%cd==0, matching GVGAI exactly.
    if target_spawn_cd > 0:
        init_val = target_spawn_cd - 1 - ((2 * state.step_count) % target_spawn_cd)
        state = state.replace(
            spawn_timers=state.spawn_timers.at[target_type].set(
                jnp.where(should_fill, init_val,
                           state.spawn_timers[target_type])))

    # Update spawn counts — only for spawners that actually got a slot
    n_filled = should_fill.sum()
    spawn_rank = jnp.cumsum(should_spawn)
    actually_spawned = should_spawn & (spawn_rank <= n_filled)
    new_counts = state.spawn_counts[type_idx] + actually_spawned.astype(jnp.int32)
    state = state.replace(
        spawn_counts=state.spawn_counts.at[type_idx].set(new_counts))

    # Reset spawn_timers for all spawners that attempted (timer_ready)
    attempted = is_alive & timer_ready & under_total
    new_timers = jnp.where(attempted, 0, state.spawn_timers[type_idx])
    state = state.replace(
        spawn_timers=state.spawn_timers.at[type_idx].set(new_timers))

    # Kill spawners that reached total
    total_reached = (total > 0) & (new_counts >= total)
    state = state.replace(
        alive=state.alive.at[type_idx].set(
            state.alive[type_idx] & ~(total_reached & actually_spawned)))

    return state


def flicker_age(state, type_idx, flicker_limit):
    """GVGAI Flicker.update() aging: if(age > limit) kill; age++.

    Called inside NPC updaters for Flicker-derived types (Flicker, OrientedFlicker,
    Spreader) so aging happens per-type during the NPC loop, matching GVGAI's
    Flicker.update() which runs inside the tick() NPC loop. This prevents newly
    spawned Flicker sprites from being aged on their spawn tick (they weren't in
    the iteration list when their type's NPC updater ran).

    Step 4's global aging skips types with flicker_limit > 0 since they age here.
    """
    alive = state.alive[type_idx]
    ages = state.ages[type_idx]
    expired = (ages > flicker_limit) & alive
    new_alive = alive & ~expired
    new_ages = jnp.where(new_alive, ages + 1, ages)
    return state.replace(
        alive=state.alive.at[type_idx].set(new_alive),
        ages=state.ages.at[type_idx].set(new_ages),
    )


def update_spreader(state: GameState, type_idx, flicker_limit, spreadprob,
                    target_type=-1, block_size=1):
    """Spreader: at age==2, spread to 4 adjacent cells with probability `spreadprob` each.

    GVGAI: `int newType = (itype == -1) ? this.getType() : itype;`
    When target_type >= 0, spawn into that type's slots; otherwise into self.

    GVGAI timing: Spreader.update() calls super.update() (Flicker), which does
    `if(age > limit) kill; age++`. Then Spreader checks `if(age == 2)` using the
    post-increment age. We match this by calling flicker_age() first (which does
    the death check + increment), then checking the updated ages.

    Neighbor positions are one cell away = block_size pixels in each direction.
    """
    # Step 1: Flicker aging (super.update()) — death check + age increment
    state = flicker_age(state, type_idx, flicker_limit)

    rng, key = jax.random.split(state.rng)
    max_n = state.alive.shape[1]

    is_alive = state.alive[type_idx]
    ages = state.ages[type_idx]
    # GVGAI: Flicker.update() did age++, then Spreader checks age==2.
    # flicker_age() already incremented, so check ages==2 directly.
    should_spread = is_alive & (ages == 2)

    # Random gate per sprite per direction
    rng, key_rand = jax.random.split(rng)
    rand_vals = jax.random.uniform(key_rand, (max_n, 4))
    spread_gates = rand_vals < spreadprob

    # Neighbor positions: one cell away = block_size pixels
    pos = state.positions[type_idx]  # [max_n, 2] int32
    neighbor_pos = pos[:, None, :] + DIRECTION_DELTAS[None, :, :] * block_size  # [max_n, 4, 2]

    spawn_mask = should_spread[:, None] & spread_gates
    flat_mask = spawn_mask.reshape(-1)
    flat_pos = neighbor_pos.reshape(-1, 2)

    # Determine target type: use stype if resolved, else self
    tgt = target_type if target_type >= 0 else type_idx

    should_fill, src_idx = prefix_sum_allocate(state.alive[tgt], flat_mask)
    src_pos = flat_pos[src_idx]

    state = state.replace(
        alive=state.alive.at[tgt].set(
            state.alive[tgt] | should_fill),
        positions=state.positions.at[tgt].set(
            jnp.where(should_fill[:, None], src_pos,
                      state.positions[tgt])),
        ages=state.ages.at[tgt].set(
            jnp.where(should_fill, 0, state.ages[tgt])),
        cooldown_timers=state.cooldown_timers.at[tgt].set(
            jnp.where(should_fill, 0, state.cooldown_timers[tgt])),
        is_first_tick=state.is_first_tick.at[tgt].set(
            jnp.where(should_fill, True, state.is_first_tick[tgt])),
        rng=rng,
    )
    return state


def update_random_inertial(state: GameState, type_idx, mass, strength):
    """RandomInertial: ContinuousPhysics NPC, random force direction each tick.

    velocity += (random_direction * strength) / mass
    position += (int32) velocity
    orientation = velocity direction
    """
    rng, key = jax.random.split(state.rng)
    max_n = state.alive.shape[1]

    dir_indices = jax.random.randint(key, (max_n,), 0, N_DIRECTIONS)
    force = DIRECTION_DELTAS_F32[dir_indices] * strength

    vel = state.velocities[type_idx]
    new_vel = vel + force / mass
    is_alive = state.alive[type_idx]
    new_vel = jnp.where(is_alive[:, None], new_vel, vel)

    # Position update: truncate velocity to int32 (matching GVGAI's (int) cast)
    new_pos = state.positions[type_idx] + (new_vel * is_alive[:, None]).astype(jnp.int32)

    speed = jnp.sqrt(jnp.sum(new_vel ** 2, axis=-1, keepdims=True))
    new_ori = jnp.where(
        (speed > 1e-6) & is_alive[:, None],
        new_vel / speed,
        state.orientations[type_idx])

    return state.replace(
        positions=state.positions.at[type_idx].set(new_pos),
        velocities=state.velocities.at[type_idx].set(new_vel),
        orientations=state.orientations.at[type_idx].set(new_ori),
        rng=rng,
    )


def update_walk_jumper(state: GameState, type_idx, prob, strength, gravity, mass):
    """WalkJumper NPC: horizontal walker with random upward jumps under gravity.

    Moves horizontally in orientation direction each tick.
    With probability (1-prob), applies upward velocity impulse when grounded.
    Gravity pulls down every tick.
    """
    alive = state.alive[type_idx]
    n = alive.shape[0]

    vel = state.velocities[type_idx]
    pf = state.passive_forces[type_idx]

    grounded = (pf[:, 0] == 0.0)

    rng, key = jax.random.split(state.rng)
    rand_vals = jax.random.uniform(key, (n,))
    wants_jump = (rand_vals > prob) & grounded & alive

    h_dir = state.orientations[type_idx, :, 1]

    active_row = jnp.where(wants_jump, -strength, 0.0)
    active_col = h_dir * strength * alive.astype(jnp.float32)

    friction_col = jnp.where(grounded, -vel[:, 1] / mass, 0.0)

    new_vel_row = vel[:, 0] + active_row / mass + gravity
    new_vel_col = vel[:, 1] + (active_col + friction_col) / mass
    new_vel = jnp.stack([new_vel_row, new_vel_col], axis=-1)

    # Position update: truncate velocity to int32
    new_pos = state.positions[type_idx] + (new_vel * alive.astype(jnp.float32)[:, None]).astype(jnp.int32)

    new_pf_row = jnp.where(alive, gravity * mass, pf[:, 0])
    new_pf = jnp.stack([new_pf_row, pf[:, 1]], axis=-1)

    return state.replace(
        positions=state.positions.at[type_idx].set(new_pos),
        velocities=state.velocities.at[type_idx].set(new_vel),
        passive_forces=state.passive_forces.at[type_idx].set(new_pf),
        rng=rng,
    )


# ── Continuous physics avatar updates ─────────────────────────────────


def update_inertial_avatar(state: GameState, action, avatar_type, n_move,
                           mass, strength):
    """InertialAvatar: ContinuousPhysics, no gravity. Input = force direction.

    velocity += (direction * strength) / mass
    position += (int32) velocity
    orientation = velocity direction (for rendering)
    """
    is_move = action < n_move
    move_idx = jnp.clip(action, 0, 3)
    force = jax.lax.cond(
        is_move,
        lambda: DIRECTION_DELTAS_F32[move_idx] * strength,
        lambda: jnp.array([0.0, 0.0], dtype=jnp.float32))

    vel = state.velocities[avatar_type, 0]
    new_vel = vel + force / mass
    # Truncate to int32 for position update (matching GVGAI's (int) cast)
    new_pos = state.positions[avatar_type, 0] + new_vel.astype(jnp.int32)

    speed = jnp.sqrt(jnp.sum(new_vel ** 2))
    new_ori = jnp.where(speed > 1e-6, new_vel / speed,
                        state.orientations[avatar_type, 0])

    return state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        velocities=state.velocities.at[avatar_type, 0].set(new_vel),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )


def update_mario_avatar(state: GameState, action, avatar_type,
                        mass, strength, jump_strength, gravity,
                        airsteering):
    """MarioAvatar: GravityPhysics. 6-action space.

    Actions: LEFT=0, RIGHT=1, JUMP=2, JUMP_LEFT=3, JUMP_RIGHT=4, NOOP=5.

    Coordinate system: (row, col) where +row = down.
    Gravity = (+gravity_val, 0) in row direction.
    Jump = negative row velocity (upward).
    """
    h = jnp.where(
        (action == 0) | (action == 3), -1.0,
        jnp.where((action == 1) | (action == 4), 1.0, 0.0))
    wants_jump = (action == 2) | (action == 3) | (action == 4)

    vel = state.velocities[avatar_type, 0]
    pf = state.passive_forces[avatar_type, 0]

    grounded = (pf[0] == 0.0)

    active_row = jnp.where(grounded & wants_jump, -jump_strength, 0.0)
    active_col = jnp.where(
        grounded | airsteering, h * strength, 0.0)
    active_force = jnp.array([active_row, active_col])

    friction_col = jnp.where(
        grounded | airsteering, -vel[1] / mass, 0.0)
    friction_force = jnp.array([0.0, friction_col])

    new_vel = vel + (active_force + friction_force) / mass
    new_vel = new_vel.at[0].add(gravity)

    # Position update: truncate velocity to int32
    new_pos = state.positions[avatar_type, 0] + new_vel.astype(jnp.int32)

    new_pf = jnp.array([gravity * mass, 0.0])

    new_ori = jnp.where(
        jnp.abs(h) > 0.0,
        jnp.array([0.0, h]),
        state.orientations[avatar_type, 0])

    return state.replace(
        positions=state.positions.at[avatar_type, 0].set(new_pos),
        velocities=state.velocities.at[avatar_type, 0].set(new_vel),
        passive_forces=state.passive_forces.at[avatar_type, 0].set(new_pf),
        orientations=state.orientations.at[avatar_type, 0].set(new_ori),
    )
