"""
RNG replay infrastructure for cross-engine validation.

Two components:
1. RNGRecorder — eagerly replays VGDLx's RNG split sequence to produce a
   per-step record of all random draws (direction indices, spawn rolls, etc.)
2. ReplayRandomGenerator — drop-in replacement for py-vgdl's random.Random
   that returns pre-recorded values instead of generating new ones.

For GVGAI validation, build_gvgai_rng_records() + write_gvgai_rng_file()
produce a JSON file that GVGAI's TraceAgent reads to inject matching RNG.

Together, these ensure both engines use identical random outcomes.
"""
import jax
import jax.numpy as jnp
import numpy as np

from vgdl_jax.data_model import SpriteClass, STATIC_CLASSES, AVATAR_CLASSES

# ── Direction mapping ─────────────────────────────────────────────────
#
# JAX DIRECTION_DELTAS: {0:UP(-1,0), 1:DOWN(1,0), 2:LEFT(0,-1), 3:RIGHT(0,1)}
#   (row, col) coordinates
#
# py-vgdl BASEDIRS: [UP(0,-1), LEFT(-1,0), DOWN(0,1), RIGHT(1,0)]
#   (x, y) pixel coordinates, indices 0-3
#   UP=0, LEFT=1, DOWN=2, RIGHT=3
#
# Mapping from JAX dir index → BASEDIRS index:
JAX_TO_BASEDIRS = {0: 0, 1: 2, 2: 1, 3: 3}


class RNGRecorder:
    """Eagerly replay vgdl-jax's RNG sequence to record all random draws.

    Walks the same split path as _step_inner in step.py, recording every
    random value that the JAX engine would produce. This record can then
    be fed into ReplayRandomGenerator to make py-vgdl follow the same path.
    """

    def __init__(self, sprite_configs, effects, game_def):
        """
        Args:
            sprite_configs: list of dicts from CompiledGame (same as step.py receives)
            effects: list of compiled effect dicts
            game_def: GameDef for sprite key resolution
        """
        self.sprite_configs = sprite_configs
        self.effects = effects
        self.game_def = game_def
        self.max_n = None  # set on first record_step call

    def record_step(self, rng_key, max_n=None):
        """Walk _step_inner's RNG split path and record all draws.

        Args:
            rng_key: current JAX PRNGKey (state.rng before the step)
            max_n: max sprites per type (from state.alive.shape[1])

        Returns:
            (record, next_rng): where record is a dict:
                {type_idx: {
                    'class': SpriteClass int,
                    'key': str,
                    'dir_indices': np.array of shape [max_n] (for RandomNPC/Chaser/Fleeing),
                    'spawn_rolls': np.array of shape [max_n] (for SpawnPoint/Bomber),
                }}
            and next_rng is the consumed PRNGKey.
        """
        if max_n is not None:
            self.max_n = max_n
        assert self.max_n is not None, "max_n must be provided on first call"

        rng = rng_key
        record = {}

        # Walk NPC update order in REVERSE (matching step.py's reverse loop)
        for type_idx in range(len(self.sprite_configs) - 1, -1, -1):
            cfg = self.sprite_configs[type_idx]
            sc = cfg['sprite_class']
            if sc in AVATAR_CLASSES or sc in STATIC_CLASSES:
                continue

            sprite_key = self.game_def.sprites[type_idx].key

            if sc == SpriteClass.RANDOM_NPC:
                rng, key = jax.random.split(rng)
                dir_indices = np.array(jax.random.randint(key, (self.max_n,), 0, 4))
                record[type_idx] = {
                    'class': sc, 'key': sprite_key,
                    'dir_indices': dir_indices,
                }

            elif sc in (SpriteClass.CHASER, SpriteClass.FLEEING):
                rng, key = jax.random.split(rng)
                # Fallback random dirs (always consumed even when targets alive)
                dir_indices = np.array(jax.random.randint(key, (self.max_n,), 0, 4))
                record[type_idx] = {
                    'class': sc, 'key': sprite_key,
                    'dir_indices': dir_indices,
                }

            elif sc == SpriteClass.SPAWN_POINT:
                rng, key = jax.random.split(rng)
                spawn_rolls = np.array(jax.random.uniform(key, (self.max_n,)))
                record[type_idx] = {
                    'class': sc, 'key': sprite_key,
                    'spawn_rolls': spawn_rolls,
                }

            elif sc == SpriteClass.BOMBER:
                # Missile update: no RNG
                # SpawnPoint update: consumes RNG
                rng, key = jax.random.split(rng)
                spawn_rolls = np.array(jax.random.uniform(key, (self.max_n,)))
                record[type_idx] = {
                    'class': sc, 'key': sprite_key,
                    'spawn_rolls': spawn_rolls,
                }
            # MISSILE, FLICKER, ORIENTED_FLICKER, WALKER: no RNG

        # Effects: teleport_to_exit consumes RNG
        for eff in self.effects:
            if eff.get('effect_type') == 'teleport_to_exit':
                exit_type = eff.get('kwargs', {}).get('exit_type_idx', -1)
                if exit_type >= 0:
                    rng, key = jax.random.split(rng)
                    # Record the key for teleport destination selection
                    record[('teleport', eff['type_a'], eff['type_b'])] = {
                        'class': 'teleport_to_exit',
                        'key_array': np.array(key),
                    }

        return record, rng


def patch_chaser_directions(record, prev_jax_state, sprite_configs,
                            height, width, block_size=1):
    """Recompute actual chaser/fleeing directions using the distance field.

    Instead of extracting directions from position deltas (which fails for
    stepBacked sprites), this recomputes the distance field and argmin/argmax
    exactly as update_chaser does, producing the correct intended direction.

    Args:
        record: the step record from RNGRecorder.record_step()
        prev_jax_state: JAX GameState BEFORE the step (same state chaser sees)
        sprite_configs: list of sprite config dicts
        height: grid height
        width: grid width
        block_size: pixels per cell
    """
    from vgdl_jax.sprites import _manhattan_distance_field

    for type_idx, cfg in enumerate(sprite_configs):
        sc = cfg['sprite_class']
        if sc not in (SpriteClass.CHASER, SpriteClass.FLEEING):
            continue
        if type_idx not in record:
            continue

        fleeing = (sc == SpriteClass.FLEEING)
        target_type_idx = cfg.get('target_type_idx', 0)

        chaser_pos = prev_jax_state.positions[type_idx]  # int32 pixels
        target_pos = prev_jax_state.positions[target_type_idx]
        target_alive = prev_jax_state.alive[target_type_idx]
        any_target_alive = bool(jnp.any(target_alive))

        if not any_target_alive:
            # No targets — keep fallback random direction (already recorded)
            continue

        # Recompute distance field (same as update_chaser)
        dist_field = _manhattan_distance_field(target_pos, target_alive,
                                               height, width, block_size)

        # Pixel→cell for distance field lookup
        chaser_cells = chaser_pos // block_size
        r = np.array(jnp.clip(chaser_cells[:, 0], 0, height - 1))
        c = np.array(jnp.clip(chaser_cells[:, 1], 0, width - 1))
        INF = height + width
        d_up = np.where(r > 0,
                        np.array(dist_field)[np.clip(r - 1, 0, height - 1), c], INF)
        d_down = np.where(r < height - 1,
                          np.array(dist_field)[np.clip(r + 1, 0, height - 1), c], INF)
        d_left = np.where(c > 0,
                          np.array(dist_field)[r, np.clip(c - 1, 0, width - 1)], INF)
        d_right = np.where(c < width - 1,
                           np.array(dist_field)[r, np.clip(c + 1, 0, width - 1)], INF)

        neighbor_dists = np.stack([d_up, d_down, d_left, d_right], axis=-1)
        if fleeing:
            best_dir = np.argmax(neighbor_dists, axis=-1)
        else:
            best_dir = np.argmin(neighbor_dists, axis=-1)

        record[type_idx]['dir_indices'] = best_dir.astype(np.int32)


class ReplayRandomGenerator:
    """Drop-in replacement for py-vgdl's random.Random.

    When injected into a py-vgdl game via set_random_generator(), intercepts
    random calls from sprites and returns pre-recorded values from RNGRecorder.

    The game's tick() loop calls notify_sprite_update() before each sprite's
    update(), allowing us to track which sprite is currently consuming RNG.

    Usage:
        recorder = RNGRecorder(sprite_configs, effects, game_def)
        replay = ReplayRandomGenerator(game_def)

        for step, action in enumerate(actions):
            record, rng_key = recorder.record_step(rng_key, max_n)
            replay.set_step_record(record)
            game.tick(action)
    """

    def __init__(self, game_def):
        """
        Args:
            game_def: GameDef for key->type_idx mapping
        """
        self._key_to_type_idx = {}
        for sd in game_def.sprites:
            self._key_to_type_idx[sd.key] = sd.type_idx

        self._record = {}
        self._current_key = None
        self._current_slot = 0

    def set_step_record(self, record):
        """Set the pre-recorded random values for this step."""
        self._record = record

    def notify_sprite_update(self, sprite_key, slot_idx):
        """Called by the modified tick() loop before each sprite.update().

        Args:
            sprite_key: the sprite's key (e.g. 'alien', 'catcher')
            slot_idx: 0-indexed position within sprites of this key
        """
        self._current_key = sprite_key
        self._current_slot = slot_idx

    def _get_type_record(self):
        """Look up the record for the current sprite's type."""
        if self._current_key is None:
            return None
        type_idx = self._key_to_type_idx.get(self._current_key)
        if type_idx is None:
            return None
        return self._record.get(type_idx)

    def choice(self, options):
        """Replace random.choice — used by RandomNPC, Chaser, Fleeing, teleport.

        For direction choices (options is a list of tuples), look up the recorded
        JAX direction index and map it to the corresponding BASEDIRS entry.

        For non-direction choices (e.g., teleport exit sprites), use the recorded
        JAX key to deterministically pick the same index JAX would.
        """
        if not options:
            return None

        # Detect direction choice: options are tuples or Vector2 (len-2 sequences)
        # Non-direction choices (e.g. teleport exit Sprite objects) go to _teleport_choice
        try:
            _ = options[0][0], options[0][1]
            is_direction = (len(options[0]) == 2)
        except (TypeError, IndexError):
            is_direction = False

        if not is_direction:
            return self._teleport_choice(options)

        rec = self._get_type_record()
        if rec is not None and 'dir_indices' in rec:
            jax_dir = int(rec['dir_indices'][self._current_slot])

            # Check if options is BASEDIRS (4 directions)
            if len(options) == 4:
                basedirs_idx = JAX_TO_BASEDIRS[jax_dir]
                return options[basedirs_idx]

            # For Chaser/Fleeing: options is a subset of BASEDIRS (good moves).
            # After patch_chaser_directions, dir_indices contains the actual
            # direction JAX moved. Map it to BASEDIRS and return it —
            # even if not in options — to match JAX behavior.
            from vgdl.ontology.constants import BASEDIRS
            basedirs_idx = JAX_TO_BASEDIRS[jax_dir]
            target_dir = BASEDIRS[basedirs_idx]

            # Try to find it in options
            for opt in options:
                if (opt[0] == target_dir[0] and opt[1] == target_dir[1]):
                    return opt

            # If JAX's chosen direction isn't in options, return it anyway
            # to match JAX behavior (e.g., distance field sees through walls)
            return target_dir

        # Fallback: no record, return first option
        if options:
            return options[0]
        return None

    def _teleport_choice(self, options):
        """Handle teleport exit selection using recorded JAX key."""
        # Find matching teleport record
        for key, rec in self._record.items():
            if isinstance(key, tuple) and key[0] == 'teleport':
                key_array = rec.get('key_array')
                if key_array is not None:
                    jax_key = jnp.array(key_array, dtype=jnp.uint32)
                    idx = int(jax.random.randint(jax_key, (), 0, len(options)))
                    return options[idx]
        # Fallback: first option
        return options[0]

    def random(self):
        """Replace random.random() — used by SpawnPoint prob check.

        Returns the recorded uniform [0,1) value for this sprite slot.
        """
        rec = self._get_type_record()
        if rec is not None and 'spawn_rolls' in rec:
            return float(rec['spawn_rolls'][self._current_slot])
        # Fallback: always succeed spawn check
        return 0.0

    def randint(self, a, b):
        """Replace random.randint(a, b) — rarely used in standard games."""
        rec = self._get_type_record()
        if rec is not None and 'dir_indices' in rec:
            return int(rec['dir_indices'][self._current_slot]) % (b - a + 1) + a
        return a

    def seed(self, s):
        """No-op for compatibility with game.set_seed()."""
        pass


# ── GVGAI RNG Injection ─────────────────────────────────────────────

# Orientation (row, col) → DBASEDIRS index.
# DBASEDIRS: [DUP(0,-1), DLEFT(-1,0), DDOWN(0,1), DRIGHT(1,0)]
# In (row, col): UP=(-1,0)→0, DOWN=(1,0)→2, LEFT=(0,-1)→1, RIGHT=(0,1)→3
_ORI_TO_BASEDIRS = {(-1, 0): 0, (0, -1): 1, (1, 0): 2, (0, 1): 3}


def build_gvgai_rng_record(pre_state, post_state, game_def, block_size,
                           spawn_target_map=None):
    """Build one step's GVGAI RNG injection record from VGDLx state transition.

    For direction-consuming sprites (RandomNPC, Chaser, Fleeing):
      Extract effective direction from post-step orientation.

    For spawn-consuming sprites (SpawnPoint, Bomber):
      Compare pre/post alive counts to determine spawn outcome.

    For teleportToExit effects:
      Detect position change to exit portal position, record exit coords.

    For flipDirection effects:
      Record post-step orientation of actor sprites.

    Args:
        pre_state: GameState before step
        post_state: GameState after step
        game_def: GameDef for sprite metadata
        block_size: GVGAI pixel block size
        spawn_target_map: optional dict {type_idx: target_type_idx} for spawn outcome detection

    Returns:
        dict: {sprite_key: [{"pos": [py, px], "dir"|"roll"|"flip_dir"|"teleport_exit": ...}, ...]}
    """
    from vgdl_jax.data_model import SPRITE_REGISTRY

    record = {}

    # Bulk-convert JAX arrays to numpy once (avoids per-element device→host transfers)
    pre_alive = np.array(pre_state.alive)
    pre_pos = np.array(pre_state.positions)
    post_pos = np.array(post_state.positions)
    post_ori = np.array(post_state.orientations)
    post_alive = np.array(post_state.alive)

    for sd in game_def.sprites:
        ti = sd.type_idx
        sc = sd.sprite_class
        sc_def = SPRITE_REGISTRY.get(sc)
        if sc_def is None:
            continue

        # Direction-consuming NPC types
        if sc in (SpriteClass.RANDOM_NPC, SpriteClass.CHASER, SpriteClass.FLEEING):
            alive_mask = pre_alive[ti]
            slots = np.where(alive_mask)[0]
            if len(slots) == 0:
                continue
            entries = []
            for slot in slots:
                r, c = pre_pos[ti, slot, 0], pre_pos[ti, slot, 1]
                ori_r, ori_c = post_ori[ti, slot, 0], post_ori[ti, slot, 1]
                basedirs_idx = _ORI_TO_BASEDIRS.get(
                    (round(float(ori_r)), round(float(ori_c))), -1)
                entries.append({
                    "pos": [float(r), float(c)],  # already in pixels
                    "dir": basedirs_idx,
                })
            record[sd.key] = entries

        # Spawn-consuming types
        elif sc in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            alive_mask = pre_alive[ti]
            slots = np.where(alive_mask)[0]
            if len(slots) == 0:
                continue

            # Detect spawn outcome from target type alive count change
            target_ti = spawn_target_map.get(ti) if spawn_target_map else None
            if target_ti is not None:
                pre_count = int(pre_alive[target_ti].sum())
                post_count = int(post_alive[target_ti].sum())
                spawned = post_count > pre_count
                roll = 0.0 if spawned else 1.0
            else:
                roll = 0.0  # no target info — default to spawn succeeds

            entries = []
            for slot in slots:
                r, c = pre_pos[ti, slot, 0], pre_pos[ti, slot, 1]
                entries.append({
                    "pos": [float(r), float(c)],  # already in pixels
                    "roll": roll,
                })
            record[sd.key] = entries

    # --- Effect-level RNG: teleportToExit ---
    # Detect teleport outcomes by comparing actor positions to exit portal positions.
    # The recorded pos is the entrance portal position (= avatar's position when effect
    # fires on the GVGAI side, after movement to the portal cell).
    for ed in game_def.effects:
        if ed.effect_type != 'teleport_to_exit':
            continue
        actor_indices = game_def.resolve_stype(ed.actor_stype)
        entrance_indices = game_def.resolve_stype(ed.actee_stype)
        if not entrance_indices:
            continue
        ent_ti = entrance_indices[0]
        ent_sd = game_def.sprites[ent_ti]
        if not ent_sd.portal_exit_stype:
            continue
        exit_indices = game_def.resolve_stype(ent_sd.portal_exit_stype)
        if not exit_indices:
            continue
        exit_ti = exit_indices[0]

        exit_alive_mask = pre_alive[exit_ti]
        exit_slots = np.where(exit_alive_mask)[0]
        ent_alive_mask = pre_alive[ent_ti]
        ent_slots = np.where(ent_alive_mask)[0]

        for actor_ti in actor_indices:
            key = game_def.sprites[actor_ti].key
            alive_mask = pre_alive[actor_ti]
            for slot in np.where(alive_mask)[0]:
                a_pre = pre_pos[actor_ti, slot]
                a_post = post_pos[actor_ti, slot]
                if np.array_equal(a_pre, a_post):
                    continue  # no position change

                # Check if post position matches an exit portal
                matched_exit = None
                for es in exit_slots:
                    ep = pre_pos[exit_ti, es]
                    if abs(int(a_post[0]) - int(ep[0])) + abs(int(a_post[1]) - int(ep[1])) < 2:
                        matched_exit = ep
                        break
                if matched_exit is None:
                    continue  # position change was movement, not teleport

                # Find entrance portal closest to actor's pre-step position
                best_ent = None
                best_d = float('inf')
                for es in ent_slots:
                    ep = pre_pos[ent_ti, es]
                    d = abs(int(a_pre[0]) - int(ep[0])) + abs(int(a_pre[1]) - int(ep[1]))
                    if d < best_d:
                        best_d = d
                        best_ent = ep
                if best_ent is not None and best_d <= block_size:
                    record.setdefault(key, []).append({
                        "pos": [float(best_ent[0]), float(best_ent[1])],
                        "teleport_exit": [float(matched_exit[0]), float(matched_exit[1])],
                    })

    # --- Effect-level RNG: flipDirection ---
    # Record post-step orientation for all alive actor sprites of flipDirection effects.
    # Java side only consumes for sprites that actually collided (position-matched lookup).
    for ed in game_def.effects:
        if ed.effect_type != 'flip_direction':
            continue
        actor_indices = game_def.resolve_stype(ed.actor_stype)
        for actor_ti in actor_indices:
            key = game_def.sprites[actor_ti].key
            alive_mask = pre_alive[actor_ti]
            for slot in np.where(alive_mask)[0]:
                ori_r, ori_c = post_ori[actor_ti, slot, 0], post_ori[actor_ti, slot, 1]
                basedirs_idx = _ORI_TO_BASEDIRS.get(
                    (round(float(ori_r)), round(float(ori_c))), -1)
                if basedirs_idx < 0:
                    continue
                # Use post-state position (effects fire after movement)
                pr, pc = post_pos[actor_ti, slot, 0], post_pos[actor_ti, slot, 1]
                record.setdefault(key, []).append({
                    "pos": [float(pr), float(pc)],
                    "flip_dir": basedirs_idx,
                })

    return record


def build_gvgai_rng_records(compiled, game_def, actions, seed=42):
    """Run VGDLx trajectory and build per-step GVGAI RNG injection records.

    Args:
        compiled: CompiledGame from compile_game()
        game_def: GameDef
        actions: list of int action indices
        seed: random seed

    Returns:
        (records, raw_states) where records is a list of dicts (one per step)
        and raw_states is the list of GameState objects [init, step1, step2, ...].
    """
    import jax
    from vgdl_jax.data_model import get_block_size

    block_size = get_block_size(game_def)

    # Build spawn target map: {spawner_type_idx: target_type_idx}
    spawn_target_map = {}
    for sd in game_def.sprites:
        if sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            if sd.spawner_stype:
                targets = game_def.resolve_stype(sd.spawner_stype)
                if targets:
                    spawn_target_map[sd.type_idx] = targets[0]

    state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))
    raw_states = [state]
    records = []

    for action in actions:
        if bool(state.done):
            break
        pre_state = state
        state = compiled.step_fn(pre_state, action)
        raw_states.append(state)
        record = build_gvgai_rng_record(pre_state, state, game_def, block_size,
                                        spawn_target_map=spawn_target_map)
        records.append(record)

    return records, raw_states


def write_gvgai_rng_file(records, path):
    """Write GVGAI RNG injection records to JSON file.

    Args:
        records: list of dicts from build_gvgai_rng_records()
        path: output file path
    """
    import json
    with open(path, 'w') as f:
        json.dump(records, f)
