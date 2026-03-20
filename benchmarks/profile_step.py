"""
Profile the compiled step function to identify runtime bottlenecks.

Analyzes:
1. XLA HLO cost analysis (FLOPs, memory)
2. Phase-level timing via decomposed step function
3. Identifies which game components are most expensive
"""
import os
import time
import jax
import jax.numpy as jnp
from vgdl_jax.env import VGDLxEnv
from vgdl_jax.parser import parse_vgdl
from vgdl_jax.compiler import compile_game

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def profile_hlo(game_name, game_file, level_file):
    """Analyze the compiled XLA HLO for a game's step function."""
    print(f"\n{'='*70}")
    print(f"  HLO Analysis: {game_name}")
    print(f"{'='*70}")

    env = VGDLxEnv(game_file, level_file)
    compiled = env.compiled

    print(f"  n_types={len(compiled.game_def.sprites)}, "
          f"max_n={compiled.init_state.alive.shape[1]}, "
          f"n_actions={env.n_actions}")

    # Count compiled effects and their types
    game_def = compiled.game_def
    # Access the step fn's closure to count effects
    # Instead, count from game_def
    effect_counts = {}
    for ed in game_def.effects:
        et = ed.effect_type
        effect_counts[et] = effect_counts.get(et, 0) + 1
    print(f"\n  Effect types ({len(game_def.effects)} total rules):")
    for et, cnt in sorted(effect_counts.items(), key=lambda x: -x[1]):
        print(f"    {et}: {cnt}")

    # Count NPC types
    from vgdl_jax.data_model import SpriteClass, STATIC_CLASSES, AVATAR_CLASSES
    # Build reverse map from SpriteClass int values to names
    sc_names = {v.value if hasattr(v, 'value') else v: k
                for k, v in SpriteClass.__members__.items()} if hasattr(SpriteClass, '__members__') else {}
    npc_counts = {}
    for sd in game_def.sprites:
        if sd.sprite_class not in STATIC_CLASSES and sd.sprite_class not in AVATAR_CLASSES:
            sc_name = sc_names.get(sd.sprite_class, str(sd.sprite_class))
            npc_counts[sc_name] = npc_counts.get(sc_name, 0) + 1
    if npc_counts:
        print(f"\n  NPC sprite types:")
        for name, cnt in sorted(npc_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {cnt}")

    # Check for expensive features
    has_chaser = any(sd.sprite_class == SpriteClass.CHASER for sd in game_def.sprites)
    has_spawn = any(sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER)
                    for sd in game_def.sprites)
    has_continuous = any(sd.physics_type in ('continuous', 'gravity')
                         for sd in game_def.sprites)
    has_sweep = any(sd.speed > 1.0 for sd in game_def.sprites)
    has_fori_effects = any(ed.effect_type in ('kill_sprite', 'step_back')
                           for ed in game_def.effects)

    print(f"\n  Expensive features:")
    print(f"    Chaser (distance field): {has_chaser}")
    print(f"    SpawnPoint/Bomber: {has_spawn}")
    print(f"    Continuous physics (AABB): {has_continuous}")
    print(f"    Sweep collision (speed>1): {has_sweep}")

    # Lowered HLO analysis
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(0))
    action = jnp.int32(0)

    # Lower and compile to get cost analysis
    lowered = jax.jit(compiled.step_fn).lower(state, action)
    xla_compiled = lowered.compile()

    cost = xla_compiled.cost_analysis()
    if cost:
        # cost_analysis returns a list of dicts (one per device)
        if isinstance(cost, list):
            cost = cost[0]
        print(f"\n  XLA cost analysis:")
        for k, v in sorted(cost.items()):
            if isinstance(v, (int, float)):
                print(f"    {k}: {v:,.0f}")

    # Count HLO ops by looking at the text representation
    hlo_text = lowered.as_text()
    op_counts = {}
    for line in hlo_text.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('//') and not line.startswith('{'):
            # Extract op name (after the = sign)
            parts = line.split('=')
            if len(parts) >= 2:
                rhs = parts[-1].strip()
                op_name = rhs.split('(')[0].strip().split()[-1] if rhs else ''
                if op_name and op_name[0].isalpha():
                    op_counts[op_name] = op_counts.get(op_name, 0) + 1

    print(f"\n  HLO op counts (top 15):")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {op}: {cnt}")

    total_ops = sum(op_counts.values())
    print(f"    TOTAL: {total_ops}")

    return env, compiled


def profile_phases(env, compiled, game_name, n_envs=256, n_warmup=5, n_measure=50):
    """Time individual phases of the step function."""
    print(f"\n{'='*70}")
    print(f"  Phase Timing: {game_name} ({n_envs} envs, {n_measure} steps)")
    print(f"{'='*70}")

    # Setup batched state
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, n_envs)
    _, state_batch = jax.vmap(env.reset)(rngs)

    step_vmap = jax.jit(jax.vmap(env.step))

    # Warmup
    for _ in range(n_warmup):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (n_envs,), 0, env.n_actions)
        _, state_batch, _, _, _ = step_vmap(state_batch, actions)
    jax.block_until_ready(state_batch.positions)

    # Time full step
    rng, key = jax.random.split(rng)
    actions = jax.random.randint(key, (n_envs,), 0, env.n_actions)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        _, state_batch, _, _, _ = step_vmap(state_batch, actions)
    jax.block_until_ready(state_batch.positions)
    t_full = (time.perf_counter() - t0) / n_measure
    sps = n_envs / t_full
    print(f"\n  Full step: {t_full*1000:.3f}ms  ({sps:,.0f} steps/sec)")

    # Time just observation (the _get_obs call inside env.step)
    obs_fn = jax.jit(jax.vmap(env._get_obs))
    # Warmup obs
    obs_fn(state_batch)
    jax.block_until_ready(obs_fn(state_batch))
    t0 = time.perf_counter()
    for _ in range(n_measure):
        obs = obs_fn(state_batch)
    jax.block_until_ready(obs)
    t_obs = (time.perf_counter() - t0) / n_measure
    print(f"  Obs (_get_obs): {t_obs*1000:.3f}ms  ({t_obs/t_full*100:.1f}%)")

    # Time just the step_fn (without obs)
    step_only = jax.jit(jax.vmap(compiled.step_fn))
    step_only(state_batch, actions)  # warmup
    jax.block_until_ready(step_only(state_batch, actions).positions)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        new_state = step_only(state_batch, actions)
    jax.block_until_ready(new_state.positions)
    t_step = (time.perf_counter() - t0) / n_measure
    print(f"  Step (no obs): {t_step*1000:.3f}ms  ({t_step/t_full*100:.1f}%)")
    print(f"  Overhead (obs+reward+done): {(t_full-t_step)*1000:.3f}ms  ({(t_full-t_step)/t_full*100:.1f}%)")


def profile_step_decomposed(game_name, game_file, level_file, n_envs=256,
                             n_warmup=5, n_measure=50):
    """Decompose step function into phases and time each one."""
    print(f"\n{'='*70}")
    print(f"  Decomposed Step: {game_name}")
    print(f"{'='*70}")

    game_def = parse_vgdl(game_file, level_file)
    compiled = compile_game(game_def)

    from vgdl_jax.data_model import SpriteClass, STATIC_CLASSES, AVATAR_CLASSES
    from vgdl_jax.step import (
        _build_occupancy_grid, _collision_mask, _collision_mask_aabb,
        _collision_mask_sweep, _apply_all_effects,
    )
    from vgdl_jax.sprites import (
        update_missile, update_random_npc, update_chaser, update_spawn_point,
    )
    from vgdl_jax.terminations import check_all_terminations

    state = compiled.init_state.replace(rng=jax.random.PRNGKey(0))
    max_n = state.alive.shape[1]
    height = game_def.level.height
    width = game_def.level.width

    # Setup batched
    rngs = jax.random.split(jax.random.PRNGKey(0), n_envs)
    env = VGDLxEnv(game_file, level_file)
    _, state_batch = jax.vmap(env.reset)(rngs)
    jax.block_until_ready(state_batch.positions)

    # Count chasers, missiles, random NPCs
    sprite_configs = []
    for sd in game_def.sprites:
        if sd.sprite_class in STATIC_CLASSES or sd.sprite_class in AVATAR_CLASSES:
            continue
        sprite_configs.append((sd.type_idx, sd.sprite_class, sd.key))

    # Check if game has chasers
    chaser_types = [(idx, key) for idx, sc, key in sprite_configs
                    if sc == SpriteClass.CHASER]
    missile_types = [(idx, key) for idx, sc, key in sprite_configs
                     if sc == SpriteClass.MISSILE]

    if chaser_types:
        print(f"\n  Chaser types: {chaser_types}")
        print(f"  Grid size: {height}x{width} = {height*width} cells")
        print(f"  Distance field iterations: {height+width}")
        print(f"  Distance field work per step: ~{height*width*(height+width):,} ops")

    # Count compiled effects that will execute
    n_compiled_effects = 0
    n_aabb = 0
    n_sweep = 0
    n_grid = 0
    from vgdl_jax.effects import VGDL_TO_KEY
    for ed in game_def.effects:
        actor_indices = game_def.resolve_stype(ed.actor_stype)
        if ed.actee_stype == 'EOS':
            n_compiled_effects += len(actor_indices)
            continue
        actee_indices = game_def.resolve_stype(ed.actee_stype)
        for ta in actor_indices:
            for tb in actee_indices:
                n_compiled_effects += 1
                ta_sd = game_def.sprites[ta]
                tb_sd = game_def.sprites[tb]
                if ta_sd.speed > 1.0 or tb_sd.speed > 1.0:
                    n_sweep += 1
                elif (ta_sd.physics_type in ('continuous', 'gravity') or
                      tb_sd.physics_type in ('continuous', 'gravity') or
                      ta_sd.speed != 1.0 or tb_sd.speed != 1.0):
                    n_aabb += 1
                else:
                    n_grid += 1

    print(f"\n  Compiled effects: {n_compiled_effects} total")
    print(f"    Grid collision: {n_grid}")
    print(f"    AABB collision: {n_aabb}")
    print(f"    Sweep collision: {n_sweep}")

    # Estimate relative cost
    print(f"\n  Estimated relative costs (per step, per env):")
    print(f"    Occupancy grid build: ~{n_grid} x O({max_n}) = O({n_grid * max_n})")
    if n_aabb > 0:
        print(f"    AABB pairwise: ~{n_aabb} x O({max_n}^2) = O({n_aabb * max_n * max_n})")
    if chaser_types:
        n_chasers = len(chaser_types)
        print(f"    Distance field: ~{n_chasers} x O({height}*{width}*{height+width}) = O({n_chasers * height * width * (height + width)})")


if __name__ == '__main__':
    games = [
        ("Chase", "chase.txt", "chase_lvl0.txt"),
        ("Zelda", "zelda.txt", "zelda_lvl0.txt"),
        ("Aliens", "aliens.txt", "aliens_lvl0.txt"),
        ("Sokoban", "sokoban.txt", "sokoban_lvl0.txt"),
        ("BoulderDash", "boulderdash.txt", "boulderdash_lvl0.txt"),
        ("SurviveZombies", "survivezombies.txt", "survivezombies_lvl0.txt"),
        ("Frogs", "frogs.txt", "frogs_lvl0.txt"),
    ]

    for name, gf, lf in games:
        game_file = os.path.join(GAMES_DIR, gf)
        level_file = os.path.join(GAMES_DIR, lf)
        try:
            env, compiled = profile_hlo(name, game_file, level_file)
            profile_phases(env, compiled, name)
            profile_step_decomposed(name, game_file, level_file)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  Done.")
