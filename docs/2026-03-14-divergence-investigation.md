# VGDLx vs GVGAI Divergence Investigation Report

**Date**: 2026-03-14
**State**: 16/42 match, 26 failures
**Method**: 6 parallel investigator agents → 1 synthesis agent → 6 parallel root-cause agents

## Root Causes (ranked by impact)

| # | Root Cause | Games | Regression? | Fix Difficulty |
|---|-----------|-------|-------------|----------------|
| RC1 | RNG replay infra bug (not engine) | 12 | superman | Medium |
| RC2 | rotateInPlace not implemented | 12 | — | Medium |
| RC3 | Shared cooldown_timers (Bomber spawn resets movement timer) | 6 | — | Medium |
| RC4 | Projectile spawn position (at avatar, not 1 cell ahead) | 7 | — | Easy |
| RC5 | `_collision_grid_mask_static_a` missing 2x2 AABB | 1 | butterflies | Medium |
| RC6 | bounceForward `repeat` kwarg ignored + same-type `partner_idx=-1` | 1 | thecitadel | Hard |
| RC7 | SpawnPoint singleton not checked | 1 | waitforbreakfast | Easy |
| RC8 | FlakAvatar ammo check missing | 1 | — | Easy |
| RC9 | RandomNPC cons DNONE resets cooldown | 2 | — | Easy |
| RC10 | Spawn position (0,0) | 3 | — | Needs debugging |

## Detailed Findings

### RC1 — RNG replay infra bug (12 games, not an engine bug)

**Games**: chopper, defender, myAliens, plaqueattack, sheriff, waves, whackamole, + overlap with RC3 games (aliens, eggomania, ikaruga, superman)

**Root cause**: `build_gvgai_rng_record()` in `vgdl_jax/validate/rng_replay.py` (lines 418-441) assigns the **same** roll to ALL spawners of a type. It detects spawn outcome by comparing pre/post alive counts of the target type as a whole:

```python
pre_count = int(pre_alive[target_ti].sum())
post_count = int(post_alive[target_ti].sum())
spawned = post_count > pre_count
roll = 0.0 if spawned else 1.0
```

If VGDLx spawns from 1 out of 16 spawners (with prob=0.01), the function records `roll=0.0` for ALL 16 spawners. When GVGAI reads these injected values, all 16 SpawnPoints pass the probability check and spawn simultaneously.

**Not an engine bug** — VGDLx's per-spawner `prob` check is correct and matches GVGAI's natural behavior (each SpawnPoint draws independently from `game.getRandomGenerator().nextFloat()`).

**Fix**: Track per-spawner spawn outcomes by position matching. For each alive spawner slot, check if a new sprite appeared at that spawner's position in the target type. Assign `roll=0.0` only for spawners whose position matches a newly appeared target sprite, and `roll=1.0` for the rest.

---

### RC2 — rotateInPlace not implemented (12 games)

**Games**: assemblyline, factorymanager, glow, infection, islands, jaws, missilecommand, plaqueattack, rivers, sheriff, surround, wildgunman

**Root cause**: GVGAI `OrientedAvatar.loadDefaults()` sets `rotateInPlace = true` by default. When `rotateInPlace=true`, `GridPhysics.activeMovement()` has a two-step gate:

1. If the action direction differs from current orientation: `_updateOrientation()` fires, returns `MOVEMENT.ROTATE`. No position change.
2. If the action direction matches the current orientation: falls through to `_updatePos()`, which moves the sprite.

VGDLx has zero implementation of `rotateInPlace` — the string does not appear anywhere in the codebase. Every directional action immediately moves the avatar.

**Note**: 2 games (avoidgeorge, defender) have explicit `rotateInPlace=False`, so RC2 does NOT apply to those — only RC4 applies.

**Fix**:
1. Add `rotate_in_place: bool` to `AvatarConfig` in `data_model.py`
2. Parse `rotateInPlace` from game files in `parser.py`
3. Set in `_build_avatar_config()`: `True` if OrientedAvatar subclass AND not overridden to `False`
4. In `_update_avatar_single()`: compare intended orientation vs current. If `rotate_in_place` and different, update orientation only (no position change). Use Python `if` for trace-time dispatch.

---

### RC3 — Shared cooldown_timers: Bomber spawn resets movement timer (6 games)

**Games**: aliens, eggomania, ikaruga, superman, angelsdemons, wildgunman

**Root cause**: `_npc_bomber()` calls `update_spawn_point()` then `update_missile()`, both using the same `state.cooldown_timers[type_idx]`. Since `spawn_cooldown == cooldown` (both set from the VGDL `cooldown` value), every time the timer reaches `cooldown`, the spawn fires first and resets the timer to 0. The subsequent `update_missile()` → `_move_with_cooldown()` sees `cooldown_timers = 0` and `cooldown_ok = (0 >= cooldown)` is always false. **The Bomber can never move.**

In GVGAI, spawning and movement use **independent timing mechanisms**:
- Spawning: `(start + gameTick) % cooldown == 0` — global game tick
- Movement: `cooldown <= lastmove` — per-sprite counter incremented by `preMovement()`, reset only on actual movement

These never interfere with each other.

**Fix**: Add a separate `spawn_timers: [n_types, max_n] int32` field to `GameState`. Change `update_spawn_point()` to use `spawn_timers` instead of `cooldown_timers`. Leave `cooldown_timers` exclusively for movement timing.

---

### RC4 — Projectile spawns at avatar position instead of one cell ahead (7 games)

**Games**: avoidgeorge, islands, jaws, missilecommand, glow, rivers, assemblyline

**Root cause**: In GVGAI's `ShootAvatar.shoot()` (ShootAvatar.java lines 87-102), the projectile is spawned at:

```java
new Vector2d(this.rect.x + dir.x * this.lastrect.width,
             this.rect.y + dir.y * this.lastrect.height)
```

This places the projectile **one cell ahead** of the avatar in the orientation direction.

In VGDLx's `spawn_sprite()` (sprites.py line 257), the projectile is spawned at `state.positions[spawner_type, spawner_idx]` — the avatar's own position.

**Note**: FlakAvatar spawns at own position (`this.rect.x, this.rect.y`) in GVGAI, so VGDLx is correct for FlakAvatar.

**Fix**: In `_update_avatar_single()`, compute `offset = orientation_int * block_size` for ShootAvatar family and pass to `spawn_sprite()` via a new `spawn_offset` parameter. Add a `projectile_offset: bool` flag to `AvatarConfig` (True for ShootAvatar, False for FlakAvatar).

---

### RC5 — `_collision_grid_mask_static_a` missing 2x2 AABB (1 game, REGRESSION)

**Game**: butterflies

**Root cause**: `_collision_grid_mask_static_a()` in step.py (lines 460-468) only checks the **single cell** that each type_b sprite's position truncates to via integer division. It does NOT account for mid-cell sprites' bounding boxes spanning a 2x2 cell region.

The symmetric function `_collision_mask_static_b_grid()` (lines 419-457) correctly handles this by checking all 4 cells and performing pixel AABB verification. `_collision_grid_mask_static_a` was never given this treatment.

**Why regression**: Latent bug. Before tick() refactor, different NPC loop order produced different RNG → different butterfly positions that didn't expose the gap. After refactor, some butterflies land mid-cell overlapping cocoons in adjacent cells.

**Fix**: Make `_collision_grid_mask_static_a()` check all 4 cells of the 2x2 bounding box with pixel AABB verification, mirroring `_collision_mask_static_b_grid()`.

---

### RC6 — bounceForward `repeat` + same-type `partner_idx=-1` (1 game, REGRESSION)

**Game**: thecitadel

**Root cause**: Two independent issues:

1. **`repeat` kwarg ignored**: GVGAI parser duplicates the effect N times. Each copy gets fresh collision detection, enabling chain propagation. VGDLx creates exactly one `CompiledEffect` regardless of `repeat`.

2. **Same-type `partner_idx` always -1**: `_collision_mask()` (grid mode, lines 302-307) returns `partner_idx = jnp.full(eff_a, -1)` for same-type collisions. The `partner_delta` handler checks `valid = (partner_idx >= 0) & mask`, so it's always a no-op. The `pixel_aabb` mode handles same-type correctly.

3. (Minor) Dead `once_guard` key on line 553: checks `'partner_delta'` but actual keys are `'bounce_forward'`/`'pull_with_it'`.

**Fix**:
1. In `_build_compiled_effects()`, read `repeat` from kwargs and duplicate the `CompiledEffect` that many times
2. Add `partner_idx` support for `type_a == type_b` when `need_partner=True` in `_collision_mask()`
3. Fix or remove dead `once_guard` key

---

### RC7 — SpawnPoint singleton not checked (1 game, REGRESSION)

**Game**: waitforbreakfast

**Root cause**: `update_spawn_point()` never checks target type's `singleton` flag. GVGAI's `Game.addSprite()` checks `singletons[typeInt]` and refuses to create the sprite if one already exists. VGDLx parses `singleton` but only uses it for avatar projectiles.

**Fix**: Gate spawn on `~jnp.any(state.alive[target_type])` when `target_singleton=True`. Add `target_singleton: bool` to `SpriteConfig`, populate from `game_def.sprites[target_idx].singleton`.

---

### RC8 — FlakAvatar ammo check missing (1 game)

**Game**: eggomania

**Root cause**: Zero implementation of `ammo`/`minAmmo`/`ammoCost`. Parser doesn't extract, `AvatarConfig` has no fields, no checks before shooting. GVGAI's `FlakAvatar.updateUse()` calls `hasAmmo()` which checks `resources.get(ammoId) > minAmmo`.

**Fix**:
1. Parser: extract `ammo`, `minAmmo`, `ammoCost` from FlakAvatar kwargs
2. Data model: add `ammo_resource_idx`, `min_ammo`, `ammo_cost` to `AvatarConfig`
3. Compiler: look up ammo resource in `resource_name_to_idx`
4. Step: check resource count before shooting, subtract `ammo_cost` after

---

### RC9 — RandomNPC cons DNONE resets cooldown (2 games)

**Games**: surround, infection

**Root cause**: In GVGAI, when `getRandomMove()` returns DNONE (during cons repeat phase), `GridPhysics.activeMovement()` checks `action != DNONE` and skips `_updatePos()` entirely. `lastmove` keeps incrementing, accumulating cooldown across the DNONE period.

In VGDLx, the (0,0) delta is passed to `_move_with_cooldown()`, which treats it as a valid movement attempt and resets `cooldown_timers` to 0 when `cooldown_ok=True`.

**Impact for surround** (dog: cooldown=5, cons=15): VGDLx resets cooldown every 5 ticks during DNONE phase, so dog can't move at tick 16 when GVGAI can (lastmove=16 >= 5).

**Fix**: In `update_random_npc()`, compute `is_dnone = (deltas == 0).all(axis=-1)` per sprite. After `_move_with_cooldown()`, mask: `can_move = can_move & ~is_dnone`.

---

### RC10 — Spawn position (0,0) (3 games, inconclusive)

**Games**: angelsdemons, wildgunman, aliens

**Root cause**: Partly a symptom of RC3 (aliens portal IS at (0,0) — sprite just doesn't move). Wildgunman shows one enemy_slow at (0,0) while niceGuy spawns correctly from the same SpawnPoint mechanism — needs runtime debugging to pinpoint.

**Hypothesis**: Could involve interaction between `prefix_sum_allocate` slot assignment and the specific type indices, or a JAX compilation edge case. Requires printing intermediate values in `update_spawn_point` for wildgunman.

---

## Game → Root Cause Mapping

| Game | Root Causes |
|------|-------------|
| aliens | RC1, RC3, RC10 |
| angelsdemons | RC3, RC10 |
| assemblyline | RC2, RC4 |
| avoidgeorge | RC4 |
| butterflies | RC5 |
| chopper | RC1 |
| defender | RC1, RC4 |
| eggomania | RC3, RC8 |
| factorymanager | RC2 |
| glow | RC2, RC4 |
| ikaruga | RC1, RC3 |
| infection | RC2, RC9 |
| islands | RC2, RC4 |
| jaws | RC2, RC4 |
| missilecommand | RC2, RC4 |
| myAliens | RC1 |
| plaqueattack | RC1, RC2 |
| rivers | RC2, RC4 |
| sheriff | RC1, RC2 |
| superman | RC1, RC3 |
| surround | RC2, RC9 |
| thecitadel | RC6 |
| waitforbreakfast | RC7 |
| waves | RC1 |
| whackamole | RC1 |
| wildgunman | RC2, RC10 |

## Suggested Fix Order

1. **RC1** (validation infra) — fixes 12 game comparisons without engine changes
2. **RC3** (Bomber shared timers) — fixes 6 games, highest engine impact
3. **RC2** (rotateInPlace) — fixes 12 games
4. **RC4** (projectile position) — easy fix, 7 games
5. **RC7** (singleton) — easy fix, regression
6. **RC5** (static_a AABB) — regression
7. **RC8** (FlakAvatar ammo) — easy fix
8. **RC9** (RandomNPC cons) — easy fix
9. **RC6** (bounceForward repeat) — hardest fix
10. **RC10** (spawn (0,0)) — needs runtime debugging
