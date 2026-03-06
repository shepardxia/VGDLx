# Engine Comparison: GVGAI vs VGDLx

Open behavioral divergences between GVGAI and VGDLx. Historical 1.0/2.0
differences are in the appendix — VGDLx inherits VGDL 2.0 behavior.

- **GVGAI**: General Video Game AI framework (Java, `GVGAI/src/`)
- **VGDLx**: JAX-compiled port of VGDL 2.0

---

## Summary Table

| # | Divergence | Severity | Notes |
|---|-----------|----------|-------|
| **Game Loop** | | | |
| 2 | Sprite update order | MODERATE | GVGAI reverse z-order; VGDLx compiler-defined |
| 3 | Kill list clearing | MODERATE | GVGAI explicit clearAll phase; VGDLx alive mask |
| 4 | Effect application timing | MODERATE | GVGAI 3-phase (time+EOS+collision); VGDLx fori_loop + batch |
| **Physics** | | | |
| 5 | GridPhysics speed model | HIGH | GVGAI pixel displacement; VGDLx float multiplier |
| 7 | MarioAvatar.strength | MODERATE | GVGAI speed/mass; VGDLx strength/jump_strength |
| 8 | ContinuousPhysics friction | MODERATE | GVGAI has friction; VGDLx removed |
| 9 | NoFrictionPhysics | LOW | GVGAI has class; VGDLx does not |
| **Effects** | | | |
| 10 | killIfSlow speed calc | HIGH | GVGAI orientation divergence; VGDLx scalar speed diff |
| 12 | transformTo state transfer | HIGH | GVGAI copies ori+resources+health+player; VGDLx ori+resources |
| 13 | wallStop friction | MINOR | All four differ; no game uses friction kwarg |
| 14 | wallStop once_per_step | MODERATE | GVGAI has guard; VGDLx missing |
| 15 | wallStop position correction | MINOR | GVGAI pixel-perfect; VGDLx grid-cell +-1 |
| 16 | wallBounce batch handling | MODERATE | GVGAI proximity-sorted batch; VGDLx per-pair |
| 17 | pullWithIt ContinuousPhysics | MODERATE | GVGAI full support; VGDLx position-delta only |
| 18 | pullWithIt once_per_step | MODERATE | GVGAI has guard; VGDLx missing |
| **Collision** | | | |
| 22 | Collision detection method | MINOR | GVGAI integer AABB; VGDLx grid occupancy or float AABB |
| 23 | Continuous-physics threshold | MINOR | GVGAI integer AABB; VGDLx `1.0 - 1e-3` |
| **NPCs** | | | |
| 24 | Chaser pathfinding | MODERATE | GVGAI greedy Manhattan; VGDLx distance field (better) |
| 25 | RandomNPC consecutive moves | MODERATE | GVGAI `cons` param; VGDLx always random |
| 26 | SpawnPoint cooldown model | MODERATE | GVGAI offset-based; VGDLx per-sprite timers |
| **GVGAI-Only** | | | |
| 27 | StopCounter gating | MODERATE | |
| 28 | count_score on termination | LOW | |
| 29 | Shield/immunity system | MODERATE | |
| 30 | Batch effect dispatch | MODERATE | |
| 31 | Time effects (priority queue) | MODERATE | |
| 32 | Health point system | MODERATE | |
| 33 | Per-player scoreChange | LOW | |

---

## 1. Game Loop

### 1.1 Sprite Update Order (MODERATE)

- **GVGAI** (`Game.java:1358-1384`): Two-phase — avatars first, then other sprites
  in **reverse z-order**.
- **VGDLx**: Per-type in compiler-defined order. Avatar movement is a separate
  phase before NPC updates.

### 1.2 Kill List Clearing (MODERATE)

- **GVGAI** (`Game.java:1667-1691`): Explicit `clearAll()` phase **between**
  `eventHandling()` and `terminationHandling()`. Effects re-check `kill_list`
  membership before execution.
- **VGDLx**: Sprites marked dead via `alive` mask — immediately visible to
  subsequent effects. No deferred clearing.

### 1.3 Effect Application Timing (MODERATE)

- **GVGAI** (`Game.java:1477-1529`): Three-phase: time effects (priority queue),
  edge-of-screen effects, then pairwise collisions. Supports **batch mode**
  (`inBatch` flag) for multi-partner effects.
- **VGDLx**: Same-type effects use `fori_loop` for sequential processing. Cross-type
  effects remain batched (mask-then-apply). Zelda: 1 divergent step in 20.

---

## 2. Physics

### 2.1 GridPhysics Speed Model (HIGH)

| Engine | Speed semantics | Position update | Coordinates |
|--------|----------------|-----------------|-------------|
| **GVGAI** | `speed` x `gridsize` = pixel displacement; independent `cooldown` field | `rect.translate(orientation * (int)(speed * gridsize.width))` | Integer pixels |
| **VGDLx** | `speed` = float multiplier (`delta * speed`) | `pos += orientation * speed` | Float32 grid cells |

For `speed=0.5`: GVGAI moves 5 pixels per tick (half cell, truncated);
VGDLx moves 0.5 cells per tick (fractional).

### 2.2 MarioAvatar.strength (MODERATE)

- **GVGAI**: Uses `speed` directly. No strength/jump_strength — acceleration = `action / mass`.
- **VGDLx**: `strength = 3` + `jump_strength = 10`. Config-fixable.

### 2.3 ContinuousPhysics Friction (MODERATE)

- **GVGAI**: Exponential velocity decay: `speed *= (1 - friction)` each tick.
- **VGDLx**: Friction removed entirely. Force-based model, no damping.

### 2.4 NoFrictionPhysics (LOW)

- **GVGAI**: No separate class — sprites set `friction = 0` directly.
- **VGDLx**: Class does not exist.

---

## 3. Effects

### 3.1 killIfSlow — All Engines Differ (HIGH)

| Engine | Speed calculation | What it measures |
|--------|-----------------|------------------|
| **GVGAI** | `magnitude(sprite1.orientation - sprite2.orientation)` | **Orientation vector divergence** (not speed) |
| **VGDLx** | `abs(speed_a - speed_b)` (scalar) | **Speed scalar difference** |

For static partner (speed=0): both reduce to checking actor's absolute speed.

### 3.2 transformTo State Transfer (HIGH)

| Engine | Copies orientation | Copies resources | Copies health | Copies avatar state |
|--------|-------------------|-----------------|---------------|-------------------|
| **VGDLx** | Yes | Yes | No | No |
| **GVGAI** | Yes (+ `forceOrientation`) | Yes | Yes | Yes (player ID, score, win, keyHandler) |

### 3.3 wallStop Friction/Velocity (MINOR)

| Engine | Friction behavior | Velocity after stop |
|--------|------------------|---------------------|
| **GVGAI** | Friction **commented out** | Recalculates `speed = mag * speed` from surviving component; gravity floor |
| **VGDLx** | Scales surviving axis by `(1 - friction)` | Speed unchanged |

No standard game uses `friction` on wallStop.

### 3.4 wallStop once_per_step Guard (MODERATE)

- **GVGAI**: Once-per-sprite-per-tick guard prevents double-application.
- **VGDLx**: No guard. May double-fire with multiple wall types.

### 3.5 wallStop Position Correction (MINOR)

- **GVGAI**: `calculatePixelPerfect()` + axis detection. Zeros collision-axis
  velocity and **preserves sliding magnitude** on perpendicular axis.
- **VGDLx**: `wall_pos +- 1.0` in grid-cell coordinates.

### 3.6 wallBounce Batch Handling (MODERATE)

- **GVGAI**: `executeBatch()` sorts partners by proximity, synthesizes unified
  collision boundary. Also applies upward force if sprite has gravity.
- **VGDLx**: Per-pair via `partner_idx`, center-to-center axis. No batch mode.

### 3.7 pullWithIt (MODERATE)

- **GVGAI**: Position delta + speed/orientation update. Forces Y position above
  partner and zeros X orientation for ContinuousPhysics. Supports `pixelPerfect`.
  `once_per_step` guard.
- **VGDLx**: Position delta only. No speed/orientation update, no once_per_step.

### 3.8 collectResource — Kill Configurability (LOW)

- **GVGAI**: Configurable `killResource` flag — can collect without destroying.
- **VGDLx**: Always kills resource sprite on success.

---

## 4. Collision Detection

| Engine | Method | Coordinates |
|--------|--------|-------------|
| **GVGAI** | `Rectangle.intersects()` AABB | Integer pixels |
| **VGDLx** | Grid occupancy (default) or AABB (`|diff| < 1.0 - 1e-3`) | Float32 grid cells |

VGDLx supports 7 modes: `grid`, `expanded_grid_a`/`_b`, `aabb`, `sweep`,
`static_b_grid`, `static_b_expanded`. Mode selected per compiled effect.

---

## 5. NPCs

### 5.1 Chaser Pathfinding (MODERATE)

- **GVGAI**: Greedy 1-step Manhattan, random tiebreak, no wall awareness.
- **VGDLx**: Distance field relaxation — routes around walls, deterministic tiebreak.

### 5.2 RandomNPC Consecutive Moves (MODERATE)

- **GVGAI**: `cons` parameter (default 0) repeats same direction for `cons` ticks.
- **VGDLx**: New random direction every move tick.

### 5.3 SpawnPoint Cooldown Model (MODERATE)

| Engine | Cooldown model |
|--------|---------------|
| **GVGAI** | Offset-based: `(start_tick + game_tick) % cooldown == 0` |
| **VGDLx** | Per-sprite cooldown timers decremented each tick |

### 5.4 Walker NPC (MODERATE)

- **GVGAI**: Gravity-based with `airsteering` parameter, `groundIntersects()`.
  Default `speed = 5`, `max_speed = 5`.
- **VGDLx**: `WALK_JUMPER` sprite class — horizontal walking + random jumps under gravity.

---

## 6. GVGAI-Only Features

Features present in GVGAI but absent from VGDLx:

### 6.1 Shield / Immunity System

`ShieldFrom` registers immunity: `game.addShield(sprite_type, shield_type,
effect_hash)`. Engine checks shields before executing collision effects.

### 6.2 Batch Effect Dispatch

Effects set `inBatch = true` and override `executeBatch()` to receive all
colliding partners at once. Used by `WallBounce` and `WallReverse`.

### 6.3 Time Effects (Priority Queue)

`TimeEffect` with `timer`, `repeating`, `nextExecution`. Managed via `TreeSet`
priority queue. Enables periodic events independent of collision.

### 6.4 Health Point System

`AddHealthPoints`, `AddHealthPointsToMax`, `SubtractHealthPoints`. Separate
from resources.

### 6.5 Per-Player Score Changes

`scoreChange` parsed per player ID. Enables asymmetric multiplayer scoring.

### 6.6 StopCounter Termination

Conditional gating — sets/clears `Termination.canEnd` flag that gates other
terminations. Checks up to 3 sprite types against a limit.

### 6.7 count_score Termination Logic

When `count_score = true`, winners determined by comparing avatar scores.

### 6.8 Effect `repeat` Field

`repeat` parameter (default 1) controls how many times an effect fires per
collision per step.

---

## 7. GVGAI-Only Effects

Effects in GVGAI not implemented in VGDLx (37 in VGDLx vs ~55 in GVGAI):

### 7.1 Unary Effects

| Effect | GVGAI behavior |
|--------|---------------|
| `AddHealthPoints` | Adds to healthPoints |
| `AddHealthPointsToMax` | Adds health, capped at max |
| `SubtractHealthPoints` | Subtracts health, kills at 0 |
| `HalfSpeed` | `speed *= 0.5` |
| `KillAll` | Kills all sprites of sprite1's type |
| `KillIfFast` | Kills if `speed > limspeed` |
| `KillIfNotUpright` | Kills if orientation != UP |
| `RemoveScore` | Deducts score from all avatars |
| `ShieldFrom` | Registers collision immunity |
| `SpawnAbove/Below/Left/Right/Behind` | Spawns at directional offset |
| `SpawnIfHasLess` | Spawn if resource < limit |
| `SpawnIfCounterSubTypes` | Spawn based on subtype counter |
| `TransformToRandomChild` | Transform to random subtype |
| `UpdateSpawnType` | Changes SpawnPoint's target type |
| `WaterPhysics` | Applies drag/physics modifier |

### 7.2 Binary Effects

| Effect | GVGAI behavior |
|--------|---------------|
| `AddTimer` | Attaches a timed effect |
| `Align` | Aligns sprite1 position to sprite2 |
| `CollectResourceIfHeld` | Collect only if holding specific resource |
| `DecreaseSpeedToAll` | Decreases speed of all sprites of type |
| `IncreaseSpeedToAll` | Increases speed of all sprites of type |
| `SetSpeedForAll` | Sets speed of all sprites of type |
| `KillIfFrontal` | Kill if collision from the front |
| `KillIfNotFrontal` | Kill if collision NOT from front |
| `TransformIfCount` | Transform if counter meets threshold |
| `TransformToAll` | Transform all type B sprites |
| `TransformToSingleton` | Transform ensuring only one instance |
| `WallReverse` | Wall bounce with orientation reversal + batch |

### 7.3 GVGAI Avatar Types Not in VGDLx

| Avatar | GVGAI behavior |
|--------|---------------|
| `NullAvatar` | No player control |
| `BirdAvatar` | Continuous angle movement (flappy-bird style) |
| `CarAvatar` | Forward/back + steering |
| `LanderAvatar` | Lunar lander thrust physics |
| `MissileAvatar` | Self-propelling projectile as avatar |
| `OngoingAvatar` | Persistent movement direction |
| `OngoingShootAvatar` | Ongoing + shooting |
| `OngoingTurningAvatar` | Ongoing + turning |
| `PlatformerAvatar` | Platformer-specific physics |
| `SpaceshipAvatar` | Rotation + thrust |
| `WizardAvatar` | Teleportation-based movement |

---

## 8. Cross-Engine Validation (2.0 <-> VGDLx)

100% feature parity with VGDL 2.0 (22 sprite classes, 14 avatar classes, 37
effects, 4 terminations, 3 physics types).

73/74 cross-engine tests pass. 8/9 games exact match. Zelda: 1/20 steps
diverged (step 20 monsterNormal position — cross-type effect batching residual).

### Feature Coverage (VGDLx vs GVGAI)

| Category | GVGAI | VGDLx | Coverage |
|----------|-------|----------|----------|
| Unary Effects | ~32 | ~16 | ~50% |
| Binary Effects | ~23 | ~13 | ~57% |
| Avatar Types | ~19 | 14 | ~74% |
| Terminations | 4+ (StopCounter) | 4 | ~80% |
| Physics | 3 | 3 | 100% |
| Shield System | Yes | No | 0% |
| Health System | Yes | No | 0% |
| Time Effects | Yes | No | 0% |
| Batch Effects | Yes | No | 0% |

---

## Appendix A: VGDL 1.0 / 2.0 Historical Differences

Differences between 1.0 and 2.0 where GVGAI and VGDLx agree (or GVGAI is unknown).
VGDLx inherits 2.0 behavior in all cases.

| # | Divergence | Engines | Notes |
|---|-----------|---------|-------|
| 1 | Termination timing | 1.0 ≠ {2.0, GVGAI, jax} | 1.0 checks at tick start; others at end |
| 6 | GravityPhysics.gravity default | 1.0 ≠ {2.0, jax}; GVGAI unknown | 1.0: `0.5`; 2.0/VGDLx: `1`. Config-fixable |
| 20 | partner_delta int32 truncation | {GVGAI, jax} truncate; {1.0, 2.0} float | GVGAI and VGDLx agree |

### Termination Timing

**1.0**: Terminations checked **at the start** of `tick()`, before sprite updates
and collision effects. For `Timeout(limit=N)`: 1.0 runs N-1 full steps; others run N.

**2.0 / GVGAI / VGDLx**: Checked **at the end**, after updates and effects.

### 1.0 Physics Defaults

- **GravityPhysics.gravity**: 1.0 uses `0.5`; 2.0/VGDLx use `1`. Config-fixable.
- **NoFrictionPhysics**: 1.0 has `class NoFrictionPhysics(ContinuousPhysics): friction = 0`.
  2.0/VGDLx removed it (redundant — ContinuousPhysics has no friction).

### 1.0 killIfSlow

| Engine | Speed calculation |
|--------|-----------------|
| **1.0** | `vectNorm(sprite.velocity - partner.velocity)` — relative velocity magnitude |
| **2.0** | Same formula, but **`relSpeed`/`relspeed` typo** → `NameError` for two-moving-sprites |

### 1.0 / 2.0 transformTo

- **1.0 / 2.0**: Copies orientation only (no resources, no health).

### 1.0 / 2.0 wallStop

- **1.0**: Scales non-collision axis by `(1 - friction)`.
- **2.0**: Friction accepted but never applied (dead code).

### 1.0 / 2.0 pullWithIt

- **1.0 / 2.0**: Position delta + `speed = partner.speed` + `orientation =
  partner.lastdirection` for ContinuousPhysics. `once_per_step` guard.

### 1.0 / 2.0 SpawnPoint

- **1.0 / 2.0**: Global step-count modulo: `step_count % cooldown == 0`.

---

## Appendix B: Engine Lineage

```
VGDL Spec (Tom Schaul, 2013)
+-- VGDL 1.0 (Python, original py-vgdl)
|   +-- VGDL 2.0 (Python, refactored fork)
|       +-- VGDLx (JAX, compiled port of 2.0)
+-- GVGAI (Java, independent implementation)
```
