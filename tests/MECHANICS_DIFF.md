# Engine Comparison: VGDL 1.0 · VGDL 2.0 · GVGAI · VGDLx

Four-way comparison of behavioral differences across the VGDL engine family.

- **VGDL 1.0**: Tom Schaul's original py-vgdl (monolithic `ontology.py`)
- **VGDL 2.0**: The py-vgdl fork in this repo (`vgdl/ontology/` package split)
- **GVGAI**: General Video Game AI framework (Java, `GVGAI/src/`)
- **VGDLx**: JAX-compiled port, validated against VGDL 2.0

VGDLx was validated against VGDL 2.0, not 1.0 or GVGAI, so it inherits all 2.0
divergences. GVGAI is an independent Java implementation that shares the VGDL
specification but has its own behavioral choices.

---

## Summary Table

| # | Divergence | Engines | Severity | Notes |
|---|-----------|---------|----------|-------|
| **Game Loop** | | | | |
| 1 | Termination timing | 1.0 ≠ {2.0, GVGAI, jax} | HIGH | 1.0 checks at tick start |
| 2 | Sprite update order | GVGAI ≠ {1.0, 2.0, jax} | MODERATE | GVGAI reverse z-order |
| 3 | Kill list clearing | GVGAI ≠ {1.0, 2.0, jax} | MODERATE | GVGAI explicit clearAll phase |
| 4 | Effect application timing | all four differ | MODERATE | See §1.4 |
| **Physics** | | | | |
| 5 | GridPhysics speed model | GVGAI ≠ {1.0, 2.0} ≠ jax | HIGH | 3 distinct models |
| 6 | GravityPhysics.gravity default | 1.0 ≠ {2.0, jax}; GVGAI unknown | MODERATE | Config-fixable |
| 7 | MarioAvatar.strength split | 1.0 ≠ {2.0, jax}; GVGAI differs | MODERATE | Config-fixable |
| 8 | ContinuousPhysics friction | {1.0, GVGAI} ≠ {2.0, jax} | MODERATE | 2.0/jax removed friction |
| 9 | NoFrictionPhysics removed | {1.0, GVGAI} ≠ {2.0, jax} | LOW | Class absent in 2.0/jax |
| **Effects** | | | | |
| 10 | killIfSlow speed calc | all four differ | HIGH | Critical algorithm divergence |
| 11 | turnAround mechanics | — | ~~HIGH~~ FIXED | jax now matches 1.0/2.0 (2-cell-down displacement) |
| 12 | transformTo state transfer | GVGAI ≠ {1.0, 2.0} ≠ jax | HIGH | 3-way: 1.0/2.0 ori only; jax ori+resources; GVGAI ori+resources+health+player |
| 13 | wallStop friction | all four differ | MINOR | No game uses friction kwarg |
| 14 | wallStop once_per_step | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax missing guard |
| 15 | wallStop position correction | GVGAI ≠ {2.0, jax} | MINOR | GVGAI pixel-perfect + velocity preservation |
| 16 | wallBounce batch handling | GVGAI ≠ {1.0, 2.0, jax} | MODERATE | GVGAI has proximity sort |
| 17 | pullWithIt ContinuousPhysics | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax position-delta only |
| 18 | pullWithIt once_per_step | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax missing guard |
| 19 | teleportToExit cooldown reset | GVGAI ≠ {1.0, 2.0} | ~~LOW~~ FIXED | jax now resets cooldown (matches GVGAI) |
| 20 | partner_delta int32 truncation | {GVGAI, jax} truncate; {1.0, 2.0} float | MODERATE | GVGAI int pixels; jax explicit cast |
| 21 | wrapAround offset | {1.0, 2.0, jax} ≠ GVGAI | ~~LOW~~ FIXED | jax now supports offset (matches 1.0/2.0) |
| **Collision** | | | | |
| 22 | Collision detection method | all four differ | MINOR | See §5 |
| 23 | Continuous-physics threshold | GVGAI ≠ 2.0 ≠ jax | MINOR | GVGAI integer AABB |
| **NPCs** | | | | |
| 24 | Chaser pathfinding | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax distance field is better |
| 25 | RandomNPC consecutive moves | GVGAI ≠ {1.0, 2.0, jax} | MODERATE | GVGAI has `cons` param |
| 26 | SpawnPoint cooldown model | GVGAI ≠ {1.0, 2.0} ≠ jax | MODERATE | 3 distinct models |
| **GVGAI-Only** | | | | |
| 27 | StopCounter gating | GVGAI only | MODERATE | |
| 28 | count_score on termination | GVGAI only | LOW | |
| 29 | Shield/immunity system | GVGAI only | MODERATE | |
| 30 | Batch effect dispatch | GVGAI only | MODERATE | |
| 31 | Time effects (priority queue) | GVGAI only | MODERATE | |
| 32 | Health point system | GVGAI only | MODERATE | |
| 33 | Per-player scoreChange | GVGAI only | LOW | |

---

## 1. Game Loop / Tick Ordering

### 1.1 Termination Timing (HIGH)

**1.0**: Terminations checked **at the start** of `tick()`, before sprite updates
and collision effects.

**2.0 / GVGAI / VGDLx**: Checked **at the end**, after updates and effects.

For `Timeout(limit=N)`: 1.0 runs N-1 full steps; others run N.

### 1.2 Sprite Update Order (MODERATE)

**1.0 / 2.0**: Sprites updated in `spriteOrder` (z-order, avatar last).

**GVGAI** (`Game.java:1358–1384`): Two-phase — avatars first, then other sprites
in **reverse z-order**.

**VGDLx**: Per-type in compiler-defined order. Avatar movement is a separate
phase before NPC updates.

### 1.3 Kill List Clearing (MODERATE)

**1.0 / 2.0**: Dead sprites removed inline during collision handling or at tick end.

**GVGAI** (`Game.java:1667–1691`): Explicit `clearAll()` phase **between**
`eventHandling()` and `terminationHandling()`. Effects re-check `kill_list`
membership before execution.

**VGDLx**: Sprites marked dead via `alive` mask — immediately visible to
subsequent effects. No deferred clearing.

### 1.4 Effect Application Timing (MODERATE)

**1.0 / 2.0**: Sequential per collision pair within one effect type.

**GVGAI** (`Game.java:1477–1529`): Three-phase: time effects (priority queue),
edge-of-screen effects, then pairwise collisions. Supports **batch mode**
(`inBatch` flag) for multi-partner effects.

**VGDLx**: Same-type effects use `fori_loop` for sequential processing. Cross-type
effects remain batched (mask-then-apply). Zelda: 1 divergent step in 20.

---

## 2. Physics

### 2.1 GridPhysics Speed Model (HIGH) — Three Distinct Models

| Engine | Speed semantics | Position update | Coordinates |
|--------|----------------|-----------------|-------------|
| **1.0 / 2.0** | `speed` → cooldown: `1/speed` ticks between moves | 1 cell per allowed tick | Integer (pygame) |
| **GVGAI** | `speed` × `gridsize` = pixel displacement; independent `cooldown` field | `rect.translate(orientation * (int)(speed * gridsize.width))` | Integer pixels |
| **VGDLx** | `speed` = float multiplier (`delta * speed`) | `pos += orientation * speed` | Float32 grid cells |

For `speed=0.5`: 1.0/2.0 move 1 cell every 2 ticks; GVGAI moves 5 pixels per
tick (half cell, truncated); VGDLx moves 0.5 cells per tick (fractional).

### 2.2 GravityPhysics.gravity Default (MODERATE)

- **1.0**: `gravity = 0.5`
- **2.0 / VGDLx**: `gravity = 1`
- **GVGAI**: Per-sprite attribute, no global default.

Config-fixable.

### 2.3 MarioAvatar.strength Split (MODERATE)

- **1.0**: Single `strength = 10`. Horizontal ≈ 3.16, Jump = -10.
- **2.0 / VGDLx**: `strength = 3` + `jump_strength = 10`.
- **GVGAI**: Uses `speed` directly. No strength/jump_strength — acceleration = `action / mass`.

Config-fixable.

### 2.4 ContinuousPhysics Friction (MODERATE)

- **1.0 / GVGAI**: Exponential velocity decay: `speed *= (1 - friction)` each tick.
- **2.0 / VGDLx**: Friction removed entirely. Force-based model, no damping.

### 2.5 NoFrictionPhysics (LOW)

- **1.0**: `class NoFrictionPhysics(ContinuousPhysics): friction = 0`
- **GVGAI**: No separate class — sprites set `friction = 0` directly.
- **2.0 / VGDLx**: Class does not exist (redundant given §2.4).

---

## 3. Effects — Behavioral Divergences

### 3.1 killIfSlow — All Four Differ (HIGH)

| Engine | Speed calculation | What it measures |
|--------|-----------------|------------------|
| **1.0** | `vectNorm(sprite.velocity - partner.velocity)` | Relative velocity magnitude |
| **2.0** | Same formula, but **`relSpeed`/`relspeed` typo** → `NameError` for two-moving-sprites | Broken for dynamic pairs |
| **GVGAI** | `magnitude(sprite1.orientation - sprite2.orientation)` | **Orientation vector divergence** (not speed) |
| **VGDLx** | `abs(speed_a - speed_b)` (scalar) | **Speed scalar difference** |

For static partner (speed=0): all reduce to checking actor's absolute speed.

### 3.2 turnAround (FIXED)

- **1.0 / 2.0 / GVGAI / VGDLx**: Restores position, displaces 2 cells down,
  reverses direction. VGDLx now matches (clamped to grid bounds).

### 3.3 transformTo State Transfer (HIGH)

| Engine | Copies orientation | Copies resources | Copies health | Copies avatar state |
|--------|-------------------|-----------------|---------------|-------------------|
| **1.0 / 2.0** | Yes | No | No | No |
| **VGDLx** | Yes | Yes | No | No |
| **GVGAI** | Yes (+ `forceOrientation`) | Yes | Yes | Yes (player ID, score, win, keyHandler) |

### 3.4 wallStop Friction/Velocity — All Four Differ (MINOR)

| Engine | Friction behavior | Velocity after stop |
|--------|------------------|---------------------|
| **1.0** | Scales non-collision axis by `(1 - friction)` | Orientation unchanged |
| **2.0** | Friction accepted but never applied (dead code) | Orientation unchanged |
| **GVGAI** | Friction **commented out** | Recalculates `speed = mag * speed` from surviving component; gravity floor |
| **VGDLx** | Scales surviving axis by `(1 - friction)` | Speed unchanged |

No standard game uses `friction` on wallStop.

### 3.5 wallStop once_per_step Guard (MODERATE)

- **1.0 / 2.0 / GVGAI**: Once-per-sprite-per-tick guard prevents double-application.
- **VGDLx**: No guard. May double-fire with multiple wall types.

### 3.6 wallStop Position Correction (MINOR)

- **1.0 / 2.0**: Pixel-precise `pygame.Rect.clip()`.
- **GVGAI**: `calculatePixelPerfect()` + axis detection. Zeros collision-axis
  velocity and **preserves sliding magnitude** on perpendicular axis.
- **VGDLx**: `wall_pos ± 1.0` in grid-cell coordinates.

### 3.7 wallBounce Batch Handling (MODERATE)

- **1.0 / 2.0**: Per-pair sequential, `once_per_step` guard.
- **GVGAI**: `executeBatch()` sorts partners by proximity, synthesizes unified
  collision boundary. Also applies upward force if sprite has gravity.
- **VGDLx**: Per-pair via `partner_idx`, center-to-center axis. No batch mode.

### 3.8 pullWithIt (MODERATE)

- **1.0 / 2.0**: Position delta + `speed = partner.speed` + `orientation =
  partner.lastdirection` for ContinuousPhysics. `once_per_step` guard.
- **GVGAI**: Same plus: forces Y position above partner and zeros X orientation
  for ContinuousPhysics. Supports `pixelPerfect`.
- **VGDLx**: Position delta only. No speed/orientation update, no once_per_step.

### 3.9 teleportToExit — Cooldown Reset (FIXED)

All engines copy position and exit orientation. GVGAI and VGDLx reset
cooldown on teleport, allowing immediate movement after arrival.
1.0/2.0 do not reset cooldown.

### 3.10 partner_delta int32 Truncation (MODERATE)

- **1.0 / 2.0**: Float position deltas (sub-integer precision preserved).
- **GVGAI / VGDLx**: Integer truncation (GVGAI by int storage, VGDLx by explicit cast).

### 3.11 Chaser Pathfinding (MODERATE)

- **1.0 / 2.0 / GVGAI**: Greedy 1-step Manhattan, random tiebreak, no wall awareness.
- **VGDLx**: Distance field relaxation — routes around walls, deterministic tiebreak.

### 3.12 collectResource — Kill Configurability (LOW)

- **GVGAI**: Configurable `killResource` flag — can collect without destroying.
- **Others**: Always kills resource sprite on success.

### 3.13 wrapAround offset (FIXED)

- **1.0 / 2.0 / VGDLx**: `offset` parameter shifts wrap destination.
- **GVGAI**: No offset support.

---

## 4. NPC Behavioral Differences

### 4.1 RandomNPC Consecutive Moves (MODERATE)

- **GVGAI**: `cons` parameter (default 0) repeats same direction for `cons` ticks.
- **Others**: New random direction every move tick.

### 4.2 SpawnPoint Cooldown Model (MODERATE)

| Engine | Cooldown model |
|--------|---------------|
| **1.0 / 2.0** | Global step-count modulo: `step_count % cooldown == 0` |
| **GVGAI** | Offset-based: `(start_tick + game_tick) % cooldown == 0` |
| **VGDLx** | Per-sprite cooldown timers decremented each tick |

### 4.3 Walker NPC (MODERATE)

- **1.0 / 2.0**: Walker with gravity, horizontal movement, direction changes.
- **GVGAI**: Gravity-based with `airsteering` parameter, `groundIntersects()`.
  Default `speed = 5`, `max_speed = 5`.
- **VGDLx**: `WALK_JUMPER` sprite class — horizontal walking + random jumps under gravity.

---

## 5. Collision Detection

| Engine | Method | Coordinates |
|--------|--------|-------------|
| **1.0 / 2.0** | `pygame.Rect` AABB overlap | Integer pixels |
| **GVGAI** | `Rectangle.intersects()` AABB | Integer pixels |
| **VGDLx** | Grid occupancy (default) or AABB (`\|diff\| < 1.0 - 1e-3`) | Float32 grid cells |

VGDLx supports 7 modes: `grid`, `expanded_grid_a`/`_b`, `aabb`, `sweep`,
`static_b_grid`, `static_b_expanded`. Mode selected per compiled effect.

---

## 6. GVGAI-Only Features

Features present in GVGAI but absent from all Python engines:

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

## 8. Cross-Engine Validation (2.0 ↔ VGDLx)

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

## Appendix: Engine Lineage

```
VGDL Spec (Tom Schaul, 2013)
├── VGDL 1.0 (Python, original py-vgdl)
│   └── VGDL 2.0 (Python, refactored fork)
│       └── VGDLx (JAX, compiled port of 2.0)
└── GVGAI (Java, independent implementation)
```
