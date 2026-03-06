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
| 11 | turnAround mechanics | {1.0, 2.0, GVGAI} ≠ jax | HIGH | jax simplified |
| 12 | transformTo state transfer | GVGAI ≠ {1.0, 2.0} ≠ jax | HIGH | 3-way: 1.0/2.0 ori only; jax ori+resources; GVGAI ori+resources+health+player |
| 13 | wallStop friction | all four differ | MINOR | No game uses friction kwarg |
| 14 | wallStop once_per_step | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax missing guard |
| 15 | wallStop position correction | GVGAI ≠ {2.0, jax} | MINOR | GVGAI pixel-perfect + velocity preservation |
| 16 | wallBounce batch handling | GVGAI ≠ {1.0, 2.0, jax} | MODERATE | GVGAI has proximity sort |
| 17 | pullWithIt ContinuousPhysics | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax position-delta only |
| 18 | pullWithIt once_per_step | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax missing guard |
| 19 | teleportToExit cooldown reset | GVGAI ≠ {1.0, 2.0, jax} | LOW | GVGAI resets cooldown on teleport |
| 20 | partner_delta int32 truncation | {GVGAI, jax} truncate; {1.0, 2.0} float | MODERATE | GVGAI int pixels; jax explicit cast |
| 21 | wrapAround offset unsupported | {1.0, 2.0} ≠ jax | LOW | jax ignores offset |
| **Collision** | | | | |
| 22 | Collision detection method | all four differ | MINOR | See §5 |
| 23 | Continuous-physics threshold | GVGAI ≠ 2.0 ≠ jax | MINOR | GVGAI integer AABB |
| **NPCs** | | | | |
| 24 | Chaser pathfinding | {1.0, 2.0, GVGAI} ≠ jax | MODERATE | jax distance field is better |
| 25 | RandomNPC consecutive moves | GVGAI ≠ {1.0, 2.0, jax} | MODERATE | GVGAI has `cons` param |
| 26 | SpawnPoint cooldown model | GVGAI ≠ {1.0, 2.0} ≠ jax | MODERATE | 3 distinct models |
| **Terminations** | | | | |
| 27 | StopCounter gating | GVGAI only | MODERATE | Not in 1.0/2.0/jax |
| 28 | count_score on termination | GVGAI only | LOW | Not in 1.0/2.0/jax |
| **Architecture** | | | | |
| 29 | Shield/immunity system | GVGAI only | MODERATE | Not in 1.0/2.0/jax |
| 30 | Batch effect dispatch | GVGAI only | MODERATE | Not in 1.0/2.0/jax |
| 31 | Time effects (priority queue) | GVGAI only | MODERATE | Not in 1.0/2.0/jax |
| 32 | Health point system | GVGAI only | MODERATE | Not in 1.0/2.0/jax |
| 33 | Per-player scoreChange | GVGAI only | LOW | jax/2.0 single score |

---

## 1. Game Loop / Tick Ordering

### 1.1 Termination Timing (HIGH)

**1.0**: Terminations checked **at the start** of `tick()`, before sprite updates
and collision effects.

**2.0** and **VGDLx**: Terminations checked **at the end**, after updates and effects.

**GVGAI** (`Game.java:1092–1106`): `gameCycle()` runs:
1. `gameTick++`
2. `fwdModel.update()` (snapshot)
3. `tick()` (sprite movement)
4. `eventHandling()` (collisions + effects)
5. `clearAll()` (process kill_list)
6. `terminationHandling()` (check win/loss)
7. `checkTimeOut()`

GVGAI checks terminations **at the end**, matching 2.0/VGDLx (not 1.0).

For `Timeout(limit=N)`: 1.0 runs N-1 full steps; 2.0/GVGAI/jax run N.

### 1.2 Sprite Update Order (MODERATE)

**1.0 / 2.0**: Sprites updated in `spriteOrder` (z-order, avatar last).

**GVGAI** (`Game.java:1358–1384`): Two-phase update:
1. Avatars first (explicit loop over `no_players`)
2. Other sprites in **reverse z-order** (`spriteOrder[]` iterated from end to start)

**VGDLx**: All sprites updated per-type in compiler-defined order. Avatar movement
is a separate phase before NPC updates.

GVGAI's avatar-first update could cause subtle ordering differences when avatar
movement triggers effects that affect NPCs updated later in the same tick.

### 1.3 Kill List Clearing (MODERATE)

**1.0 / 2.0**: Dead sprites removed inline during collision handling or at tick end.

**GVGAI** (`Game.java:1667–1691`): Explicit `clearAll()` phase **between**
`eventHandling()` and `terminationHandling()`. Sprites added to `kill_list` during
effects remain in the game state until `clearAll()` runs. Effects re-check
`kill_list` membership before execution (`Game.java:1520`).

**VGDLx**: Sprites marked dead via `alive` mask — immediately visible to subsequent
effects within the same step. No deferred clearing.

### 1.4 Effect Application Timing (MODERATE) — All Four Differ

**1.0**: Effects applied immediately per collision, sequential iteration.

**2.0**: Same as 1.0 — sequential per collision pair within one effect type.

**GVGAI** (`Game.java:1477–1529`): Three-phase event handling:
1. Time effects (priority queue)
2. Edge-of-screen effects
3. Pairwise collisions — iterates `definedEffects` pairs, for each pair iterates
   all sprites. Supports **batch mode** (`inBatch` flag) where all colliding
   partners are passed to `executeBatch()` at once.

**VGDLx**: Same-type effects use `fori_loop` for sequential processing (matching
2.0). Cross-type effects remain batched (mask-then-apply). Only zelda shows a
residual: 1 divergent step in 20.

---

## 2. Physics

### 2.1 GridPhysics Speed Model (HIGH) — Three Distinct Models

| Engine | Speed semantics | Cooldown | Position update | Coordinates |
|--------|----------------|----------|-----------------|-------------|
| **1.0** | `speed` → cooldown: `1/speed` ticks between moves | Derived from speed | Move 1 cell per allowed tick | Integer (via pygame) |
| **2.0** | Same as 1.0 | Same as 1.0 | Move 1 cell per allowed tick | Integer (via pygame) |
| **GVGAI** | `speed` × `gridsize` = pixel displacement per move | Independent `cooldown` field (default varies). `lastmove` incremented each tick; moves when `lastmove >= cooldown`, then resets (`VGDLSprite.java:558–561`) | `rect.translate(orientation * (int)(speed * gridsize.width))` | Integer pixels |
| **VGDLx** | `speed` = movement multiplier (`delta * speed`) | Per-sprite cooldown timers (for NPCs via `_move_with_cooldown`) | `pos += orientation * speed` | Float32 grid cells |

**Key differences**:
- 1.0/2.0: Speed and cooldown are **coupled** (speed→cooldown). Always moves 1 cell.
- GVGAI: Speed and cooldown are **independent**. Speed controls distance (pixel-scaled), cooldown controls frequency. Position is integer pixels.
- VGDLx: Speed controls distance (float multiplier), cooldown for some NPC types. Position is fractional float32.

For `speed=1.0, cooldown=1`, all engines produce identical 1-cell-per-tick movement.

For fractional speeds (e.g., `speed=0.5`), behavior diverges:
- 1.0/2.0: Move 1 cell every 2 ticks (cooldown = 2)
- GVGAI: Move `(int)(0.5×10) = 5` pixels per allowed tick = half a cell (with default gridsize 10, truncated to int)
- VGDLx: Move 0.5 cells per tick (fractional position, no truncation)

### 2.2 GravityPhysics.gravity Default (MODERATE)

- **1.0**: `gravity = 0.5`
- **2.0** / **VGDLx**: `gravity = 1`
- **GVGAI**: Gravity is a sprite attribute, no global default in physics class.
  `ContinuousPhysics.java` applies `gravity * mass` as downward acceleration.

Config-fixable — game file can specify `gravity=X`.

### 2.3 MarioAvatar.strength Split (MODERATE)

- **1.0**: Single `strength = 10`. Horizontal ≈ 3.16, Jump = -10.
- **2.0** / **VGDLx**: `strength = 3` + `jump_strength = 10`.
- **GVGAI**: `ShootAvatar` and `MovingAvatar` use `speed` parameter directly.
  No `strength`/`jump_strength` split — physics acceleration uses `action / mass`.

Config-fixable where applicable.

### 2.4 ContinuousPhysics Friction (MODERATE)

- **1.0**: `friction = 0.02`, applied every tick: `speed *= (1 - friction)`.
- **2.0** / **VGDLx**: No `friction` attribute. Force-based model, no velocity damping.
- **GVGAI** (`ContinuousPhysics.java:42`): `sprite.speed *= (1 - sprite.friction)`.
  Exponential velocity decay each tick. Friction is a per-sprite attribute.

GVGAI matches 1.0's friction model. 2.0/VGDLx removed it entirely.

### 2.5 NoFrictionPhysics (LOW)

- **1.0**: `class NoFrictionPhysics(ContinuousPhysics): friction = 0`
- **GVGAI**: No separate class — sprites can set `friction = 0` directly.
- **2.0** / **VGDLx**: Class does not exist (redundant given §2.4 removal).

---

## 3. Effects — Behavioral Divergences

### 3.1 killIfSlow — All Four Engines Differ (HIGH)

| Engine | Speed calculation | What it measures |
|--------|-----------------|------------------|
| **1.0** | `vectNorm(sprite.velocity - partner.velocity)` | Relative velocity vector magnitude |
| **2.0** | Same formula, but **`relSpeed` vs `relspeed` typo** → crashes for two-moving-sprites | Broken for dynamic pairs |
| **GVGAI** | `magnitude(sprite1.orientation - sprite2.orientation)` | **Orientation vector divergence** (not speed!) |
| **VGDLx** | `abs(speed_a - speed_b)` (scalar) | **Speed scalar difference** |

Example — two sprites moving at speed 1, perpendicular directions:
- **1.0**: `|[1,0] - [0,1]| = √2` → may not kill (high relative velocity)
- **GVGAI**: `|[1,0] - [0,1]| = √2` → same as 1.0 (coincidence: both use direction vectors)
- **VGDLx**: `|1.0 - 1.0| = 0` → would kill (same speed, ignores direction)

For static partner (speed=0): all engines reduce to checking actor's absolute speed.
Divergence only matters with two moving sprites (no standard game exercises this).

### 3.2 turnAround (HIGH)

- **1.0 / 2.0**: Restores position, bypasses cooldown, calls `activeMovement(DOWN)`
  **twice**, reverses direction. Net: 2 cells down + reversed orientation.
- **GVGAI** (`TurnAround.java:33–39`): Restores to `lastrect`, sets
  `lastmove = cooldown` (immediate re-move), calls `activeMovement(DOWN)` twice,
  reverses direction, updates collision dict. Same as 1.0/2.0.
- **VGDLx**: Negates orientation only. No position restore, no displacement.

All three OOP engines agree; VGDLx is simplified.

### 3.3 transformTo State Transfer (HIGH)

| Engine | Copies orientation | Copies resources | Copies health | Copies avatar state |
|--------|-------------------|-----------------|---------------|-------------------|
| **1.0** | Yes | No | No | No |
| **2.0** | Yes | No | No | No |
| **GVGAI** | Yes (+ `forceOrientation` flag) | **Yes** (all resources) | **Yes** (healthPoints) | **Yes** (player ID, score, win, keyHandler) |
| **VGDLx** | Yes | Yes | No | No |

Three-way split. 1.0/2.0 copy only orientation. VGDLx copies orientation +
resources. GVGAI (`TransformTo.java:59–117`) copies orientation + resources +
health + avatar identity (player ID, score, win, keyHandler).

### 3.4 wallStop — All Four Differ in Friction/Velocity Handling (MINOR)

| Engine | Friction behavior | Velocity after stop |
|--------|------------------|---------------------|
| **1.0** | Scales non-collision axis by `(1 - friction)` | Orientation unchanged |
| **2.0** | Friction parameter accepted but never applied (dead code) | Orientation unchanged |
| **GVGAI** | Friction parameter exists but **commented out** (`WallStop.java:65,69`) | **Unique**: recalculates `speed = mag * speed` from surviving orientation component. Gravity floor: if speed < gravity, speed = gravity |
| **VGDLx** | Scales surviving velocity axis by `(1 - friction)` | Speed unchanged |

No standard game specifies `friction` on wallStop. GVGAI's velocity recalculation
(`WallStop.java:73–79`) is the most significant behavioral difference — it
adjusts the scalar speed based on the magnitude of the non-zeroed orientation
component. For example, if a sprite moving diagonally (orientation `(0.7, 0.7)`)
hits a vertical wall, the X component is zeroed, `mag = 0.7`, and
`speed *= 0.7`. The other engines leave speed unchanged.

### 3.5 wallStop once_per_step Guard (MODERATE)

- **1.0 / 2.0**: `once_per_step()` prevents double-application per sprite per tick.
- **GVGAI** (`WallStop.java:45–54`): Tracks `lastGameTime` + `spritesThisCycle`
  list — fires once per sprite per game tick.
- **VGDLx**: No equivalent guard. Effects fire per `(type_a, type_b)` pair.
  Correct for single wall type; may double-fire with multiple wall types.

### 3.6 wallStop Position Correction (MINOR)

- **1.0 / 2.0**: Pixel-precise `pygame.Rect.clip()`.
- **GVGAI** (`WallStop.java:57–79`): `calculatePixelPerfect()` + axis detection
  (horizontal vs vertical collision by center distance). Zeros collision-axis
  velocity and **preserves sliding magnitude** on the perpendicular axis.
- **VGDLx**: `wall_pos ± 1.0` in grid-cell coordinates. No magnitude preservation.

### 3.7 wallBounce Batch Handling (MODERATE)

- **1.0 / 2.0**: Per-pair sequential, `once_per_step` guard.
- **GVGAI** (`WallBounce.java:29,51–69`): `inBatch = true`. `executeBatch()`
  sorts colliding partners by proximity, synthesizes unified collision boundary,
  then bounces against it. Also applies upward force if sprite has gravity.
- **VGDLx**: Per-pair via `partner_idx`, center-to-center axis determination.
  No batch mode, no proximity sorting.

### 3.8 pullWithIt — ContinuousPhysics Handling (MODERATE)

- **1.0 / 2.0**: Applies partner's position delta + sets `speed = partner.speed`
  and `orientation = partner.lastdirection` for ContinuousPhysics. Uses
  `once_per_step` guard.
- **GVGAI** (`PullWithIt.java:63–91`): Same plus: if sprite uses ContinuousPhysics,
  forces Y position to `partner.rect.y - partner.rect.height` (above partner) and
  zeros X orientation. Supports `pixelPerfect` snapping.
- **VGDLx**: Position delta only. No speed/orientation update, no once_per_step,
  no ContinuousPhysics handling.

### 3.9 teleportToExit — Cooldown Reset (LOW)

| Engine | Copies position | Copies orientation | Resets cooldown |
|--------|----------------|-------------------|-----------------|
| **1.0 / 2.0** | Yes | Yes (if exit is oriented) | No |
| **GVGAI** | Yes | Yes (if exit is oriented) | Yes (`lastmove = 0`) |
| **VGDLx** | Yes | Yes | No |

All engines copy position and exit orientation. GVGAI additionally resets
cooldown (`lastmove = 0`) on teleport, allowing immediate movement after
arriving at the exit portal.

### 3.10 partner_delta int32 Truncation (MODERATE)

- **1.0**: Float position deltas.
- **2.0**: Float position deltas.
- **GVGAI**: Integer pixel deltas (positions are `Rectangle` with int coords).
- **VGDLx**: `(b_curr - b_prev).astype(jnp.int32)` truncates fractional deltas.

GVGAI and VGDLx both lose sub-integer precision (GVGAI by integer storage, VGDLx
by explicit truncation). 1.0/2.0 preserve float precision.

### 3.11 Chaser Tie-Breaking (MODERATE)

| Engine | Algorithm | Tiebreak | Sees walls? |
|--------|-----------|----------|-------------|
| **1.0** | Greedy 1-step Manhattan | Random | No |
| **2.0** | Greedy 1-step Manhattan | Random | No |
| **GVGAI** | Greedy 1-step Manhattan | Random | No |
| **VGDLx** | Distance field relaxation | Deterministic | **Yes** |

GVGAI matches 1.0/2.0 exactly. VGDLx accepted as design improvement.

### 3.12 collectResource — Kill Configurability (LOW)

- **1.0 / 2.0**: Always kills resource sprite on successful collection.
- **GVGAI** (`CollectResource.java`): Has configurable `killResource` flag —
  can collect without destroying the resource sprite.
- **VGDLx**: Always kills on success (matches 1.0/2.0).

### 3.13 wrapAround offset (LOW)

- **1.0 / 2.0**: `offset` parameter shifts wrap destination.
- **GVGAI** (`WrapAround.java`): No `offset` parameter — wraps to opposite edge.
- **VGDLx**: `offset` kwarg absorbed by `**_`, never used.

GVGAI and VGDLx both lack offset support. Only 1.0/2.0 implement it.

---

## 4. NPC Behavioral Differences

### 4.1 RandomNPC Consecutive Moves (MODERATE)

- **1.0 / 2.0**: Picks new random direction every move tick.
- **GVGAI** (`RandomNPC.java`): Has `cons` parameter (default 0). When `cons > 0`,
  repeats the same direction for `cons` ticks before picking a new random one.
- **VGDLx**: Picks new random direction every move tick (matches 1.0/2.0).

GVGAI is the only engine with this feature.

### 4.2 SpawnPoint Cooldown Model (MODERATE)

| Engine | Cooldown model |
|--------|---------------|
| **1.0** | Global step-count modulo: `step_count % cooldown == 0` |
| **2.0** | Same as 1.0 |
| **GVGAI** | Offset-based: `(start_tick + game_tick) % cooldown == 0`, where `start` is initialized on first update |
| **VGDLx** | Per-sprite cooldown timers: `cooldown_timers[type_idx]` decremented each tick |

Three distinct models. GVGAI's offset-based model means spawn timing depends on
when the spawner was created. VGDLx's per-sprite model is the most flexible.

### 4.3 Walker NPC (MODERATE)

- **1.0 / 2.0**: Walker with gravity, horizontal movement, direction changes.
- **GVGAI** (`Walker.java`): Gravity-based walker with `airsteering` parameter
  (can change direction mid-air). Default `speed = 5`, `max_speed = 5`.
  Uses `groundIntersects()` for ground detection.
- **VGDLx**: Has `WALK_JUMPER` sprite class with `update_walk_jumper()`. Horizontal
  walking + random jumps under gravity. Similar but not identical to GVGAI Walker.

### 4.4 Fleeing NPC

All four engines implement Fleeing as Chaser with inverted behavior. GVGAI
(`Fleeing.java`) extends `Chaser` with `fleeing = true`. VGDLx uses
`update_chaser(..., fleeing=True)`.

---

## 5. Collision Detection

| Engine | Method | Coordinates | Notes |
|--------|--------|-------------|-------|
| **1.0** | `pygame.Rect.colliderect()` | Integer pixels | AABB overlap |
| **2.0** | `pygame.Rect.collidelistall()` | Integer pixels | AABB overlap |
| **GVGAI** | `Rectangle.intersects()` | Integer pixels | Java AABB overlap |
| **VGDLx** | Grid occupancy (default) or AABB (`|diff| < 1.0 - 1e-3`) | Float32 grid cells | Per-pair mode selection |

VGDLx supports 7 collision modes: `grid`, `expanded_grid_a`/`_b`, `aabb`,
`sweep`, `static_b_grid`, `static_b_expanded`. Mode selected per compiled effect
based on sprite physics types. The other three engines use uniform AABB.

---

## 6. GVGAI-Only Features

Features present in GVGAI but absent from all Python engines (1.0, 2.0, VGDLx):

### 6.1 Shield / Immunity System

`ShieldFrom` effect registers immunity: `game.addShield(sprite_type, shield_type,
effect_hash)`. Game engine checks shields before executing collision effects.
Enables "sprite X becomes immune to effect Y after collecting shield Z".

### 6.2 Batch Effect Dispatch

Effects can set `inBatch = true` and override `executeBatch()` to receive all
colliding partners at once. `WallBounce` and `WallReverse` use this with
proximity-based sorting for consistent multi-wall collision resolution.

### 6.3 Time Effects (Priority Queue)

`TimeEffect` class with `timer`, `repeating`, `nextExecution` scheduling. Managed
via `TreeSet` priority queue in `eventHandling()`. Enables periodic timed events
independent of collision.

### 6.4 Health Point System

`AddHealthPoints`, `AddHealthPointsToMax`, `SubtractHealthPoints` effects.
`healthPoints` is a sprite attribute separate from resources. Not present in
any Python VGDL engine.

### 6.5 Per-Player Score Changes

`scoreChange` is a comma-separated string parsed per player ID
(`Effect.getScoreChange(playerID)`). Enables asymmetric scoring in multiplayer.
Python engines use a single `scoreChange` int.

### 6.6 StopCounter Termination

Conditional gating termination — does NOT end game, but sets/clears static
`Termination.canEnd` flag that gates other terminations. Checks up to 3 sprite
types against a limit.

### 6.7 count_score Termination Logic

When termination fires with `count_score = true`, winners are determined by
comparing avatar scores rather than the `win` parameter.

### 6.8 Effect `repeat` Field

Effects have `repeat` parameter (default 1) controlling how many times the effect
fires per collision per step.

---

## 7. GVGAI-Only Effects

Effects in GVGAI that are not implemented in VGDLx (37 effects in VGDLx vs ~55 in GVGAI):

### 7.1 Unary Effects Missing from VGDLx

| Effect | GVGAI behavior | Priority |
|--------|---------------|----------|
| `AddHealthPoints` | Adds to sprite's healthPoints | LOW (no health system) |
| `AddHealthPointsToMax` | Adds health, capped at maxHealthPoints | LOW |
| `SubtractHealthPoints` | Subtracts health, kills at 0 | LOW |
| `HalfSpeed` | `sprite.speed *= 0.5` | LOW |
| `KillAll` | Kills all sprites of sprite1's type | MODERATE |
| `KillIfFast` | Kills if `speed > limspeed` (inverse of killIfSlow) | LOW |
| `KillIfNotUpright` | Kills if orientation != UP | LOW |
| `RemoveScore` | Deducts score from all avatars | LOW |
| `ShieldFrom` | Registers collision immunity | MODERATE (needs shield system) |
| `SpawnAbove` | Spawns at position offset (0, -1) | LOW |
| `SpawnBelow` | Spawns at position offset (0, +1) | LOW |
| `SpawnLeft` | Spawns at position offset (-1, 0) | LOW |
| `SpawnRight` | Spawns at position offset (+1, 0) | LOW |
| `SpawnBehind` | Spawns at `-orientation` offset | LOW |
| `SpawnIfHasLess` | Spawn if resource < limit (inverse of SpawnIfHasMore) | LOW |
| `SpawnIfCounterSubTypes` | Spawn based on counter of subtypes | LOW |
| `TransformToRandomChild` | Transform to random subtype in hierarchy | LOW |
| `UpdateSpawnType` | Dynamically changes a SpawnPoint's target type | LOW |
| `WaterPhysics` | Applies drag/physics modifier | LOW |

### 7.2 Binary Effects Missing from VGDLx

| Effect | GVGAI behavior | Priority |
|--------|---------------|----------|
| `AddTimer` | Attaches a timed effect to sprite | LOW (needs time system) |
| `Align` | Aligns sprite1 position to sprite2 | LOW |
| `CollectResourceIfHeld` | Collect only if holding specific resource | LOW |
| `DecreaseSpeedToAll` | Decreases speed of all sprites of type | LOW |
| `IncreaseSpeedToAll` | Increases speed of all sprites of type | LOW |
| `SetSpeedForAll` | Sets speed of all sprites of type | LOW |
| `KillIfFrontal` | Kill if collision is from the front | LOW |
| `KillIfNotFrontal` | Kill if collision is NOT from front | LOW |
| `TransformIfCount` | Transform if counter meets threshold | LOW |
| `TransformToAll` | Transform all type B sprites | LOW |
| `TransformToSingleton` | Transform ensuring only one instance exists | LOW |
| `WallReverse` | Wall bounce with orientation reversal + batch | LOW |

### 7.3 GVGAI Avatar Types Not in VGDLx

| Avatar | GVGAI behavior |
|--------|---------------|
| `NullAvatar` | No player control |
| `AimedAvatar` | Angle-based aiming reticle |
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

VGDLx has 14 avatar classes in `SpriteClass` enum but several map to the same
update logic. The GVGAI avatars above would need new movement functions in
`sprites.py`.

---

## 8. VGDL 2.0 Bugs

### killIfSlow relSpeed/relspeed Typo (HIGH)

**2.0** (`effects.py:185–194`): The `else` branch writes `relSpeed` (capital S)
but line 193 reads `relspeed` (lowercase s) → `NameError` at runtime.

**1.0** is correct. **GVGAI** uses orientation vectors (different algorithm).
**VGDLx** sidesteps by using absolute speed (§3.1).

---

## 9. VGDL 2.0 Additions (not in 1.0)

**New effects**: `killBoth`, `SpendResource`, `SpendAvatarResource`, `KillOthers`,
`KillIfAvatarWithoutResource`, `AvatarCollectResource`, `TransformOthersTo`,
`NullEffect`.

**New avatar**: `ShootEverywhereAvatar` (fires all 4 directions).

**Structural changes**: Centralized `game.random_generator`; monolithic `ontology.py`
split into `ontology/` package.

All 2.0 additions are implemented in VGDLx.

---

## 10. Cross-Engine Validation Results (2.0 ↔ VGDLx)

73/74 cross-engine tests pass. Full suite: 136 passed, 1 failed (zelda step 20),
2 xfailed.

| Game | Status | Notes |
|------|--------|-------|
| Chase | **PASS** | All 20 NOOP steps match |
| Zelda | 1/20 diverged | Step 20 monsterNormal position (§3 residual) |
| Aliens | **PASS** | |
| MissileCommand | **PASS** | |
| Sokoban | **PASS** | Exact deterministic match |
| Portals | **PASS** | Fractional-speed RandomNPC matches |
| BoulderDash | **PASS** | Fractional-speed boulders match |
| SurviveZombies | **PASS** | |
| Frogs | **PASS** | Fractional-speed trucks/logs match |

No cross-engine validation exists for GVGAI ↔ VGDLx (different languages).

### Feature Coverage (VGDLx vs VGDL 2.0)

| Category | 2.0 | VGDLx | Coverage |
|----------|------|----------|----------|
| Sprite Classes | 22 | 22 | 100% |
| Avatar Classes | 14 | 14 | 100% |
| Effects | 37 | 37 | 100% |
| Terminations | 4 | 4 | 100% |
| Physics | 3 | 3 | 100% |

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

## Appendix A: Engine Lineage

```
VGDL Spec (Tom Schaul, 2013)
├── VGDL 1.0 (Python, original py-vgdl)
│   └── VGDL 2.0 (Python, refactored fork)
│       └── VGDLx (JAX, compiled port of 2.0)
└── GVGAI (Java, independent implementation)
```

GVGAI and VGDL 1.0 both implement the VGDL spec independently. GVGAI extends
the spec significantly (health, shields, batch effects, many more effect/avatar
types). VGDLx is faithful to VGDL 2.0 and does not implement GVGAI extensions.

## Appendix B: Quick Reference — Which Engines Agree?

| Aspect | Agreement Group |
|--------|----------------|
| Termination timing | {2.0, GVGAI, jax} (end of tick) vs 1.0 (start) |
| GridPhysics speed | {1.0, 2.0} (cooldown) vs GVGAI (pixel-scale) vs jax (float multiply) |
| ContinuousPhysics friction | {1.0, GVGAI} (has friction) vs {2.0, jax} (removed) |
| Chaser AI | {1.0, 2.0, GVGAI} (greedy random) vs jax (distance field) |
| killIfSlow | All four differ |
| turnAround | {1.0, 2.0, GVGAI} (displace+reverse) vs jax (flip only) |
| transformTo state | {1.0, 2.0} (ori only) vs jax (ori+resources) vs GVGAI (ori+resources+health+player) |
| once_per_step guards | {1.0, 2.0, GVGAI} (have guards) vs jax (missing) |
| Collision detection | {1.0, 2.0} (pygame AABB), GVGAI (java AABB), jax (grid/AABB/sweep) |
