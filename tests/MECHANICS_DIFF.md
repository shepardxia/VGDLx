# Open Divergences: GVGAI vs VGDLx

## Game Loop

**Sprite update order** — GVGAI: avatars first, then reverse z-order. VGDLx: compiler-defined order, avatar phase separate.

**Kill list clearing** — GVGAI: explicit `clearAll()` between effects and terminations. VGDLx: `alive` mask, immediately visible.

**Effect timing** — GVGAI: 3-phase (time effects → EOS → collisions), supports batch mode. VGDLx: `fori_loop` for same-type, batch for cross-type. Zelda: 1/20 steps diverged.

## Physics

**GridPhysics speed** (HIGH) — GVGAI: `speed × gridsize` = pixel displacement, integer coords, independent `cooldown`. VGDLx: `speed` = float multiplier, `pos += ori * speed`. For `speed=0.5`: GVGAI moves 5px/tick (truncated); VGDLx moves 0.5 cells/tick.

**MarioAvatar** — GVGAI: `action / mass`, no strength params. VGDLx: `strength=3` + `jump_strength=10`. Config-fixable.

**ContinuousPhysics friction** — GVGAI: `speed *= (1 - friction)` each tick. VGDLx: no friction.

## Effects

**killIfSlow** (HIGH) — GVGAI: `magnitude(ori1 - ori2)` (orientation divergence, not speed). VGDLx: `abs(speed_a - speed_b)` (scalar). Both agree for static partners.

**transformTo** — GVGAI copies orientation (conditional on `is_oriented` + `forceOrientation`) + resources + health + player state. VGDLx copies orientation (conditional, matching GVGAI) + resources. Missing: health, player state (GVGAI-only features).

**wallStop** — GVGAI: pixel-perfect positioning, velocity recalculation, friction commented out. VGDLx: `wall_pos ± 1.0`, applies `(1-friction)` scaling. No game uses friction.

**wallBounce** — GVGAI: `executeBatch()` sorts by proximity, unified boundary, gravity upforce. VGDLx: per-pair center-to-center axis.

**pullWithIt** — GVGAI: position delta + speed/orientation copy + ContinuousPhysics Y-force. VGDLx: position delta only.


## Collision

GVGAI: `Rectangle.intersects()`, integer pixels. VGDLx: grid occupancy or AABB (`|diff| < 1.0 - 1e-3`), float32 cells. VGDLx has 7 modes selected per compiled effect.

## NPCs

**Chaser** — GVGAI: greedy 1-step Manhattan, random tiebreak. VGDLx: distance field relaxation, routes around walls (accepted as better).

**SpawnPoint** — GVGAI: `(start_tick + game_tick) % cooldown`. VGDLx: per-sprite `spawn_timers` (independent of `cooldown_timers`), init to `cd` so first `_pre_spawn` increments to `cd+1 >= cd` firing at tick 0.

**Walker** — GVGAI: gravity + `airsteering`, `speed=5`. VGDLx: horizontal walk + random jumps.

## GVGAI-Only Features

Not implemented in VGDLx:

- **Shield/immunity** — `ShieldFrom` registers per-type collision immunity
- **Batch effects** — `executeBatch()` receives all partners at once (WallBounce, WallReverse)
- **Time effects** — priority queue for periodic events independent of collision
- **Health system** — `AddHealthPoints`, `SubtractHealthPoints`, separate from resources
- **Per-player scoring** — `scoreChange` parsed per player ID
- **StopCounter** — conditional gating via `canEnd` flag on other terminations
- **count_score** — winners by comparing avatar scores
- ~~**Effect repeat**~~ — `repeat` now supported (CompiledEffect duplicated N times)

## GVGAI-Only Effects & Avatars

**Unary**: AddHealthPoints, AddHealthPointsToMax, SubtractHealthPoints, HalfSpeed, KillAll, KillIfFast, KillIfNotUpright, RemoveScore, ShieldFrom, SpawnAbove/Below/Left/Right/Behind, SpawnIfHasLess, SpawnIfCounterSubTypes, TransformToRandomChild, UpdateSpawnType, WaterPhysics

**Binary**: AddTimer, Align, CollectResourceIfHeld, DecreaseSpeedToAll, IncreaseSpeedToAll, SetSpeedForAll, KillIfFrontal, KillIfNotFrontal, TransformIfCount, TransformToAll, TransformToSingleton, WallReverse

**Avatars**: NullAvatar, BirdAvatar, CarAvatar, LanderAvatar, MissileAvatar, OngoingAvatar, OngoingShootAvatar, OngoingTurningAvatar, PlatformerAvatar, SpaceshipAvatar, WizardAvatar

## GVGAI Cross-Engine Validation (42 supported games, 40 steps)

**29 exact match**: aliens, avoidgeorge, bait, brainman, chainreaction, chase, chipschallenge, clusters, cookmepasta, eggomania, factorymanager, flower, islands, jaws, missilecommand, modality, portals, realsokoban, rivers, shipwreck, sokoban, superman, surround, tercio, thecitadel, thesnowman, waitforbreakfast, waves, wrapsokoban

**3 RNG consumption order**: butterflies, chopper, infection — reverse NPC loop produces different RNG sequences than GVGAI's per-sprite iteration within type lists. Known limitation.

**3 spawn/RNG timing**: defender, ikaruga, whackamole — stochastic spawn timing offset produces different RNG roll outcomes.

**2 spawn position (prev_positions)**: angelsdemons, wildgunman — spawned sprites still teleport to (0,0) after step_back. prev_positions fix applied but may need further investigation.

**1 Spreader not fully implemented**: glow — noop matches, random fails. Spark (Spreader) dies before spreading due to remaining Flicker interaction.

**1 EOS/turnAround**: myAliens — EOS rect fix applied but turnAround implementation may diverge from GVGAI's `activeMovement(DDOWN)` chain.

**1 spawnorientation/RNG**: sheriff — spawnorientation fix applied but remaining divergence.

**1 block_size/misc**: assemblyline — possible block_size mismatch or prev_positions issue.

**1 spawn timing**: plaqueattack — spawn timing + prev_positions interaction.

## Coverage (VGDLx / GVGAI)

Unary effects ~50%, binary effects ~57%, avatar types ~74%, terminations ~80%, physics 100%. Shield, health, time effects, batch dispatch: 0%.

---

## Appendix: 1.0 / 2.0 Historical

Where GVGAI and VGDLx agree but 1.0/2.0 differ. VGDLx inherits 2.0.

- **Termination timing** — 1.0 checks at tick start (N-1 steps for Timeout=N); 2.0/GVGAI/VGDLx check at end
- **GravityPhysics.gravity** — 1.0: `0.5`; 2.0/VGDLx: `1`; GVGAI: per-sprite. Config-fixable
- **partner_delta** — GVGAI truncates to int32; 1.0/2.0 keep float. VGDLx keeps float, no longer clips to grid bounds (allows OOB for wrapAround)
- **killIfSlow** — 1.0: `vectNorm(vel1 - vel2)`; 2.0: same but `relSpeed`/`relspeed` typo → NameError
- **transformTo** — 1.0/2.0 copy orientation only (no resources)
- **wallStop** — 1.0: applies friction; 2.0: friction dead code
- **pullWithIt** — 1.0/2.0: full speed/orientation copy + `once_per_step`
- **SpawnPoint** — 1.0/2.0: global `step_count % cooldown`
