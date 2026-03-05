# Engine Comparison: VGDL 1.0 · VGDL 2.0 · VGDLx

Three-way comparison of behavioral differences across the VGDL engine lineage.

- **VGDL 1.0**: Tom Schaul's original py-vgdl (monolithic `ontology.py`)
- **VGDL 2.0**: The py-vgdl fork in this repo (`vgdl/ontology/` package split)
- **VGDLx**: JAX-compiled port, validated against VGDL 2.0

VGDLx was validated against VGDL 2.0, not 1.0, so it inherits all 2.0
divergences. VGDL 2.0 is not a simple port of 1.0 — it's a significant
refactor with behavioral changes. Some (wallStop rewrite, gravity) are
intentional improvements. Others (killIfSlow bug, termination timing) appear
accidental.

## Summary Table

| # | Divergence | Engines | Severity | Status | Config-fixable? |
|---|-----------|---------|----------|--------|-----------------|
| 1 | Termination timing | 1.0 ≠ 2.0 = jax | HIGH | Documented | No |
| 2 | GravityPhysics.gravity default | 1.0 ≠ 2.0 = jax | MODERATE | Documented | **Yes** |
| 3 | MarioAvatar.strength split | 1.0 ≠ 2.0 = jax | MODERATE | Documented | **Yes** |
| 4 | ContinuousPhysics friction removed | 1.0 ≠ 2.0 = jax | MODERATE | Documented | No |
| 5 | NoFrictionPhysics removed | 1.0 ≠ 2.0 = jax | LOW | Documented | No |
| 6 | killIfSlow absolute vs relative speed | all three differ | HIGH | Open | No |
| 7 | turnAround missing 2-cell displacement | 2.0 ≠ jax | HIGH | Open | No |
| 8 | killIfAlive no partner death check | 2.0 ≠ jax | MODERATE | Open | No |
| 9 | pullWithIt no ContinuousPhysics handling | 2.0 ≠ jax | MODERATE | Open | No |
| 10 | once_per_step absent for multi-type walls | 2.0 ≠ jax | MODERATE | Open | No |
| 11 | Effect application timing | 2.0 ≠ jax | MODERATE | **Partially fixed** | No |
| 12 | Chaser tie-breaking | 2.0 ≠ jax | MODERATE | **Accepted** | No |
| 13 | wallStop friction | all three differ | MINOR | **Accepted** | No |
| 14 | partner_delta int32 truncation | 2.0 ≠ jax | MODERATE | Open | No |
| 15 | wrapAround offset unsupported | 2.0 ≠ jax | LOW | Open | No |
| 16 | Collision detection method | 2.0 ≠ jax | MINOR | **Accepted** | No |
| 17 | wallStop position correction | 2.0 ≠ jax | MINOR | **Accepted** | No |
| 18 | Continuous-physics collision threshold | 2.0 ≠ jax | MINOR | **Accepted** | No |

2.0 bug (not a divergence): **killIfSlow relSpeed/relspeed typo** — crashes at
runtime for two-moving-sprites case (see §3).

Previously tracked divergences that have been **fixed**: SpawnPoint cooldown
semantics, avatar position clipping, speed→cooldown conversion.

---

## 1. VGDL 1.0 → 2.0 Divergences

These are inherited by VGDLx (which was validated against 2.0).

### 1.1 Termination Timing (HIGH)

**1.0** (`core.py:600-604`): Terminations checked **at the start** of `tick()`,
before sprite updates and collision effects.

**2.0** (`core.py:912-934`) and **VGDLx**: Terminations checked **at the end**,
after updates and effects.

For `Timeout(limit=N)`: 1.0 runs N-1 full steps; 2.0/jax run N. For
`SpriteCounter`: 1.0 detects kills next tick; 2.0 detects them same tick.
Off-by-one on every game. Not config-fixable.

### 1.2 GravityPhysics.gravity Default (MODERATE)

- **1.0** (`ontology.py:102`): `gravity = 0.5`
- **2.0** (`physics.py:93`): `gravity = 1`

All GravityPhysics games fall 2× as fast in 2.0/jax. **Config-fixable** — game file
can specify `gravity=0.5`.

### 1.3 MarioAvatar.strength Split (MODERATE)

- **1.0** (`ontology.py:660`): Single `strength = 10`. Horizontal = `sqrt(10)` ≈ 3.16. Jump = -10.
- **2.0** (`avatars.py:335-336`): Split into `strength = 3` + `jump_strength = 10`. Horizontal = 3.0. Jump = -10.

**Config-fixable** — game file can specify `strength=X jump_strength=Y`.

### 1.4 ContinuousPhysics Friction Removed (MODERATE)

- **1.0** (`ontology.py:67, 72-74`): `friction = 0.02`, applied every tick: `speed *= (1 - friction)`.
- **2.0**: No `friction` attribute. Force-based model, no velocity damping.

Not config-fixable — feature removed, not just default changed.

### 1.5 NoFrictionPhysics Removed (LOW)

- **1.0** (`ontology.py:98-99`): `class NoFrictionPhysics(ContinuousPhysics): friction = 0`
- **2.0**: class does not exist (redundant given §1.4).

Any game referencing `NoFrictionPhysics` fails to parse in 2.0/jax.

---

## 2. VGDL 2.0 ↔ JAX Divergences

### 2.1 killIfSlow — Absolute vs Relative Speed (HIGH)

- **1.0** (`ontology.py:834-844`): Relative speed via `vectNorm(sprite.velocity - partner.velocity)`. Correct for all cases.
- **2.0** (`effects.py:185-194`): Same logic but has a variable name bug (§3). Works for static-partner; crashes for two-moving-sprites.
- **VGDLx** (`effects.py:152-156`): Uses `state.speeds[type_a]` — actor's **absolute speed**, ignoring the partner.

All three differ. Mitigated by 2.0 being broken anyway (§3) and no standard
game exercising two-moving-sprites.

### 2.2 turnAround — Missing 2-Cell Displacement (HIGH)

- **1.0/2.0** (`effects.py:85-92`): Restores position, bypasses cooldown, calls `active_movement(DOWN)` **twice**, reverses direction. Net: 2 cells down + reversed orientation.
- **VGDLx** (`effects.py:241-243`): Negates orientation and restores position. No displacement.

No standard game uses `turnAround`.

### 2.3 killIfAlive — No Partner Death Check (MODERATE)

- **1.0/2.0** (`effects.py:202-205`): Only kills if `partner not in game.kill_list`.
- **VGDLx** (`effects.py:699`): Maps to `killSprite` — always kills regardless.

Matters when effect ordering kills the partner before `killIfAlive` fires. Overlaps
with §2.5 (effect timing).

### 2.4 pullWithIt — No ContinuousPhysics Handling (MODERATE)

- **1.0/2.0** (`effects.py:255-265`): Applies partner's position delta, **plus** sets `speed = partner.speed` and `orientation = partner.lastdirection` for ContinuousPhysics. Uses `oncePerStep` guard.
- **VGDLx** (`effects.py:456-472`): Position delta only. No speed/orientation update. No once-per-step guard.

No standard game uses pullWithIt with ContinuousPhysics.

### 2.5 Effect Application Timing (MODERATE) — Partially Fixed

**2.0**: Effects applied immediately per collision, sequential within one effect type.

**VGDLx**: Same-type effects use `fori_loop` for sequential processing (matching
2.0). Cross-type effects remain batched (mask-then-apply).

Only zelda shows a residual: 1 divergent step in 20.

### 2.6 once_per_step Guards Absent (MODERATE)

- **1.0/2.0**: `wallBounce`, `wallStop`, `pullWithIt` use `once_per_step()` to prevent double-application when colliding with multiple partners.
- **VGDLx**: No equivalent. Effects fire per `(type_a, type_b)` pair — correct for one wall type, but a sprite could be affected twice with multiple wall types sharing the same effect.

Standard 9 games have one wall type.

### 2.7 partner_delta int32 Truncation (MODERATE)

- **1.0/2.0**: `pullWithIt`/`bounceForward` apply float position deltas.
- **VGDLx** (`effects.py:462`): `(b_curr - b_prev).astype(jnp.int32)` truncates fractional deltas to zero for sub-integer speeds.

Standard games don't combine pullWithIt with fractional-speed partners.

### 2.8 Chaser Tie-Breaking (MODERATE) — Accepted

**2.0**: Greedy 1-step Manhattan, random tiebreak. Can't see around walls.

**VGDLx**: Global distance field via iterative relaxation, deterministic tiebreak.
Routes around walls. Accepted as a design improvement.

### 2.9 wallStop Friction (MINOR) — Accepted

All three engines differ:

1. **1.0**: Scales the non-collision orientation axis by `(1 - friction)`. Not dead code.
2. **2.0**: Friction parameter accepted but never applied. Dead code in 2.0 only.
3. **VGDLx**: Scales the surviving velocity axis by `(1 - friction)`. Third behavior.

No standard game specifies `friction` on wallStop.

### 2.10 wrapAround offset Unsupported (LOW)

- **1.0/2.0** (`effects.py:242`): `offset` parameter shifts wrap destination.
- **VGDLx** (`effects.py:260`): `offset` kwarg absorbed by `**_`, never used.

No standard game uses non-zero offset.

### 2.11 Collision Detection Method (MINOR) — Accepted

**2.0**: Pygame `rect.collidelistall()`.

**VGDLx**: Grid-based (default) or AABB (`|diff| < 1.0 - 1e-3`, for continuous/gravity pairs). Identical for grid-aligned sprites.

### 2.12 wallStop Position Correction (MINOR) — Accepted

**2.0**: Pixel-precise `pygame.Rect.clip()`.

**VGDLx**: `wall_pos ± 1.0` in grid-cell coordinates. Both place sprite at contact boundary.

### 2.13 Continuous-Physics Collision Threshold (MINOR) — Accepted

**2.0**: Pygame rect overlap (integer pixel math).

**VGDLx**: AABB with `1.0 - 1e-3` threshold. Equivalent at typical positions, may
differ at exact sub-pixel boundaries.

---

## 3. VGDL 2.0 Bugs

### killIfSlow relSpeed/relspeed Typo (HIGH)

**2.0** (`effects.py:185-194`): The `else` branch writes to `relSpeed` (capital S)
but line 193 reads `relspeed` (lowercase s) → `NameError` at runtime.

**1.0** is correct. **VGDLx** sidesteps by using absolute speed (§2.1).

---

## 4. VGDL 2.0 Additions (not in 1.0)

**New effects**: `killBoth`, `SpendResource`, `SpendAvatarResource`, `KillOthers`,
`KillIfAvatarWithoutResource`, `AvatarCollectResource`, `TransformOthersTo`,
`NullEffect`.

**New avatar**: `ShootEverywhereAvatar` (fires all 4 directions).

**Structural changes**: Centralized `game.random_generator` (1.0 used bare
`random`); monolithic `ontology.py` split into `ontology/` package.

All 2.0 additions are implemented in VGDLx.

---

## 5. Cross-Engine Validation Results

73/74 cross-engine tests pass (2.0 ↔ VGDLx, with RNG replay).
Full suite: 136 passed, 1 failed (zelda step 20), 2 xfailed.

| Game | Status | Notes |
|------|--------|-------|
| Chase | **PASS** | All 20 NOOP steps match |
| Zelda | 1/20 diverged | Step 20 monsterNormal position (§2.5 residual) |
| Aliens | **PASS** | |
| MissileCommand | **PASS** | |
| Sokoban | **PASS** | Exact deterministic match |
| Portals | **PASS** | Fractional-speed RandomNPC matches |
| BoulderDash | **PASS** | Fractional-speed boulders match |
| SurviveZombies | **PASS** | |
| Frogs | **PASS** | Fractional-speed trucks/logs match |

### Feature Coverage

| Category | 2.0 | VGDLx | Coverage |
|----------|------|----------|----------|
| Sprite Classes | 22 | 22 | 100% |
| Avatar Classes | 14 | 14 | 100% |
| Effects | 37 | 37 | 100% |
| Terminations | 4 | 4 | 100% |
| Physics | 3 | 3 | 100% |
