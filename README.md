# VGDLx

*Pronounced "VG-deluxe"*

A JAX-compiled [VGDL](http://www.intfiction.org/forum/viewtopic.php?f=11&t=6497) (Video Game Description Language) engine for GPU-accelerated batched reinforcement learning.

VGDLx ports [py-vgdl](https://github.com/schaul/py-vgdl) to JAX, replacing OOP sprites with entity arrays and pygame collision with grid/AABB overlap detection. The result is a fully `jit`-able and `vmap`-able game step function that runs thousands of environments in parallel.

## Features

- **9 supported games**: Chase, Zelda, Aliens, MissileCommand, Sokoban, Portals, BoulderDash, SurviveZombies, Frogs
- **3 physics modes**: GridPhysics, ContinuousPhysics (InertialAvatar), GravityPhysics (MarioAvatar)
- **37 effects**: killSprite, stepBack, transformTo, wallStop, wallBounce, bounceDirection, teleportToExit, and more
- **gymnax-style API**: `reset(rng) -> (obs, state)`, `step(state, action) -> (obs, state, reward, done, info)`
- **No pygame dependency**: standalone parser and renderer

## Installation

```bash
uv pip install -e '.[dev]'
```

Requires Python 3.9+ and JAX.

## Quick Start

```python
import jax
from vgdl_jax.env import VGDLJaxEnv

env = VGDLJaxEnv("path/to/game.txt", "path/to/level.txt")
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng)

# Single step
obs, state, reward, done, info = env.step(state, action=0)

# Batched (vmap)
batch_reset = jax.vmap(env.reset)
batch_step = jax.vmap(env.step)
```

## Architecture

```
VGDL text file
    │
    ▼
  Parser (pure Python, runs once)
    │  GameDef dataclasses
    ▼
  Compiler (pure Python, runs once)
    │  Closures capturing jnp arrays
    ▼
  Step function (JAX, jit-compiled)
    state, action → new_state
```

Only the step function runs in JAX. The parser and compiler are standard Python executed once at setup.

**Key design choices:**
- Sprites stored as entity arrays `[n_types, max_n, ...]` with alive masks
- Collision: grid-based occupancy `O(max_n)` for grid physics, per-pair AABB for continuous physics
- Effects: boolean masks over sprite arrays (no Python loops at runtime)
- Speed as movement multiplier: `delta * speed` per tick, with sweep collision for speed > 1

## Testing

```bash
# All tests
python -m pytest tests/

# Skip cross-engine validation (under development)
python -m pytest tests/ --ignore=tests/test_validate.py

# Benchmarks
python benchmarks/throughput.py
```

241 tests covering parsing, compilation, collision, effects, terminations, per-game integration, and cross-engine validation.

## Throughput

Steps/second on CPU (Apple M1, 256 parallel environments):

| Game | FPS |
|------|-----|
| Sokoban | ~138K |
| Aliens | ~37K |
| Zelda | ~34K |
| Portals | ~31K |
| Frogs | ~27K |
| SurviveZombies | ~19K |
| MissileCommand | ~16K |
| Chase | ~15K |
| BoulderDash | ~15K |

## Documentation

- [`tests/MECHANICS_DIFF.md`](tests/MECHANICS_DIFF.md) — Three-way engine comparison (VGDL 1.0 · 2.0 · VGDLx): 18 divergences, validation results, feature coverage

## VGDL Game Format

Games are defined in two text files:

**Domain file** (`.txt`) — 4 sections:
- `SpriteSet`: sprite hierarchy with class assignments
- `InteractionSet`: collision pairs → effects
- `LevelMapping`: character → sprite type
- `TerminationSet`: win/loss conditions

**Level file** (`_lvl0.txt`) — ASCII grid using the level mapping characters.

Example games are in the companion [py-vgdl](https://github.com/schaul/py-vgdl) repository under `vgdl/games/`.

## License

MIT
