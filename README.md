# VGDLx

JAX-compiled [VGDL](https://en.wikipedia.org/wiki/Video_Game_Description_Language) game engine for GPU-accelerated RL. 67/68 supported GVGAI games match exactly over 40-step validation trajectories.

## Install

```bash
uv pip install -e '.[dev]'
```

## Play

```bash
python scripts/play.py zelda
python scripts/play.py aliens --scale 3
python scripts/play.py boulderdash --fps 20
```

Controls: Arrow/WASD = move, Space = use, N = noop, R = restart, Q = quit.

## Environment API

```python
from vgdl_jax.env import VGDLxEnv

# Create from bundled games
env = VGDLxEnv('vgdl_jax/games/gridphysics/zelda.txt',
               'vgdl_jax/games/gridphysics/zelda_lvl0.txt')

# Or from strings
env = VGDLxEnv.from_text(game_text, level_text)
```

### Properties

```python
env.n_actions    # total number of actions
env.noop_action  # index of the NOOP action
env.obs_shape    # observation shape
env.action_map   # {0: 'LEFT', 1: 'RIGHT', 2: 'DOWN', 3: 'UP', 4: 'USE', 5: 'NIL'}
```

Action ordering is consistent across all avatar types: directional actions first (LEFT, RIGHT, DOWN, UP), then USE if available, then NIL.

### Reset & Step

```python
import jax

obs, state = env.reset()                              # default RNG key
obs, state = env.reset(jax.random.PRNGKey(42))        # explicit key
obs, state, reward, done, info = env.step(state, action)
```

### Observation Formats

```python
# Binary channels (default) — one channel per sprite type
env = VGDLxEnv(game, level, obs_type='channels')  # [n_types, H, W] bool

# Grid — per-cell type ID list (like GVGAI's getObservationGrid)
env = VGDLxEnv(game, level, obs_type='grid')       # [H, W, max_occupancy] int32

# Entity list — for attention/transformer architectures
env = VGDLxEnv(game, level, obs_type='entities')   # [max_entities, 3] int32
```

### Batched Environments

VGDLx follows the standard JAX RL pattern — the env is always single, batching is done via `jax.vmap`:

```python
# Batch reset
keys = jax.random.split(jax.random.PRNGKey(0), 256)
obs_batch, state_batch = jax.vmap(env.reset)(keys)

# Batch step
step_fn = jax.jit(jax.vmap(env.step))
obs_batch, state_batch, rewards, dones, infos = step_fn(state_batch, actions)
```

### Rendering

```python
img = env.render(state)              # [H*bs, W*bs, 3] uint8 RGB
img = env.render(state, block_size=20)
```

## Bundled Games

118 GVGAI games are included in `vgdl_jax/games/gridphysics/`. 68 currently compile, 67 match GVGAI exactly.

```python
from vgdl_jax.validate.discovery import discover_games
games = discover_games('vgdl_jax/games/gridphysics/', source='gvgai')
print(f'{len(games)} games available')
```

## Validation

```bash
# Run cross-engine validation against GVGAI
python scripts/validate_all.py --source gvgai --compat-filter supported

# Debug a specific game
python scripts/debug_divergence.py zelda --context 3 --all-types
```

## Architecture

```
game.txt + level.txt
    → parse_vgdl()     [parser.py]    → GameDef
    → compile_game()   [compiler.py]  → CompiledGame (init_state + jitted step_fn)
    → VGDLxEnv         [env.py]       → reset/step/render
```

Key design choices:
- **Int32 pixel coordinates** — no float drift, matches GVGAI exactly
- **Static grid optimization** — walls as `[H,W]` bool grids, not position arrays
- **O(max_n) collision** — occupancy grids, not pairwise
- **Python `if` for trace-time dispatch** — dead code elimination from JIT graph
- **Fully vmap-able** — batched RL at 50K–500K steps/sec on CPU
