"""Validation subpackage for cross-engine comparison between GVGAI and VGDLx."""
from .discovery import GameEntry, discover_games
from .constants import PYVGDL_GAMES, GVGAI_GAMES, PYVGDL_GAMES_DIR, GVGAI_GAMES_DIR, BLOCK_SIZE
from .harness import (setup_jax_game, setup_pyvgdl_game, run_comparison, run_gvgai_comparison,
                      run_jax_trajectory, run_pyvgdl_trajectory,
                      compare_states, StepComparison, TrajectoryResult,
                      get_sprite_configs, get_effects,
                      validate_pyvgdl_loads, validate_pyvgdl_state_extraction,
                      validate_pyvgdl_trajectory)
from .backend_gvgai import run_gvgai_trajectory, normalize_gvgai_state
from .state_extractor import extract_pyvgdl_state, extract_jax_state
from .rng_replay import RNGRecorder, ReplayRandomGenerator, patch_chaser_directions
