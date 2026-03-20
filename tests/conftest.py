"""Shared test fixtures and constants for vgdl-jax tests."""
import os
import sys
from vgdl_jax.validate.constants import PYVGDL_GAMES_DIR, PYVGDL_GAMES
from vgdl_jax.env import VGDLxEnv

# Aliases used by many test files
GAMES_DIR = PYVGDL_GAMES_DIR
ALL_GAMES = sorted(PYVGDL_GAMES.keys())

# py-vgdl on sys.path for test_cross_engine.py (module-level import)
PYVGDL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl')
if PYVGDL_DIR not in sys.path:
    sys.path.insert(0, PYVGDL_DIR)


def make_env(game_name):
    """Create a VGDLxEnv for the given game name."""
    entry = PYVGDL_GAMES[game_name]
    return VGDLxEnv(entry.game_file, entry.level_files[0])
