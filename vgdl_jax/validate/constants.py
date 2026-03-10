"""Single source of truth for validation constants — discovery-based, no hardcoded game lists."""
import os

from .discovery import discover_games

BLOCK_SIZE = 10

_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')

PYVGDL_GAMES_DIR = os.path.normpath(os.path.join(_BASE_DIR, 'py-vgdl', 'vgdl', 'games'))
GVGAI_GAMES_DIR = os.path.normpath(os.path.join(_BASE_DIR, 'GVGAI', 'examples', 'gridphysics'))

# py-vgdl games are always needed (tests depend on them) — discover eagerly
PYVGDL_GAMES = {g.name: g for g in discover_games(PYVGDL_GAMES_DIR, source='pyvgdl')}

# GVGAI games are only needed for --source gvgai workflows — discover lazily
_gvgai_games = None


def get_gvgai_games():
    """Lazily discover GVGAI games on first access."""
    global _gvgai_games
    if _gvgai_games is None:
        _gvgai_games = {g.name: g for g in discover_games(GVGAI_GAMES_DIR, source='gvgai')}
    return _gvgai_games


# For backward compat with code that reads GVGAI_GAMES as a module-level dict,
# provide a lazy proxy. Direct attribute access works; iteration requires calling get_gvgai_games().
class _LazyDict:
    """Dict-like proxy that triggers discovery on first access."""
    def __getitem__(self, key):
        return get_gvgai_games()[key]
    def __contains__(self, key):
        return key in get_gvgai_games()
    def __len__(self):
        return len(get_gvgai_games())
    def __iter__(self):
        return iter(get_gvgai_games())
    def __bool__(self):
        return bool(get_gvgai_games())
    def values(self):
        return get_gvgai_games().values()
    def keys(self):
        return get_gvgai_games().keys()
    def items(self):
        return get_gvgai_games().items()
    def get(self, key, default=None):
        return get_gvgai_games().get(key, default)
    def __repr__(self):
        return repr(get_gvgai_games())

GVGAI_GAMES = _LazyDict()
