"""Tests for game discovery from directories."""
import os
import pytest

from vgdl_jax.validate.discovery import discover_games, GameEntry
from vgdl_jax.validate.constants import PYVGDL_GAMES_DIR, GVGAI_GAMES_DIR


def test_discover_pyvgdl_games():
    """Discover games from py-vgdl directory — should find all 9 known games."""
    entries = discover_games(PYVGDL_GAMES_DIR, source='pyvgdl')
    names = {e.name for e in entries}

    expected = {'chase', 'zelda', 'aliens', 'missilecommand', 'sokoban',
                'portals', 'boulderdash', 'survivezombies', 'frogs'}
    assert expected == names, f"Missing: {expected - names}, extra: {names - expected}"

    for e in entries:
        assert e.source == 'pyvgdl'
        assert os.path.isfile(e.game_file)
        assert len(e.level_files) >= 1
        for lf in e.level_files:
            assert os.path.isfile(lf)


def test_discover_gvgai_games():
    """Discover games from GVGAI directory — should find 100+ games."""
    if not os.path.isdir(GVGAI_GAMES_DIR):
        pytest.skip("GVGAI directory not found")

    entries = discover_games(GVGAI_GAMES_DIR, source='gvgai')
    assert len(entries) > 50, f"Expected 100+ GVGAI games, found {len(entries)}"

    for e in entries:
        assert e.source == 'gvgai'
        assert os.path.isfile(e.game_file)
        assert len(e.level_files) >= 1


def test_discover_excludes_generator_files():
    """Generator files (*_ggame.txt, *_glvl.txt) should not be treated as games."""
    if not os.path.isdir(GVGAI_GAMES_DIR):
        pytest.skip("GVGAI directory not found")

    entries = discover_games(GVGAI_GAMES_DIR, source='gvgai')
    names = {e.name for e in entries}

    # These are generator suffixes, not game names
    for name in names:
        assert not name.endswith('_ggame'), f"Generator file leaked: {name}"
        assert not name.endswith('_glvl'), f"Generator file leaked: {name}"


def test_level_files_sorted_by_index():
    """Level files should be sorted by their numeric index."""
    entries = discover_games(PYVGDL_GAMES_DIR, source='pyvgdl')

    for e in entries:
        if len(e.level_files) > 1:
            # Extract level numbers from filenames
            nums = []
            for lf in e.level_files:
                base = os.path.basename(lf)
                # <name>_lvl<N>.txt
                n_str = base.replace(f'{e.name}_lvl', '').replace('.txt', '')
                nums.append(int(n_str))
            assert nums == sorted(nums), f"{e.name}: levels not sorted: {nums}"


def test_discover_nonexistent_dir():
    """Discovering from a nonexistent directory returns empty list."""
    entries = discover_games('/nonexistent/path', source='test')
    assert entries == []


def test_game_entry_fields():
    """GameEntry from discovery has all expected fields populated."""
    entries = discover_games(PYVGDL_GAMES_DIR, source='pyvgdl')
    assert len(entries) > 0

    e = entries[0]
    assert isinstance(e.name, str) and len(e.name) > 0
    assert isinstance(e.game_file, str) and e.game_file.endswith('.txt')
    assert isinstance(e.level_files, list) and len(e.level_files) > 0
    assert e.source == 'pyvgdl'


def test_constants_pyvgdl_games_dict():
    """PYVGDL_GAMES constant is a populated dict of GameEntry objects."""
    from vgdl_jax.validate.constants import PYVGDL_GAMES
    assert isinstance(PYVGDL_GAMES, dict)
    assert len(PYVGDL_GAMES) == 9
    for name, entry in PYVGDL_GAMES.items():
        assert isinstance(entry, GameEntry)
        assert entry.name == name
