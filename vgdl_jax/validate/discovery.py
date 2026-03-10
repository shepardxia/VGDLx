"""Game discovery: scan directories for VGDL game/level file pairs."""
import os
import re
from dataclasses import dataclass, field


@dataclass
class GameEntry:
    name: str              # e.g. "aliens"
    game_file: str         # absolute path to game definition .txt
    level_files: list[str] = field(default_factory=list)  # absolute paths to _lvlN.txt, sorted by N
    source: str = "unknown"  # "pyvgdl", "gvgai", or custom label


_LVL_PATTERN = re.compile(r'^(.+)_lvl(\d+)\.txt$')
_EXCLUDE_SUFFIXES = ('_ggame.txt', '_glvl.txt')


def discover_games(game_dir: str, source: str = "unknown") -> list[GameEntry]:
    """Scan directory for VGDL game/level file pairs.

    Discovery logic:
    1. List all *.txt files
    2. Exclude files matching *_lvlN.txt, *_ggame.txt, *_glvl.txt
    3. Remaining .txt files are game definitions
    4. For each <name>.txt, find <name>_lvl<N>.txt files, sorted by N
    5. Skip games with 0 levels
    """
    game_dir = os.path.abspath(game_dir)
    if not os.path.isdir(game_dir):
        return []

    txt_files = [f for f in os.listdir(game_dir) if f.endswith('.txt')]

    # Separate level files from game files
    level_map: dict[str, list[tuple[int, str]]] = {}  # name -> [(N, path), ...]
    level_filenames = set()

    for fname in txt_files:
        m = _LVL_PATTERN.match(fname)
        if m:
            name, n = m.group(1), int(m.group(2))
            level_map.setdefault(name, []).append((n, os.path.join(game_dir, fname)))
            level_filenames.add(fname)

    # Game definition files: .txt that are NOT level files and NOT generator files
    entries = []
    for fname in sorted(txt_files):
        if fname in level_filenames:
            continue
        if any(fname.endswith(suffix) for suffix in _EXCLUDE_SUFFIXES):
            continue

        name = fname[:-4]  # strip .txt
        levels = level_map.get(name, [])
        if not levels:
            continue

        # Sort levels by N
        levels.sort(key=lambda x: x[0])
        level_paths = [path for _, path in levels]

        entries.append(GameEntry(
            name=name,
            game_file=os.path.join(game_dir, fname),
            level_files=level_paths,
            source=source,
        ))

    return entries
