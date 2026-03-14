"""
Standalone VGDL text parser — no pygame dependency.
Reads .txt game files and level files, produces a GameDef.
"""
from collections import defaultdict
from vgdl_jax.data_model import (
    SpriteClass, TerminationType, STATIC_CLASSES, SPRITE_REGISTRY,
    SpriteDef, EffectDef, TerminationDef, LevelDef, GameDef,
    PHYSICS_GRID, PHYSICS_CONTINUOUS, PHYSICS_GRAVITY,
)
from vgdl_jax.effects import VGDL_TO_KEY


# ── Derived from SPRITE_REGISTRY ──────────────────────────────────────

# VGDL class name → SpriteClass enum
CLASS_MAP = {
    name: sc
    for sc, d in SPRITE_REGISTRY.items()
    for name in d.vgdl_names
}

# Sprite classes that default to speed=1 (VGDL spec)
SPEED_1_CLASSES = frozenset(
    sc for sc, d in SPRITE_REGISTRY.items() if d.default_speed > 0
)


# VGDL convention: Vector2(x, y) with screen coords: UP=(0,-1), DOWN=(0,1)
# Our JAX format uses (row, col): UP=(-1,0), DOWN=(1,0)
ORIENTATION_MAP = {
    'UP': (-1.0, 0.0),
    'DOWN': (1.0, 0.0),
    'LEFT': (0.0, -1.0),
    'RIGHT': (0.0, 1.0),
}

# ── Color constants (VGDL standard colors) ──────────────────────────

COLOR_MAP = {
    'GREEN': (0, 200, 0),
    'BLUE': (0, 0, 200),
    'RED': (200, 0, 0),
    'GRAY': (90, 90, 90),
    'WHITE': (250, 250, 250),
    'BROWN': (140, 120, 100),
    'BLACK': (0, 0, 0),
    'ORANGE': (250, 160, 0),
    'YELLOW': (250, 250, 0),
    'PINK': (250, 200, 200),
    'GOLD': (250, 212, 0),
    'LIGHTRED': (250, 50, 50),
    'LIGHTORANGE': (250, 200, 100),
    'LIGHTBLUE': (50, 100, 250),
    'LIGHTGREEN': (50, 250, 50),
    'LIGHTGRAY': (150, 150, 150),
    'DARKGRAY': (30, 30, 30),
    'DARKBLUE': (20, 20, 100),
}

# Default color per sprite class name (matches ontology class definitions)
DEFAULT_CLASS_COLORS = {
    'Immovable': (90, 90, 90),       # GRAY
    'Immutable': (90, 90, 90),       # GRAY
    'Passive': (200, 0, 0),          # RED
    'ResourcePack': (90, 90, 90),    # GRAY
    'Resource': (200, 0, 0),         # RED
    'Missile': (250, 250, 250),      # WHITE
    'RandomNPC': (250, 250, 250),    # WHITE
    'Chaser': (250, 250, 250),       # WHITE
    'AStarChaser': (250, 250, 250),  # WHITE
    'Fleeing': (250, 250, 250),      # WHITE
    'Flicker': (200, 0, 0),          # RED
    'OrientedFlicker': (200, 0, 0),  # RED
    'SpawnPoint': (0, 0, 0),         # BLACK
    'Bomber': (250, 160, 0),         # ORANGE
    'Walker': (250, 250, 250),       # WHITE
    'MovingAvatar': (250, 250, 250), # WHITE
    'FlakAvatar': (0, 200, 0),       # GREEN
    'ShootAvatar': (250, 250, 250),  # WHITE
    'HorizontalAvatar': (250, 250, 250),
    'VerticalAvatar': (250, 250, 250),
    'OrientedAvatar': (250, 250, 250),
    'RotatingAvatar': (250, 250, 250),
    'InertialAvatar': (250, 250, 250),
    'MarioAvatar': (250, 250, 250),
}


# ── Indent tree parser ────────────────────────────────────────────────

class Node:
    """Lightweight indented tree node."""
    def __init__(self, content, indent, parent=None):
        self.content = content
        self.indent = indent
        self.children = []
        self.parent = None
        if parent:
            parent.insert(self)

    def insert(self, node):
        if self.indent < node.indent:
            if self.children:
                assert self.children[0].indent == node.indent
            self.children.append(node)
            node.parent = self
        else:
            assert self.parent, 'Root node too indented'
            self.parent.insert(node)

    def get_root(self):
        if self.parent:
            return self.parent.get_root()
        return self


def indent_tree_parser(s, tabsize=8):
    """Parse an indented string into a tree of Nodes."""
    s = s.expandtabs(tabsize)
    s = s.replace('(', ' ').replace(')', ' ').replace(',', ' ')
    last = Node("", -1)
    for line in s.split("\n"):
        if '#' in line:
            line = line.split('#')[0]
        content = line.strip()
        if content:
            indent = len(line) - len(line.lstrip())
            last = Node(content, indent, last)
    return last.get_root()


# ── Value parser ──────────────────────────────────────────────────────

def _parse_value(val_str):
    """Try to interpret a value string as a Python literal or known constant."""
    # Check orientation constants
    if val_str in ORIENTATION_MAP:
        return val_str  # Keep as string, convert later
    # Boolean
    if val_str in ('True', 'true'):
        return True
    if val_str in ('False', 'false'):
        return False
    # Number
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    # String (class name, sprite key, etc.)
    return val_str


def _parse_args(s):
    """Parse 'ClassName key=val key=val ...' into (class_name, kwargs)."""
    parts = [x.strip() for x in s.split() if x.strip()]
    if not parts:
        return None, {}
    class_name = None
    kwargs = {}
    start = 0
    if '=' not in parts[0]:
        class_name = parts[0]
        start = 1
    for part in parts[start:]:
        if '=' in part:
            k, v = part.split('=', 1)
            kwargs[k] = _parse_value(v)
    return class_name, kwargs


# ── Section parsers ───────────────────────────────────────────────────

def _parse_sprites(nodes, parent_class=None, parent_args=None, parent_types=None):
    """
    Recursively parse SpriteSet nodes.
    Returns list of (key, class_name, args, stypes) tuples.
    """
    if parent_args is None:
        parent_args = {}
    if parent_types is None:
        parent_types = []

    results = []
    for node in nodes:
        assert '>' in node.content, f"Sprite line missing '>': {node.content}"
        key, sdef = [x.strip() for x in node.content.split('>', 1)]
        class_name, args = _parse_args(sdef)
        # Inherit parent class and args
        if class_name is None:
            class_name = parent_class
        merged_args = {**parent_args, **args}
        stypes = parent_types + [key]

        if not node.children:
            # Leaf node — actual sprite type
            results.append((key, class_name, merged_args, stypes))
        else:
            # Non-leaf — recurse
            results.extend(_parse_sprites(
                node.children, class_name, merged_args, stypes))
    return results


def _parse_interactions(nodes):
    """Parse InteractionSet nodes into list of (actor, actee, effect_name, kwargs)."""
    results = []
    for node in nodes:
        if '>' not in node.content:
            continue
        pair, edef = [x.strip() for x in node.content.split('>', 1)]
        effect_name, kwargs = _parse_args(edef)
        objs = [x.strip() for x in pair.split() if x.strip()]
        actor = objs[0]
        for actee in objs[1:]:
            results.append((actor, actee, effect_name, kwargs))
    return results


def _parse_mappings(nodes):
    """Parse LevelMapping nodes into char → [key, ...] dict."""
    mapping = {}
    for node in nodes:
        if '>' not in node.content:
            continue
        c, val = [x.strip() for x in node.content.split('>', 1)]
        assert len(c) == 1, f"Only single character mappings allowed, got '{c}'"
        keys = [x.strip() for x in val.split() if x.strip()]
        mapping[c] = keys
    return mapping


def _parse_terminations(nodes):
    """Parse TerminationSet nodes into list of (class_name, kwargs)."""
    results = []
    for node in nodes:
        class_name, kwargs = _parse_args(node.content)
        results.append((class_name, kwargs))
    return results


# ── SpriteDef builder ─────────────────────────────────────────────────

def _build_sprite_def(key, class_name, args, stypes, type_idx):
    """Convert parsed sprite data into a SpriteDef."""
    # Map class name to SpriteClass enum
    sc = CLASS_MAP.get(class_name)
    if sc is None:
        raise ValueError(f"Unknown sprite class '{class_name}'")

    # Extract known parameters with class-based defaults
    # Moving sprite classes: MovingAvatar, RandomNPC, Chaser, Fleeing, Missile, Walker, Bomber
    # all default to speed=1
    default_speed = 1.0 if sc in SPEED_1_CLASSES else 0.0
    speed = args.get('speed', default_speed)
    if speed is None:
        speed = default_speed
    speed = float(speed)

    cooldown = int(args.get('cooldown', 0))

    # Orientation
    ori_val = args.get('orientation', None)
    if isinstance(ori_val, str) and ori_val in ORIENTATION_MAP:
        orientation = ORIENTATION_MAP[ori_val]
    else:
        orientation = (0.0, 1.0)  # default RIGHT in (row, col)

    is_static = sc in STATIC_CLASSES

    singleton = bool(args.get('singleton', False))

    # Flicker limit — only applies to Flicker/OrientedFlicker classes
    # (For Resource sprites, 'limit' means resource capacity, not expiry)
    if sc in (SpriteClass.FLICKER, SpriteClass.ORIENTED_FLICKER, SpriteClass.SPREADER):
        flicker_limit = int(args.get('limit', 1))
    else:
        flicker_limit = 0

    # Spawner stype (for SpawnPoint, Bomber, ShootAvatar, FlakAvatar, Chaser, Fleeing)
    spawner_stype = args.get('stype', None)
    if not isinstance(spawner_stype, str):
        spawner_stype = None

    # ErraticMissile and WalkJumper default to prob=0.1 (VGDL spec)
    default_prob = 0.1 if sc in (SpriteClass.ERRATIC_MISSILE, SpriteClass.WALK_JUMPER) else 1.0
    spawner_prob = float(args.get('prob', default_prob))
    spawner_total = int(args.get('total', 0))

    # Color: explicit override from game file > class default > white fallback
    color_val = args.get('color', None)
    if isinstance(color_val, str) and color_val in COLOR_MAP:
        color = COLOR_MAP[color_val]
    elif class_name and class_name in DEFAULT_CLASS_COLORS:
        color = DEFAULT_CLASS_COLORS[class_name]
    else:
        color = (250, 250, 250)  # WHITE fallback

    # Sprite image path (e.g. "oryx/alien1")
    img = args.get('img', None)
    if not isinstance(img, str):
        img = None

    # Shrinkfactor: default 0.0 for all types.
    # Note: VGDL Avatar mixin sets shrinkfactor=0.15, but VGDLSprite
    # sets shrinkfactor=0.0 and wins via MRO (MovingAvatar -> VGDLSprite -> Avatar).
    # So the actual runtime default for avatars is 0.0, matching non-avatars.
    shrinkfactor = float(args.get('shrinkfactor', 0.0))

    # Resource fields
    resource_name = None
    resource_value = 1
    resource_limit = 1
    if sc == SpriteClass.RESOURCE:
        # Resource sprites use their key as the resource name
        resource_name = key
        resource_value = int(args.get('value', 1))
        resource_limit = int(args.get('limit', 1))
        # res_type override
        if 'res_type' in args:
            resource_name = str(args['res_type'])

    # Portal fields
    portal_exit_stype = None
    if sc == SpriteClass.PORTAL:
        portal_exit_stype = args.get('stype', None)

    # Physics type inference
    physics_type = PHYSICS_GRID
    if sc in (SpriteClass.INERTIAL_AVATAR, SpriteClass.RANDOM_INERTIAL):
        physics_type = PHYSICS_CONTINUOUS
    elif sc in (SpriteClass.MARIO_AVATAR, SpriteClass.WALK_JUMPER):
        physics_type = PHYSICS_GRAVITY
    # Explicit physicstype kwarg overrides
    pt_kwarg = args.get('physicstype', None)
    if pt_kwarg == 'ContinuousPhysics':
        physics_type = PHYSICS_CONTINUOUS
    elif pt_kwarg == 'GravityPhysics':
        physics_type = PHYSICS_GRAVITY

    mass = float(args.get('mass', 1.0))
    # MarioAvatar defaults to strength=3, WalkJumper to 10, others to 1
    default_strength = 3.0 if sc == SpriteClass.MARIO_AVATAR else (10.0 if sc == SpriteClass.WALK_JUMPER else 1.0)
    strength = float(args.get('strength', default_strength))
    jump_strength = float(args.get('jump_strength', 10.0))
    airsteering = bool(args.get('airsteering', False))
    angle_diff = float(args.get('angle_diff', 0.05))
    cons = int(args.get('cons', 0))

    # RC8: FlakAvatar ammo fields
    ammo = args.get('ammo', None)
    if ammo is not None:
        ammo = str(ammo)
    min_ammo = int(args.get('minAmmo', -1))
    ammo_cost = int(args.get('ammoCost', 1))

    # RC2: rotateInPlace — OrientedAvatar subclasses default to True
    rotate_in_place_default = SPRITE_REGISTRY[sc].rotate_in_place_default
    rotate_in_place = bool(args.get('rotateInPlace', rotate_in_place_default))

    return SpriteDef(
        key=key,
        type_idx=type_idx,
        sprite_class=sc,
        stypes=stypes,
        speed=speed,
        orientation=orientation,
        cooldown=cooldown,
        is_static=is_static,
        singleton=singleton,
        flicker_limit=flicker_limit,
        spawner_stype=spawner_stype,
        spawner_prob=spawner_prob,
        spawner_total=spawner_total,
        color=color,
        img=img,
        shrinkfactor=shrinkfactor,
        resource_name=resource_name,
        resource_value=resource_value,
        resource_limit=resource_limit,
        portal_exit_stype=portal_exit_stype,
        physics_type=physics_type,
        mass=mass,
        strength=strength,
        jump_strength=jump_strength,
        airsteering=airsteering,
        angle_diff=angle_diff,
        cons=cons,
        ammo=ammo,
        min_ammo=min_ammo,
        ammo_cost=ammo_cost,
        rotate_in_place=rotate_in_place,
    )


# ── EffectDef builder ─────────────────────────────────────────────────

def _build_effect_def(actor, actee, effect_name, kwargs):
    """Convert parsed effect data into an EffectDef."""
    et = VGDL_TO_KEY.get(effect_name)
    if et is None:
        raise ValueError(f"Unknown effect '{effect_name}'")
    score_change = int(kwargs.get('scoreChange', 0))

    # Pass through all kwargs except scoreChange — let _compile_effect_kwargs
    # select what it needs for each effect type.
    eff_kwargs = {k: v for k, v in kwargs.items() if k != 'scoreChange'}

    return EffectDef(
        effect_type=et,
        actor_stype=actor,
        actee_stype=actee,
        score_change=score_change,
        kwargs=eff_kwargs,
    )


# ── TerminationDef builder ───────────────────────────────────────────

def _build_termination_def(class_name, kwargs):
    """Convert parsed termination data into a TerminationDef."""
    if class_name == 'SpriteCounter':
        return TerminationDef(
            term_type=TerminationType.SPRITE_COUNTER,
            win=bool(kwargs.get('win', False)),
            score_change=int(kwargs.get('scoreChange', 0)),
            kwargs={
                'stype': kwargs.get('stype', ''),
                'limit': int(kwargs.get('limit', 0)),
            },
        )
    elif class_name == 'MultiSpriteCounter':
        # stypeN=... kwargs
        stypes = {}
        for k, v in kwargs.items():
            if k.startswith('stype'):
                stypes[k] = v
        return TerminationDef(
            term_type=TerminationType.MULTI_SPRITE_COUNTER,
            win=bool(kwargs.get('win', False)),
            score_change=int(kwargs.get('scoreChange', 0)),
            kwargs={
                'stypes': list(stypes.values()),
                'limit': int(kwargs.get('limit', 0)),
            },
        )
    elif class_name == 'Timeout':
        return TerminationDef(
            term_type=TerminationType.TIMEOUT,
            win=bool(kwargs.get('win', False)),
            score_change=int(kwargs.get('scoreChange', 0)),
            kwargs={
                'limit': int(kwargs.get('limit', 0)),
            },
        )
    elif class_name == 'ResourceCounter':
        return TerminationDef(
            term_type=TerminationType.RESOURCE_COUNTER,
            win=bool(kwargs.get('win', False)),
            score_change=int(kwargs.get('scoreChange', 0)),
            kwargs={
                'resource': kwargs.get('stype', ''),
                'limit': int(kwargs.get('limit', 0)),
            },
        )
    else:
        raise ValueError(f"Unknown termination class '{class_name}'")


# ── Level parser ──────────────────────────────────────────────────────

def _parse_level(level_str, char_mapping, sprite_key_to_idx):
    """Parse a level string into a LevelDef."""
    lines = [l for l in level_str.split('\n') if l.strip()]
    height = len(lines)
    width = max(len(l) for l in lines) if lines else 0

    initial_sprites = []
    for row, line in enumerate(lines):
        for col, c in enumerate(line):
            keys = char_mapping.get(c, [])
            for key in keys:
                if key in sprite_key_to_idx:
                    initial_sprites.append((sprite_key_to_idx[key], row, col))

    return LevelDef(height=height, width=width, initial_sprites=initial_sprites)


# ── Main entry point ──────────────────────────────────────────────────

def parse_vgdl(game_file, level_file=None):
    """
    Parse a VGDL game file (and optional level file) into a GameDef.

    Args:
        game_file: path to the .txt game definition file
        level_file: path to the _lvlN.txt level file (optional)

    Returns:
        GameDef
    """
    with open(game_file) as f:
        game_text = f.read()

    level_text = None
    if level_file:
        with open(level_file) as f:
            level_text = f.read()

    return parse_vgdl_text(game_text, level_text)


def parse_vgdl_text(game_text, level_text=None):
    """
    Parse VGDL game text (and optional level text) into a GameDef.
    """
    tree = indent_tree_parser(game_text)
    # The root's first child is the game node (e.g. "BasicGame square_size=32")
    game_node = tree.children[0]

    # Parse game-level params from header (e.g. "BasicGame square_size=32")
    game_header_parts = game_node.content.strip().split()
    game_params = {}
    for part in game_header_parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            game_params[k] = v
    square_size = int(game_params.get('square_size', 0))

    raw_sprites = []
    raw_interactions = []
    # Default mapping matching GVGAI's VGDLFactory defaults
    raw_mappings = {'w': ['wall'], 'A': ['avatar']}
    raw_terminations = []

    for section in game_node.children:
        header = section.content.strip().split()[0]
        if header == 'SpriteSet':
            raw_sprites = _parse_sprites(section.children)
        elif header == 'InteractionSet':
            raw_interactions = _parse_interactions(section.children)
        elif header == 'LevelMapping':
            # Merge with defaults — explicit mappings override
            raw_mappings.update(_parse_mappings(section.children))
        elif header == 'TerminationSet':
            raw_terminations = _parse_terminations(section.children)

    # Build SpriteDefs with assigned type indices
    sprite_defs = []
    sprite_order = []
    for i, (key, class_name, args, stypes) in enumerate(raw_sprites):
        sd = _build_sprite_def(key, class_name, args, stypes, type_idx=i)
        sprite_defs.append(sd)
        sprite_order.append(key)

    # Build key → type_idx lookup
    key_to_idx = {sd.key: sd.type_idx for sd in sprite_defs}

    # Build stype_to_indices: maps each stype name → list of type indices
    stype_to_indices = defaultdict(list)
    for sd in sprite_defs:
        for stype in sd.stypes:
            if stype != sd.key:  # parent types
                stype_to_indices[stype].append(sd.type_idx)
        # The key itself maps to just this sprite
        stype_to_indices[sd.key].append(sd.type_idx)

    # Build EffectDefs
    effect_defs = []
    for actor, actee, effect_name, kwargs in raw_interactions:
        ed = _build_effect_def(actor, actee, effect_name, kwargs)
        effect_defs.append(ed)

    # Build TerminationDefs
    term_defs = []
    for class_name, kwargs in raw_terminations:
        td = _build_termination_def(class_name, kwargs)
        term_defs.append(td)

    # Parse level if provided
    level_def = None
    if level_text:
        level_def = _parse_level(level_text, raw_mappings, key_to_idx)

    return GameDef(
        sprites=sprite_defs,
        effects=effect_defs,
        terminations=term_defs,
        level=level_def,
        char_mapping=raw_mappings,
        sprite_order=sprite_order,
        stype_to_indices=dict(stype_to_indices),
        square_size=square_size,
    )
