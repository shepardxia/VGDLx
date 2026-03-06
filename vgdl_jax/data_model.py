import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# py-vgdl physics operates in pixel coordinates where 1 grid cell = block_size pixels.
# vgdl-jax positions are in grid-cell units. All physics constants (forces, velocities)
# from VGDL files are in pixel units and must be divided by this scale factor.
PHYSICS_SCALE = 24

# Named constants
AABB_THRESHOLD = 1.0 - 1e-3        # collision overlap threshold (1.0 - AABB_EPS)
N_DIRECTIONS = 4                     # cardinal directions (UP, DOWN, LEFT, RIGHT)
DEFAULT_RESOURCE_LIMIT = 100         # fallback resource capacity
NOISY_AVATAR_NOISE_LEVEL = 0.4      # NoisyRotatingFlippingAvatar noise probability
GRAVITY_ACCEL = 1.0 / PHYSICS_SCALE  # standard gravity in grid-cell units
SPRITE_HEADROOM = 10                 # extra slots per type beyond level count


class SpriteClass(enum.IntEnum):
    IMMOVABLE = 0
    MISSILE = 1
    RANDOM_NPC = 2
    CHASER = 3
    FLEEING = 4
    FLICKER = 5
    SPAWN_POINT = 6
    BOMBER = 7
    WALKER = 8
    ORIENTED_FLICKER = 9
    # Avatar classes (handled specially but tracked here)
    MOVING_AVATAR = 10
    FLAK_AVATAR = 11
    SHOOT_AVATAR = 12
    HORIZONTAL_AVATAR = 13
    ORIENTED_AVATAR = 14
    # New sprite classes
    RESOURCE = 15
    PASSIVE = 16
    PORTAL = 17
    INERTIAL_AVATAR = 18
    MARIO_AVATAR = 19
    VERTICAL_AVATAR = 20
    CONVEYOR = 21
    ERRATIC_MISSILE = 22
    RANDOM_INERTIAL = 23
    RANDOM_MISSILE = 24
    ROTATING_AVATAR = 25
    ROTATING_FLIPPING_AVATAR = 26
    NOISY_ROTATING_FLIPPING_AVATAR = 27
    SHOOT_EVERYWHERE_AVATAR = 28
    AIMED_AVATAR = 29
    AIMED_FLAK_AVATAR = 30
    SPREADER = 31
    WALK_JUMPER = 32


class TerminationType(enum.IntEnum):
    SPRITE_COUNTER = 0
    MULTI_SPRITE_COUNTER = 1
    TIMEOUT = 2
    RESOURCE_COUNTER = 3


@dataclass(frozen=True)
class SpriteClassDef:
    vgdl_names: tuple               # VGDL parser names, e.g. ('Missile',)
    is_static: bool = False
    is_avatar: bool = False
    default_speed: float = 0.0      # 0.0 = stationary default, 1.0 = speed-1
    is_moving_npc: bool = False     # participates in NPC update loop
    # Avatar properties
    n_move_actions: int = 0
    can_shoot: bool = False
    is_horizontal: bool = False
    physics_type: str = 'grid'
    is_rotating: bool = False
    is_flipping: bool = False
    noise_level: float = 0.0
    shoot_everywhere: bool = False
    is_aimed: bool = False
    can_move_aimed: bool = False


# ---------------------------------------------------------------------------
# SPRITE_REGISTRY: single source of truth for per-SpriteClass metadata.
# Maps SpriteClass → SpriteClassDef.
# ---------------------------------------------------------------------------
SPRITE_REGISTRY: Dict[SpriteClass, SpriteClassDef] = {
    # --- Static / non-moving sprite classes ---
    SpriteClass.IMMOVABLE: SpriteClassDef(
        vgdl_names=('Immovable', 'Immutable'),
        is_static=True,
    ),
    SpriteClass.PASSIVE: SpriteClassDef(
        vgdl_names=('Passive',),
        is_static=True,
    ),
    SpriteClass.RESOURCE: SpriteClassDef(
        vgdl_names=('ResourcePack', 'Resource'),
        is_static=True,
    ),
    SpriteClass.PORTAL: SpriteClassDef(
        vgdl_names=('Portal',),
        is_static=True,
    ),
    SpriteClass.CONVEYOR: SpriteClassDef(
        vgdl_names=('Conveyor',),
        is_static=True,
    ),

    # --- Moving NPC classes (speed defaults to 1.0) ---
    SpriteClass.MISSILE: SpriteClassDef(
        vgdl_names=('Missile',),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.RANDOM_NPC: SpriteClassDef(
        vgdl_names=('RandomNPC',),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.CHASER: SpriteClassDef(
        vgdl_names=('Chaser', 'AStarChaser'),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.FLEEING: SpriteClassDef(
        vgdl_names=('Fleeing',),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.FLICKER: SpriteClassDef(
        vgdl_names=('Flicker',),
    ),
    SpriteClass.ORIENTED_FLICKER: SpriteClassDef(
        vgdl_names=('OrientedFlicker',),
    ),
    SpriteClass.SPAWN_POINT: SpriteClassDef(
        vgdl_names=('SpawnPoint',),
    ),
    SpriteClass.BOMBER: SpriteClassDef(
        vgdl_names=('Bomber',),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.WALKER: SpriteClassDef(
        vgdl_names=('Walker',),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.ERRATIC_MISSILE: SpriteClassDef(
        vgdl_names=('ErraticMissile',),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.RANDOM_INERTIAL: SpriteClassDef(
        vgdl_names=('RandomInertial',),
        default_speed=1.0,
        is_moving_npc=True,
        physics_type='continuous',
    ),
    SpriteClass.RANDOM_MISSILE: SpriteClassDef(
        vgdl_names=('RandomMissile',),
        default_speed=1.0,
        is_moving_npc=True,
    ),
    SpriteClass.SPREADER: SpriteClassDef(
        vgdl_names=('Spreader',),
    ),
    SpriteClass.WALK_JUMPER: SpriteClassDef(
        vgdl_names=('WalkJumper',),
        default_speed=1.0,
        is_moving_npc=True,
        physics_type='gravity',
    ),

    # --- Avatar classes ---
    SpriteClass.MOVING_AVATAR: SpriteClassDef(
        vgdl_names=('MovingAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4,
    ),
    SpriteClass.FLAK_AVATAR: SpriteClassDef(
        vgdl_names=('FlakAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=2, can_shoot=True, is_horizontal=True,
    ),
    SpriteClass.SHOOT_AVATAR: SpriteClassDef(
        vgdl_names=('ShootAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4, can_shoot=True,
    ),
    SpriteClass.HORIZONTAL_AVATAR: SpriteClassDef(
        vgdl_names=('HorizontalAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=2, is_horizontal=True,
    ),
    SpriteClass.ORIENTED_AVATAR: SpriteClassDef(
        vgdl_names=('OrientedAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4,
    ),
    SpriteClass.VERTICAL_AVATAR: SpriteClassDef(
        vgdl_names=('VerticalAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=2,
    ),
    SpriteClass.INERTIAL_AVATAR: SpriteClassDef(
        vgdl_names=('InertialAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4, physics_type='continuous',
    ),
    SpriteClass.MARIO_AVATAR: SpriteClassDef(
        vgdl_names=('MarioAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=5, physics_type='gravity',
    ),
    SpriteClass.ROTATING_AVATAR: SpriteClassDef(
        vgdl_names=('RotatingAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4, is_rotating=True,
    ),
    SpriteClass.ROTATING_FLIPPING_AVATAR: SpriteClassDef(
        vgdl_names=('RotatingFlippingAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4, is_rotating=True, is_flipping=True,
    ),
    SpriteClass.NOISY_ROTATING_FLIPPING_AVATAR: SpriteClassDef(
        vgdl_names=('NoisyRotatingFlippingAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4, is_rotating=True, is_flipping=True,
        noise_level=NOISY_AVATAR_NOISE_LEVEL,
    ),
    SpriteClass.SHOOT_EVERYWHERE_AVATAR: SpriteClassDef(
        vgdl_names=('ShootEverywhereAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4, can_shoot=True, shoot_everywhere=True,
    ),
    SpriteClass.AIMED_AVATAR: SpriteClassDef(
        vgdl_names=('AimedAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=2, can_shoot=True, is_aimed=True,
    ),
    SpriteClass.AIMED_FLAK_AVATAR: SpriteClassDef(
        vgdl_names=('AimedFlakAvatar',),
        is_avatar=True, default_speed=1.0,
        n_move_actions=4, can_shoot=True, is_aimed=True, can_move_aimed=True,
    ),
}

# Derived sets — computed from SPRITE_REGISTRY
STATIC_CLASSES = frozenset(
    sc for sc, d in SPRITE_REGISTRY.items() if d.is_static
)
AVATAR_CLASSES = frozenset(
    sc for sc, d in SPRITE_REGISTRY.items() if d.is_avatar
)
MOVING_NPC_CLASSES = frozenset(
    sc for sc, d in SPRITE_REGISTRY.items() if d.is_moving_npc
)


@dataclass
class SpriteDef:
    key: str
    type_idx: int
    sprite_class: int
    stypes: List[str]
    speed: float
    orientation: Tuple[float, float]
    cooldown: int
    is_static: bool
    singleton: bool
    flicker_limit: int
    spawner_stype: Optional[str]
    spawner_prob: float
    spawner_total: int
    color: Tuple[int, int, int]
    img: Optional[str]        # sprite image path, e.g. "oryx/alien1" (no .png extension)
    shrinkfactor: float       # 0.0 = full size, 0.15 = avatar default, up to ~0.6
    # Resource fields (for Resource sprite class)
    resource_name: Optional[str] = None   # resource type name (e.g. "diamond")
    resource_value: int = 1               # value when collected
    resource_limit: int = 1               # max amount of this resource
    # Portal fields
    portal_exit_stype: Optional[str] = None  # stype of the exit portal
    # Continuous/gravity physics fields
    physics_type: str = 'grid'           # 'grid', 'continuous', or 'gravity'
    mass: float = 1.0
    strength: float = 1.0               # force multiplier (VGDLSprite default)
    jump_strength: float = 10.0         # MarioAvatar upward impulse
    airsteering: bool = False            # MarioAvatar air control
    angle_diff: float = 0.05             # AimedAvatar rotation step (radians)


@dataclass
class EffectDef:
    effect_type: str   # internal key (e.g. 'kill_sprite'), see effects.EFFECT_REGISTRY
    actor_stype: str
    actee_stype: str
    score_change: int = 0
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationDef:
    term_type: int
    win: bool
    score_change: int = 0
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LevelDef:
    height: int
    width: int
    # List of (type_idx, y, x) for each sprite to place
    initial_sprites: List[Tuple[int, int, int]]


@dataclass
class GameDef:
    sprites: List[SpriteDef]
    effects: List[EffectDef]
    terminations: List[TerminationDef]
    level: Optional[LevelDef]
    char_mapping: Dict[str, List[str]]
    sprite_order: List[str]
    stype_to_indices: Dict[str, List[int]]

    def type_idx(self, key: str) -> int:
        for s in self.sprites:
            if s.key == key:
                return s.type_idx
        raise KeyError(f"Unknown sprite key: {key}")

    def resolve_stype(self, stype: str) -> List[int]:
        return self.stype_to_indices.get(stype, [])


@dataclass
class CompiledEffect:
    type_a: int
    is_eos: bool
    effect_type: str
    score_change: int
    max_a: int
    static_a_grid_idx: Optional[int] = None
    static_b_grid_idx: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Non-EOS fields (ignored when is_eos=True)
    type_b: int = -1
    collision_mode: str = 'grid'
    max_speed_cells: int = 1
    max_b: int = 0


@dataclass
class AvatarConfig:
    avatar_type_idx: int
    n_move_actions: int
    cooldown: int
    can_shoot: bool
    shoot_action_idx: int = -1
    projectile_type_idx: int = -1
    projectile_orientation_from_avatar: bool = False
    projectile_default_orientation: Tuple[float, float] = (0.0, 0.0)
    projectile_speed: float = 0.0
    direction_offset: int = 0
    physics_type: str = 'grid'
    mass: float = 1.0
    strength: float = 1.0
    jump_strength: float = 1.0
    airsteering: bool = False
    gravity: float = 1.0
    is_rotating: bool = False
    is_flipping: bool = False
    noise_level: float = 0.0
    shoot_everywhere: bool = False
    is_aimed: bool = False
    can_move_aimed: bool = False
    angle_diff: float = 0.05


@dataclass
class SpriteConfig:
    sprite_class: int
    cooldown: int
    flicker_limit: int = 0
    target_type_idx: int = -1
    prob: float = 1.0
    total: int = 0
    spreadprob: float = 1.0
    mass: float = 1.0
    strength: float = 1.0
    gravity: float = 0.0
    spawn_cooldown: int = 1
    target_orientation: Tuple[float, float] = (0.0, 0.0)
    target_speed: float = 0.0
