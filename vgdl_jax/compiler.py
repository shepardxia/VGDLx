"""
Compiler: converts a GameDef (parsed VGDL) into a jit-compiled step function
and an initial GameState.
"""
import math
import warnings
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from vgdl_jax.data_model import (
    GameDef, SpriteClass, TerminationType,
    STATIC_CLASSES, AVATAR_CLASSES, MOVING_NPC_CLASSES, SPRITE_REGISTRY,
    CompiledEffect, AvatarConfig, SpriteConfig,
    DEFAULT_RESOURCE_LIMIT,
    SPRITE_HEADROOM, N_DIRECTIONS,
    PHYSICS_CONTINUOUS, PHYSICS_GRAVITY,
    speed_to_pixels, get_block_size,
    GVGAI_ACTION_TO_DIR,
)
from vgdl_jax.effects import compile_effect_kwargs, CompileContext, ALIVE_MODIFYING_EFFECTS, POSITION_MODIFYING_EFFECTS
from vgdl_jax.state import GameState, create_initial_state
from vgdl_jax.step import build_step_fn
from vgdl_jax.sprites import DIRECTION_DELTAS
from vgdl_jax.terminations import check_sprite_counter, check_multi_sprite_counter, check_timeout, check_resource_counter


@dataclass
class CompiledGame:
    init_state: GameState
    step_fn: Callable
    n_actions: int
    noop_action: int
    game_def: GameDef
    static_grid_map: dict  # type_idx → static_grid_idx (empty if no static types)
    action_names: tuple = ()  # GVGAI action names in order, e.g. ('ACTION_LEFT', ...)


def _resolve_first(game_def, stype, default=None):
    """Resolve stype to list of indices, return first or default."""
    if stype is None:
        return default
    indices = game_def.resolve_stype(stype)
    return indices[0] if indices else default


def _find_avatar(game_def):
    """Find the avatar SpriteDef."""
    for sd in game_def.sprites:
        if sd.sprite_class in AVATAR_CLASSES:
            return sd
    raise AssertionError("No avatar found in game definition")


def _find_all_avatar_types(game_def):
    """Find all avatar type indices (for games with multiple avatar subtypes)."""
    return tuple(sd.type_idx for sd in game_def.sprites
                 if sd.sprite_class in AVATAR_CLASSES)


def _build_resource_registry(game_def):
    """Build resource name→index mapping and limits list from sprites and effects."""
    resource_name_to_idx = {}
    resource_limits = []
    for sd in game_def.sprites:
        if sd.sprite_class == SpriteClass.RESOURCE and sd.resource_name:
            if sd.resource_name not in resource_name_to_idx:
                resource_name_to_idx[sd.resource_name] = len(resource_limits)
                resource_limits.append(sd.resource_limit)
    for ed in game_def.effects:
        res_name = ed.kwargs.get('resource', None)
        if res_name and res_name not in resource_name_to_idx:
            limit = ed.kwargs.get('limit', DEFAULT_RESOURCE_LIMIT)
            for sd in game_def.sprites:
                if sd.resource_name == res_name:
                    limit = sd.resource_limit
                    break
            resource_name_to_idx[res_name] = len(resource_limits)
            resource_limits.append(limit)
    return resource_name_to_idx, resource_limits


def _find_static_types(game_def):
    """Identify types stored as static grids instead of position arrays.

    Returns (static_type_indices, static_type_set, static_grid_map).
    """
    spawn_targets = set()
    for ed in game_def.effects:
        if ed.effect_type in ('transform_to', 'transform_others_to',
                               'spawn', 'spawn_if_has_more', 'spawn_if_has_less',
                               'clone_sprite', 'transform_to_singleton'):
            stype = ed.kwargs.get('stype', '') or ed.kwargs.get('target', '')
            for idx in game_def.resolve_stype(stype):
                spawn_targets.add(idx)
            # transformToSingleton's stype_other also needs to be non-static
            # (existing new_type sprites transform back to other_type)
            other_stype = ed.kwargs.get('stype_other', '')
            if other_stype:
                for idx in game_def.resolve_stype(other_stype):
                    spawn_targets.add(idx)
    for sd in game_def.sprites:
        if sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER,
                                SpriteClass.SPREADER):
            if sd.spawner_stype:
                for idx in game_def.resolve_stype(sd.spawner_stype):
                    spawn_targets.add(idx)

    FORCE_MOVE_EFFECTS = {
        'bounce_forward', 'pull_with_it', 'convey_sprite',
        'wind_gust', 'slip_forward', 'teleport_to_exit', 'wrap_around',
    }
    MODIFY_TYPE_A_EFFECTS = {
        'transform_to', 'transform_to_singleton', 'clone_sprite',
        'change_resource',
    }
    DISQUALIFYING_EFFECTS = FORCE_MOVE_EFFECTS | MODIFY_TYPE_A_EFFECTS
    position_modified_types = set()
    for ed in game_def.effects:
        if ed.effect_type in DISQUALIFYING_EFFECTS:
            for idx in game_def.resolve_stype(ed.actor_stype):
                position_modified_types.add(idx)

    STATIC_GRID_CLASSES = STATIC_CLASSES - {SpriteClass.PORTAL}

    teleport_exit_types = set()
    for ed in game_def.effects:
        if ed.effect_type == 'teleport_to_exit':
            actee_indices = game_def.resolve_stype(ed.actee_stype)
            for aidx in actee_indices:
                portal_sd = game_def.sprites[aidx]
                if portal_sd.portal_exit_stype:
                    for eidx in game_def.resolve_stype(portal_sd.portal_exit_stype):
                        teleport_exit_types.add(eidx)

    static_type_indices = []
    static_type_set = set()
    for sd in game_def.sprites:
        if (sd.sprite_class in STATIC_GRID_CLASSES
                and sd.speed == 0
                and sd.type_idx not in spawn_targets
                and sd.type_idx not in position_modified_types
                and sd.type_idx not in teleport_exit_types):
            static_type_indices.append(sd.type_idx)
            static_type_set.add(sd.type_idx)

    static_grid_map = {ti: i for i, ti in enumerate(static_type_indices)}
    return static_type_indices, static_type_set, static_grid_map


def _select_collision_mode(a_static, b_static, speed_a_px, speed_b_px, block_size):
    """Select collision detection mode for an effect pair.

    With int32 pixel coordinates, collision modes are:
    - grid: both sprites land on cell boundaries (speed_px % block_size == 0)
    - pixel_aabb: at least one sprite can land mid-cell
    - sweep: speed exceeds one cell per tick
    - static_b_grid / static_a_grid / static_both: one or both are static grids
    """
    if a_static and b_static:
        return 'static_both'
    elif b_static:
        return 'static_b_grid'
    elif a_static:
        return 'static_a_grid'
    elif speed_a_px > block_size or speed_b_px > block_size:
        return 'sweep'
    elif (speed_a_px % block_size != 0) or (speed_b_px % block_size != 0):
        return 'pixel_aabb'
    else:
        return 'grid'


def _build_avatar_config(avatar_sd, game_def, block_size, avatar_type_indices=(),
                          resource_name_to_idx=None):
    """Build AvatarConfig, n_actions, and n_move from the avatar SpriteDef."""
    sc_def = SPRITE_REGISTRY[avatar_sd.sprite_class]
    n_move = sc_def.n_move_actions
    can_shoot = sc_def.can_shoot
    direction_offset = 2 if sc_def.is_horizontal else 0

    proj_type_idx = -1
    proj_ori_from_avatar = False
    proj_default_ori = [0., 0.]
    proj_speed = 0.0
    shoot_action_idx = -1

    if can_shoot and avatar_sd.spawner_stype:
        proj_type_idx = _resolve_first(game_def, avatar_sd.spawner_stype, -1)
        if proj_type_idx >= 0:
            proj_sd = game_def.sprites[proj_type_idx]
            proj_speed = speed_to_pixels(proj_sd.speed, block_size, proj_sd.physics_type)
            if avatar_sd.sprite_class == SpriteClass.SHOOT_AVATAR:
                proj_ori_from_avatar = True
            else:
                proj_default_ori = list(proj_sd.orientation)
                proj_ori_from_avatar = False
        shoot_action_idx = n_move + 1

    # Ammo resource lookup for FlakAvatar / AimedFlakAvatar
    ammo_resource_idx = -1
    min_ammo = -1
    ammo_cost = 1
    if avatar_sd.ammo and resource_name_to_idx:
        ammo_resource_idx = resource_name_to_idx.get(avatar_sd.ammo, -1)
        min_ammo = avatar_sd.min_ammo
        ammo_cost = avatar_sd.ammo_cost

    # projectile_offset — ShootAvatar/ShootEverywhereAvatar spawn one cell ahead
    projectile_offset = sc_def.projectile_offset

    avatar_config = AvatarConfig(
        avatar_type_indices=avatar_type_indices,
        n_move_actions=n_move,
        cooldown=max(avatar_sd.cooldown, 1),
        can_shoot=can_shoot,
        shoot_action_idx=shoot_action_idx,
        projectile_type_idx=proj_type_idx,
        projectile_orientation_from_avatar=proj_ori_from_avatar,
        projectile_default_orientation=tuple(proj_default_ori),
        projectile_speed=proj_speed,
        projectile_singleton=(proj_type_idx >= 0 and game_def.sprites[proj_type_idx].singleton),
        direction_offset=direction_offset,
        physics_type=avatar_sd.physics_type,
        mass=avatar_sd.mass,
        strength=float(avatar_sd.strength),
        jump_strength=float(avatar_sd.jump_strength),
        airsteering=avatar_sd.airsteering,
        gravity=1.0,
        is_rotating=sc_def.is_rotating,
        is_flipping=sc_def.is_flipping,
        noise_level=sc_def.noise_level,
        shoot_everywhere=sc_def.shoot_everywhere,
        is_aimed=sc_def.is_aimed,
        can_move_aimed=sc_def.can_move_aimed,
        angle_diff=avatar_sd.angle_diff,
        ammo_resource_idx=ammo_resource_idx,
        min_ammo=min_ammo,
        ammo_cost=ammo_cost,
        rotate_in_place=avatar_sd.rotate_in_place,
        projectile_offset=projectile_offset,
        is_missile_avatar=(avatar_sd.sprite_class == SpriteClass.MISSILE_AVATAR),
    )
    return avatar_config, n_move


def _build_action_map(sc_def, avatar_config):
    """Build action_map: GVGAI action index → VGDLx internal action index.

    Uses gvgai_actions from the SpriteClassDef to determine the GVGAI ordering,
    then maps each to the corresponding internal action index.
    """
    n_move = avatar_config.n_move_actions
    direction_offset = avatar_config.direction_offset
    shoot_internal = avatar_config.shoot_action_idx  # n_move + 1 if can_shoot, else -1

    # Build name→internal mapping based on avatar type.
    # Each case defines the complete set of valid action names.
    if sc_def.physics_type == PHYSICS_GRAVITY:
        # MarioAvatar: LEFT=0, RIGHT=1, JUMP=2, JUMP_LEFT=3, JUMP_RIGHT=4, NOOP=5
        mapping = {
            'ACTION_NIL': 5, 'ACTION_USE': 2,
            'ACTION_LEFT': 0, 'ACTION_RIGHT': 1,
        }
    elif sc_def.is_rotating:
        # Ego-centric: forward=0, back=1, CCW=2, CW=3, NOOP=n_move
        mapping = {
            'ACTION_NIL': n_move, 'ACTION_USE': shoot_internal,
            'ACTION_UP': 0, 'ACTION_DOWN': 1,
            'ACTION_LEFT': 2, 'ACTION_RIGHT': 3,
        }
    elif sc_def.is_aimed and sc_def.can_move_aimed:
        # AimedFlakAvatar: left=0, right=1, aim_up=2, aim_down=3
        mapping = {
            'ACTION_NIL': n_move, 'ACTION_USE': shoot_internal,
            'ACTION_LEFT': 0, 'ACTION_RIGHT': 1,
            'ACTION_UP': 2, 'ACTION_DOWN': 3,
        }
    elif sc_def.is_aimed:
        # AimedAvatar: aim_up=0, aim_down=1
        mapping = {
            'ACTION_NIL': n_move, 'ACTION_USE': shoot_internal,
            'ACTION_UP': 0, 'ACTION_DOWN': 1,
        }
    else:
        # General case: direction actions via GVGAI_ACTION_TO_DIR
        mapping = {'ACTION_NIL': n_move, 'ACTION_USE': shoot_internal}
        for name, dir_idx in GVGAI_ACTION_TO_DIR.items():
            mapping[name] = dir_idx - direction_offset

    action_map = []
    for name in sc_def.gvgai_actions:
        if name not in mapping:
            raise ValueError(f"Unexpected action {name} for {sc_def.vgdl_names}")
        action_map.append(mapping[name])
    return jnp.array(action_map, dtype=jnp.int32)


def _build_sprite_configs(game_def, block_size):
    """Build SpriteConfig list for all sprite types."""
    sprite_configs = []
    for sd in game_def.sprites:
        cfg = SpriteConfig(
            sprite_class=sd.sprite_class,
            cooldown=max(sd.cooldown, 1),
            flicker_limit=sd.flicker_limit,
        )

        if sd.sprite_class == SpriteClass.RANDOM_NPC:
            cfg.cons = sd.cons
        elif sd.sprite_class in (SpriteClass.CHASER, SpriteClass.FLEEING):
            cfg.target_type_idx = _resolve_first(game_def, sd.spawner_stype, 0)
        elif sd.sprite_class == SpriteClass.SPREADER:
            cfg.spreadprob = sd.spawner_prob
            cfg.target_type_idx = _resolve_first(game_def, sd.spawner_stype, -1)
        elif sd.sprite_class == SpriteClass.ERRATIC_MISSILE:
            cfg.prob = sd.spawner_prob
        elif sd.sprite_class == SpriteClass.RANDOM_INERTIAL:
            cfg.mass = sd.mass
            cfg.strength = float(sd.strength)
        elif sd.sprite_class == SpriteClass.WALK_JUMPER:
            cfg.mass = sd.mass
            cfg.strength = float(sd.strength)
            cfg.prob = sd.spawner_prob
            cfg.gravity = 1.0
        elif sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            target_idx = _resolve_first(game_def, sd.spawner_stype, -1)
            cfg.target_type_idx = target_idx if target_idx >= 0 else 0
            cfg.prob = sd.spawner_prob
            cfg.total = sd.spawner_total
            # Both SpawnPoint and Bomber use spawn_cooldown for spawn timing
            cfg.spawn_cooldown = cfg.cooldown
            # Singleton check — refuse to spawn if target already alive
            if target_idx >= 0:
                cfg.target_singleton = game_def.sprites[target_idx].singleton
                target_sd = game_def.sprites[target_idx]
                # GVGAI SpawnPoint.update() orientation logic:
                # 1. If spawnorientation != DNONE: use spawnorientation
                # 2. Else if newSprite.orientation == DNONE: use SpawnPoint's orientation
                # 3. Else: keep target's orientation
                # In VGDLx, DNONE maps to (0,0). A target with no explicit orientation
                # would have DNONE in GVGAI, so we use the SpawnPoint's orientation.
                if sd.spawn_orientation != (0.0, 0.0):
                    cfg.target_orientation = tuple(sd.spawn_orientation)
                elif not target_sd.orientation_explicit:
                    # Target has no explicit orientation (= GVGAI DNONE)
                    # → use SpawnPoint's own orientation
                    cfg.target_orientation = tuple(sd.orientation)
                else:
                    cfg.target_orientation = tuple(target_sd.orientation)
                cfg.target_speed = speed_to_pixels(target_sd.speed, block_size, target_sd.physics_type)
                # RandomNPC targets need cons initialization (DNONE for first cons ticks)
                if target_sd.sprite_class == SpriteClass.RANDOM_NPC and target_sd.cons > 0:
                    cfg.target_cons = target_sd.cons
                # If target is itself a SpawnPoint/Bomber AND the spawner has a
                # higher type_idx (so it processes first in the reverse NPC loop),
                # the target gets _pre_spawn on its spawn tick. Set target_spawn_cd
                # so spawn_timers are initialized dynamically at runtime based on
                # step_count, matching GVGAI's (start+gameTick)%cd formula.
                if target_sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
                    if sd.type_idx > target_idx:
                        cfg.target_spawn_cd = max(target_sd.cooldown, 1)
            else:
                cfg.target_orientation = (0., 0.)
                cfg.target_speed = 0

        sprite_configs.append(cfg)
    return sprite_configs


def _build_compiled_effects(game_def, static_type_set, static_grid_map,
                             type_max_n, resource_name_to_idx, resource_limits,
                             avatar_type_idx, block_size):
    """Build list of CompiledEffects from game definition."""
    # Pixel speeds for collision mode detection
    _speed_px_by_idx = {sd.type_idx: speed_to_pixels(sd.speed, block_size, sd.physics_type)
                        for sd in game_def.sprites}

    compiled_effects = []
    for ed in game_def.effects:
        is_eos = (ed.actee_stype == 'EOS') or (ed.actor_stype == 'EOS')
        # GVGAI duplicates effects N times for repeat=N, enabling chain propagation
        repeat = int(ed.kwargs.get('repeat', 1))

        if is_eos:
            # EOS can appear as either actor or actee:
            #   sprite EOS > effect  (actee is EOS, sprites from actor_stype)
            #   EOS sprite > effect  (actor is EOS, sprites from actee_stype)
            if ed.actor_stype == 'EOS':
                eos_sprite_indices = game_def.resolve_stype(ed.actee_stype)
            else:
                eos_sprite_indices = game_def.resolve_stype(ed.actor_stype)
            for ta_idx in eos_sprite_indices:
                ce = CompiledEffect(
                    type_a=ta_idx,
                    is_eos=True,
                    effect_type=ed.effect_type,
                    score_change=ed.score_change,
                    max_a=type_max_n[ta_idx],
                    static_a_grid_idx=static_grid_map.get(ta_idx),
                    kwargs=compile_effect_kwargs(ed, CompileContext(
                        game_def, resource_name_to_idx, resource_limits,
                        avatar_type_idx, concrete_actor_idx=ta_idx,
                        concrete_actee_idx=None,
                        resolve_first=_resolve_first,
                        block_size=block_size)),
                )
                for _ in range(repeat):
                    compiled_effects.append(ce)
        else:
            actor_indices = game_def.resolve_stype(ed.actor_stype)
            actee_indices = game_def.resolve_stype(ed.actee_stype)
            for ta_idx in actor_indices:
                for tb_idx in actee_indices:
                    speed_a_px = _speed_px_by_idx.get(ta_idx, 0)
                    speed_b_px = _speed_px_by_idx.get(tb_idx, 0)
                    collision_mode = _select_collision_mode(
                        a_static=ta_idx in static_type_set,
                        b_static=tb_idx in static_type_set,
                        speed_a_px=speed_a_px, speed_b_px=speed_b_px,
                        block_size=block_size)
                    # max_speed_cells: for sweep collision
                    max_speed_px = max(speed_a_px, speed_b_px, 1)
                    eff_kwargs = compile_effect_kwargs(ed, CompileContext(
                        game_def, resource_name_to_idx, resource_limits,
                        avatar_type_idx, concrete_actor_idx=ta_idx,
                        concrete_actee_idx=tb_idx,
                        resolve_first=_resolve_first,
                        block_size=block_size))
                    # Per-instance needs_partner: e.g. transformTo with killSecond
                    needs_partner = eff_kwargs.get('kill_second', False)
                    ce = CompiledEffect(
                        type_a=ta_idx,
                        type_b=tb_idx,
                        is_eos=False,
                        effect_type=ed.effect_type,
                        score_change=ed.score_change,
                        collision_mode=collision_mode,
                        max_speed_cells=max(1, math.ceil(max_speed_px / block_size)),
                        max_a=type_max_n[ta_idx],
                        max_b=type_max_n[tb_idx],
                        static_a_grid_idx=static_grid_map.get(ta_idx),
                        static_b_grid_idx=static_grid_map.get(tb_idx),
                        kwargs=eff_kwargs,
                        needs_partner=needs_partner,
                    )
                    for _ in range(repeat):
                        compiled_effects.append(ce)
    return compiled_effects


def _build_compiled_terminations(game_def, static_type_set, static_grid_map,
                                  resource_name_to_idx, avatar_type_idx):
    """Build list of (check_fn, score_change) termination tuples."""
    compiled_terminations = []
    for td in game_def.terminations:
        if td.term_type == TerminationType.SPRITE_COUNTER:
            stype = td.kwargs.get('stype', '')
            indices = game_def.resolve_stype(stype)
            dyn_idx = [i for i in indices if i not in static_type_set]
            sg_idx = [static_grid_map[i] for i in indices if i in static_type_set]
            limit = td.kwargs.get('limit', 0)
            check_fn = _make_sprite_counter(dyn_idx, sg_idx, limit, td.win)
            compiled_terminations.append((check_fn, td.score_change))
        elif td.term_type == TerminationType.MULTI_SPRITE_COUNTER:
            stypes_list = td.kwargs.get('stypes', [])
            indices_list = [game_def.resolve_stype(st) for st in stypes_list]
            dyn_indices_list = [[i for i in grp if i not in static_type_set]
                                for grp in indices_list]
            sg_indices_list = [[static_grid_map[i] for i in grp if i in static_type_set]
                               for grp in indices_list]
            limit = td.kwargs.get('limit', 0)
            check_fn = _make_multi_sprite_counter(dyn_indices_list, sg_indices_list,
                                                   limit, td.win)
            compiled_terminations.append((check_fn, td.score_change))
        elif td.term_type == TerminationType.RESOURCE_COUNTER:
            res_name = td.kwargs.get('resource', '')
            r_idx = resource_name_to_idx.get(res_name, 0)
            limit = td.kwargs.get('limit', 0)
            check_fn = _make_resource_counter(avatar_type_idx, r_idx, limit, td.win)
            compiled_terminations.append((check_fn, td.score_change))
        elif td.term_type == TerminationType.TIMEOUT:
            limit = td.kwargs.get('limit', 0)
            check_fn = _make_timeout(limit, td.win)
            compiled_terminations.append((check_fn, td.score_change))
    return compiled_terminations



def _is_chaser_target_stable(target_idx, game_def, sprite_configs,
                              compiled_effects, avatar_type_idx):
    """Check if a chaser target's positions/alive never change at runtime.

    General analysis based on game dynamics, not specific games.
    """
    if target_idx == avatar_type_idx:
        return False
    sd = game_def.sprites[target_idx]
    cfg = sprite_configs[target_idx]
    # Moving NPC?
    if cfg.sprite_class in MOVING_NPC_CLASSES and sd.speed > 0:
        return False
    # Modified by effects?
    _ALL_MODIFYING = ALIVE_MODIFYING_EFFECTS | POSITION_MODIFYING_EFFECTS
    for eff in compiled_effects:
        if eff.is_eos:
            continue
        if eff.type_a == target_idx and eff.effect_type in _ALL_MODIFYING:
            return False
        if eff.type_b == target_idx and eff.effect_type in ('kill_both', 'transform_others_to'):
            return False
    # Spawn target?
    for cfg2 in sprite_configs:
        if cfg2.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER,
                                  SpriteClass.SPREADER):
            if hasattr(cfg2, 'target_type_idx') and cfg2.target_type_idx == target_idx:
                return False
    return True


def compile_game(game_def: GameDef, max_sprites_per_type=None):
    """Compile a GameDef into a CompiledGame with jitted step function."""
    n_types = len(game_def.sprites)
    height = game_def.level.height
    width = game_def.level.width

    avatar_sd = _find_avatar(game_def)
    resource_name_to_idx, resource_limits = _build_resource_registry(game_def)
    static_type_indices, static_type_set, static_grid_map = _find_static_types(game_def)
    n_static = len(static_type_indices)

    # GVGAI speed conversion: square_size overrides calculated block_size
    block_size = get_block_size(game_def)

    # Compute per-type max_n from level sprite counts + headroom
    if max_sprites_per_type is None:
        active_types = _find_active_types(game_def, avatar_sd)
        counts = defaultdict(int)
        for type_idx, _, _ in game_def.level.initial_sprites:
            counts[type_idx] += 1
        type_max_n = []
        for idx in range(n_types):
            if idx in static_type_set:
                type_max_n.append(0)
            else:
                base = counts.get(idx, 0)
                type_max_n.append(max(base + SPRITE_HEADROOM, SPRITE_HEADROOM) if idx in active_types else 1)
        max_n = max(max(type_max_n), SPRITE_HEADROOM)
    else:
        max_n = max_sprites_per_type
        type_max_n = [max_n] * n_types

    # Build initial state
    n_resource_types = len(resource_limits)
    state = create_initial_state(n_types=n_types, max_n=max_n,
                                 height=height, width=width,
                                 n_resource_types=n_resource_types,
                                 n_static_types=n_static)

    static_grid_data = np.zeros((max(n_static, 1), height, width), dtype=bool)
    slot_counts = defaultdict(int)
    for type_idx, row, col in game_def.level.initial_sprites:
        if type_idx in static_type_set:
            static_grid_data[static_grid_map[type_idx], row, col] = True
            continue
        slot = slot_counts[type_idx]
        if slot >= max_n:
            continue
        sd = game_def.sprites[type_idx]
        speed_px = speed_to_pixels(sd.speed, block_size, sd.physics_type)
        cd = max(sd.cooldown, 1)
        sc_def = SPRITE_REGISTRY.get(sd.sprite_class)
        # SpawnPoint/Bomber: spawn_timers=cd so first _pre_spawn increments
        # to cd+1 >= cd, firing at tick 0 (matching GVGAI: gameTick starts at -1,
        # first ++ gives 0, and (start+0)%cd==0 fires immediately).
        # Movement uses cooldown_timers (GVGAI lastmove starts at 0)
        # SpawnPoint: cooldown_timers=0 (doesn't move)
        # Bomber: cooldown_timers=0 (lastmove starts at 0, separate from spawn)
        if sd.sprite_class in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            init_spawn_timer = cd
        else:
            init_spawn_timer = 0
        init_timer = 0
        # Moving NPCs get is_first_tick=True (GVGAI isFirstTick blocks passiveMovement)
        init_first_tick = (sc_def is not None and sc_def.is_moving_npc)
        # RandomNPC cons initialization: GVGAI starts with prevAction=DNONE,
        # counter=0 (before cons is parsed). First `cons` calls to getRandomMove()
        # return DNONE (no movement). Match by setting direction_ticks=cons and
        # orientation=(0,0) so the "keep" branch produces zero delta.
        init_dir_ticks = sd.cons if sd.sprite_class == SpriteClass.RANDOM_NPC and sd.cons > 0 else 0
        init_ori = ([0.0, 0.0] if sd.sprite_class == SpriteClass.RANDOM_NPC and sd.cons > 0
                    else sd.orientation)
        state = state.replace(
            positions=state.positions.at[type_idx, slot].set(
                jnp.array([row * block_size, col * block_size], dtype=jnp.int32)),
            alive=state.alive.at[type_idx, slot].set(True),
            orientations=state.orientations.at[type_idx, slot].set(
                jnp.array(init_ori, dtype=jnp.float32)),
            speeds=state.speeds.at[type_idx, slot].set(jnp.int32(speed_px)),
            cooldown_timers=state.cooldown_timers.at[type_idx, slot].set(
                jnp.int32(init_timer)),
            spawn_timers=state.spawn_timers.at[type_idx, slot].set(
                jnp.int32(init_spawn_timer)),
            is_first_tick=state.is_first_tick.at[type_idx, slot].set(init_first_tick),
            direction_ticks=state.direction_ticks.at[type_idx, slot].set(
                jnp.int32(init_dir_ticks)),
        )
        slot_counts[type_idx] += 1
    state = state.replace(static_grids=jnp.array(static_grid_data))

    # Randomize orientations for RandomMissile sprites
    rng = jax.random.PRNGKey(0)
    for sd in game_def.sprites:
        if sd.sprite_class == SpriteClass.RANDOM_MISSILE:
            rng, key = jax.random.split(rng)
            n_placed = slot_counts.get(sd.type_idx, 0)
            if n_placed > 0:
                dir_indices = jax.random.randint(key, (n_placed,), 0, N_DIRECTIONS)
                state = state.replace(
                    orientations=state.orientations.at[
                        sd.type_idx, :n_placed].set(DIRECTION_DELTAS[dir_indices]))

    all_avatar_types = _find_all_avatar_types(game_def)
    avatar_config, n_move = _build_avatar_config(
        avatar_sd, game_def, block_size,
        avatar_type_indices=all_avatar_types,
        resource_name_to_idx=resource_name_to_idx)

    # Build GVGAI action map
    sc_def = SPRITE_REGISTRY[avatar_sd.sprite_class]
    action_map = _build_action_map(sc_def, avatar_config)
    action_names = sc_def.gvgai_actions
    n_actions = len(action_names)
    noop_action = list(action_names).index('ACTION_NIL')

    sprite_configs = _build_sprite_configs(game_def, block_size)
    compiled_effects = _build_compiled_effects(
        game_def, static_type_set, static_grid_map, type_max_n,
        resource_name_to_idx, resource_limits, avatar_sd.type_idx, block_size)
    compiled_terminations = _build_compiled_terminations(
        game_def, static_type_set, static_grid_map,
        resource_name_to_idx, avatar_sd.type_idx)

    # Analyze chaser targets for distance field caching
    chaser_target_set = set()
    for i, cfg in enumerate(sprite_configs):
        if cfg.sprite_class in (SpriteClass.CHASER, SpriteClass.FLEEING):
            chaser_target_set.add(cfg.target_type_idx)

    static_distance_fields = {}
    for target_idx in chaser_target_set:
        if _is_chaser_target_stable(target_idx, game_def, sprite_configs,
                                     compiled_effects, avatar_sd.type_idx):
            from vgdl_jax.sprites import _manhattan_distance_field
            if target_idx in static_type_set:
                # Static types have positions in static_grids, not position arrays.
                # Reconstruct position/alive arrays from the grid.
                sg_idx = static_grid_map[target_idx]
                grid = static_grid_data[sg_idx]  # [H, W] bool
                coords = np.argwhere(grid)  # [N, 2] (row, col)
                n_targets = len(coords)
                target_pos = np.zeros((max(n_targets, 1), 2), dtype=np.int32)
                target_alive = np.zeros(max(n_targets, 1), dtype=bool)
                for k, (gr, gc) in enumerate(coords):
                    target_pos[k] = [gr * block_size, gc * block_size]
                    target_alive[k] = True
                target_pos_jnp = jnp.array(target_pos)
                target_alive_jnp = jnp.array(target_alive)
            else:
                target_pos_jnp = state.positions[target_idx]
                target_alive_jnp = state.alive[target_idx]
            static_distance_fields[target_idx] = _manhattan_distance_field(
                target_pos_jnp, target_alive_jnp,
                height, width, block_size)

    # Compile-time validation
    if avatar_config.can_shoot and avatar_config.projectile_type_idx < 0:
        warnings.warn(f"Avatar can_shoot=True but projectile type not resolved "
                       f"(spawner_stype={avatar_sd.spawner_stype!r})")
    for cfg in sprite_configs:
        sc = cfg.sprite_class
        if sc in (SpriteClass.CHASER, SpriteClass.FLEEING):
            if cfg.target_type_idx == 0 and not game_def.sprites[0].sprite_class in AVATAR_CLASSES:
                warnings.warn(f"Chaser/Fleeing target defaulted to type 0")
        if sc in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            if cfg.target_type_idx == 0:
                warnings.warn(f"SpawnPoint/Bomber target defaulted to type 0")

    # Build step function
    params = dict(n_types=n_types, max_n=max_n, height=height, width=width,
                  n_resource_types=max(n_resource_types, 1),
                  resource_limits=resource_limits,
                  block_size=block_size)
    step_fn = build_step_fn(compiled_effects, compiled_terminations,
                            sprite_configs, avatar_config, params,
                            chaser_target_set=frozenset(chaser_target_set),
                            static_distance_fields=static_distance_fields,
                            action_map=action_map,
                            static_grid_map=static_grid_map)

    return CompiledGame(
        init_state=state, step_fn=step_fn, n_actions=n_actions,
        noop_action=noop_action, game_def=game_def,
        static_grid_map=static_grid_map, action_names=action_names)



def _make_sprite_counter(dyn_idx, sg_idx, limit, win):
    def check(s):
        return check_sprite_counter(s, dyn_idx, limit, win, sg_idx)
    return check


def _make_multi_sprite_counter(dyn_indices_list, sg_indices_list, limit, win):
    def check(s):
        return check_multi_sprite_counter(s, dyn_indices_list, limit, win, sg_indices_list)
    return check


def _make_resource_counter(avatar_type_idx, r_idx, limit, win):
    def check(s):
        return check_resource_counter(s, avatar_type_idx, r_idx, limit, win)
    return check


def _make_timeout(limit, win):
    def check(s):
        return check_timeout(s, limit, win)
    return check



def _find_active_types(game_def, avatar_sd):
    """Find type indices that participate in game logic.

    Inert types (e.g. hidden backgrounds) are excluded so they don't inflate max_n.
    Loops until stable to handle transitive spawner chains.
    """
    active = set()

    # Avatar and its projectile
    active.add(avatar_sd.type_idx)
    if avatar_sd.spawner_stype:
        for idx in game_def.resolve_stype(avatar_sd.spawner_stype):
            active.add(idx)

    # Types referenced in effects
    for ed in game_def.effects:
        for idx in game_def.resolve_stype(ed.actor_stype):
            active.add(idx)
        if ed.actee_stype != 'EOS':
            for idx in game_def.resolve_stype(ed.actee_stype):
                active.add(idx)

    # Types referenced in terminations
    for td in game_def.terminations:
        if td.term_type == TerminationType.SPRITE_COUNTER:
            for idx in game_def.resolve_stype(td.kwargs.get('stype', '')):
                active.add(idx)
        elif td.term_type == TerminationType.MULTI_SPRITE_COUNTER:
            for stype in td.kwargs.get('stypes', []):
                for idx in game_def.resolve_stype(stype):
                    active.add(idx)

    # Spawner targets of active types — loop until stable
    prev_size = 0
    while len(active) != prev_size:
        prev_size = len(active)
        for sd in game_def.sprites:
            if sd.type_idx in active and sd.spawner_stype:
                for idx in game_def.resolve_stype(sd.spawner_stype):
                    active.add(idx)

    return active
