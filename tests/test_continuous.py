"""Integration tests for ContinuousPhysics & GravityPhysics (InertialAvatar, MarioAvatar)."""
import jax
import jax.numpy as jnp

from vgdl_jax.parser import parse_vgdl_text
from vgdl_jax.compiler import compile_game


# ── Helper: compile inline game + level ──────────────────────────────────

def _compile(game_text, level_text):
    gd = parse_vgdl_text(game_text, level_text)
    return compile_game(gd)


# ── Game definitions ─────────────────────────────────────────────────────

MARIO_GAME = """\
BasicGame
    SpriteSet
        wall > Immovable
        goal > Immovable color=GREEN
        avatar > MarioAvatar
    InteractionSet
        avatar wall > wallStop
        avatar goal > killSprite scoreChange=1
        avatar EOS > killSprite
    LevelMapping
        G > goal
    TerminationSet
        SpriteCounter stype=avatar win=False
        SpriteCounter stype=goal win=True
"""

INERTIAL_GAME = """\
BasicGame
    SpriteSet
        wall > Immovable
        goal > Immovable color=GREEN
        avatar > InertialAvatar
    InteractionSet
        avatar wall > wallStop
        avatar goal > killSprite scoreChange=1
        avatar EOS > killSprite
    LevelMapping
        G > goal
    TerminationSet
        SpriteCounter stype=avatar win=False
        SpriteCounter stype=goal win=True
"""

# Simple level: avatar on floor with walls surrounding
# 7 wide x 5 tall
MARIO_LEVEL = """\
wwwwwww
w     w
w     w
wA   Gw
wwwwwww"""

# Open space for inertial avatar (no gravity)
# G placed far from avatar so we can test movement
INERTIAL_LEVEL = """\
wwwwwww
w    Gw
w  A  w
w     w
wwwwwww"""


# ── MarioAvatar Tests ────────────────────────────────────────────────────


class TestMarioGravity:
    """MarioAvatar should stay on floor when wallStop is present."""

    def test_grounded_stays_on_floor(self):
        """Avatar standing on floor with NOOP should not fall through."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')
        initial_pos = state.positions[avatar_type, 0].copy()

        # Step with NOOP — GVGAI ordering: LEFT=0, RIGHT=1, USE=2, NIL=3
        NOOP = 3
        for _ in range(20):
            state = cg.step_fn(state, NOOP)

        final_pos = state.positions[avatar_type, 0]
        # Avatar should remain at same row (floor), give or take float precision
        assert jnp.abs(final_pos[0] - initial_pos[0]) < 0.5, (
            f"Avatar fell through floor: {initial_pos[0]} -> {final_pos[0]}")

    def test_grounded_passive_forces(self):
        """When grounded via wallStop, passive_forces[row] should be 0."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # Step a few frames for wallStop to fire and zero passive_forces
        NOOP = 3
        for _ in range(5):
            state = cg.step_fn(state, NOOP)

        pf = state.passive_forces[avatar_type, 0]
        # Row passive_force should be zeroed by wallStop (grounded indicator)
        assert pf[0] == 0.0, f"passive_forces[row] should be 0 when grounded, got {pf[0]}"

    def test_action_count(self):
        """MarioAvatar should have 4 actions: LEFT, RIGHT, USE(JUMP), NIL."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        assert cg.n_actions == 4


class TestMarioJump:
    """MarioAvatar jump trajectory: rises then falls."""

    def test_jump_rises_then_falls(self):
        """Pressing JUMP should cause avatar to rise (row decreases) then fall back."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')
        start_row = float(state.positions[avatar_type, 0, 0])

        # GVGAI ordering: LEFT=0, RIGHT=1, USE(JUMP)=2, NIL=3
        JUMP = 2
        NOOP = 3

        # Apply jump
        state = cg.step_fn(state, JUMP)

        # Track min row (highest point) over the next frames
        min_row = float(state.positions[avatar_type, 0, 0])
        for _ in range(60):
            state = cg.step_fn(state, NOOP)
            row = float(state.positions[avatar_type, 0, 0])
            min_row = min(min_row, row)

        final_row = float(state.positions[avatar_type, 0, 0])

        # Should have risen above start
        assert min_row < start_row - 0.1, (
            f"Avatar didn't rise: start={start_row}, min={min_row}")
        # Should have returned near start (landed on floor)
        assert jnp.abs(final_row - start_row) < 0.5, (
            f"Avatar didn't land: start={start_row}, final={final_row}")

    def test_horizontal_movement(self):
        """LEFT and RIGHT should move avatar horizontally."""
        cg = _compile(MARIO_GAME, MARIO_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # Let it settle — GVGAI ordering: LEFT=0, RIGHT=1, USE=2, NIL=3
        NOOP = 3
        for _ in range(3):
            state = cg.step_fn(state, NOOP)

        start_col = float(state.positions[avatar_type, 0, 1])

        # Move RIGHT (action=1) several times
        RIGHT = 1
        for _ in range(10):
            state = cg.step_fn(state, RIGHT)

        end_col = float(state.positions[avatar_type, 0, 1])
        assert end_col > start_col, (
            f"Avatar didn't move right: {start_col} -> {end_col}")


# ── InertialAvatar Tests ─────────────────────────────────────────────────


class TestInertialVelocity:
    """InertialAvatar should accumulate velocity with inertia."""

    def test_velocity_accumulates(self):
        """Applying RIGHT force repeatedly should increase velocity."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # GVGAI ordering: LEFT=0, RIGHT=1, DOWN=2, UP=3, NIL=4
        RIGHT = 1
        for _ in range(5):
            state = cg.step_fn(state, RIGHT)

        vel = state.velocities[avatar_type, 0]
        # Velocity col component should be positive (moving right)
        assert vel[1] > 0.0, f"Velocity should be positive rightward, got {vel[1]}"

    def test_drift_continues_on_noop(self):
        """After applying force, NOOP should let avatar drift (no friction in open space)."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # GVGAI ordering: LEFT=0, RIGHT=1, DOWN=2, UP=3, NIL=4
        RIGHT = 1
        NOOP = 4

        # Apply right force for a few frames
        for _ in range(3):
            state = cg.step_fn(state, RIGHT)

        pos_before_noop = float(state.positions[avatar_type, 0, 1])

        # Now NOOP — should still drift
        for _ in range(3):
            state = cg.step_fn(state, NOOP)

        pos_after_noop = float(state.positions[avatar_type, 0, 1])
        assert pos_after_noop > pos_before_noop, (
            f"Avatar should drift: {pos_before_noop} -> {pos_after_noop}")

    def test_wallstop_zeroes_velocity(self):
        """Hitting a wall via wallStop should zero velocity on that axis."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        state = cg.init_state
        avatar_type = cg.game_def.type_idx('avatar')

        # Drive hard right until hitting wall — GVGAI ordering: RIGHT=1
        RIGHT = 1
        for _ in range(50):
            state = cg.step_fn(state, RIGHT)

        # Velocity col component should be ~0 after wallStop
        vel_col = float(state.velocities[avatar_type, 0, 1])
        assert abs(vel_col) < 0.1, (
            f"Velocity should be ~0 after wallStop, got {vel_col}")

    def test_action_count(self):
        """InertialAvatar should have 5 actions: UP, DOWN, LEFT, RIGHT, NOOP."""
        cg = _compile(INERTIAL_GAME, INERTIAL_LEVEL)
        assert cg.n_actions == 5
