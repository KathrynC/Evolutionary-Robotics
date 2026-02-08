"""constants.py

Central configuration for the Evolutionary-Robotics simulation.

How these constants are used:
    - Simulation length and timing:
        SIM_STEPS : number of physics steps per episode
        DT        : physics timestep (seconds); passed to PyBullet via p.setTimeStep
    - Motor control:
        MAX_FORCE     : motor force limit (Newtons)
        GAIT_*        : default sine gait parameters (used by motor.py; may be overridden by env)
        MOTOR_FREQ_HZ : legacy/alternate base frequency knob
    - Targets / demos:
        RANDOM_TARGETS, RNG_SEED, TARGET_RANGE, SINE_CYCLES : build a target-angle time series
        KICK_*        : optional external force "kick" window for debugging
    - World/robot physics:
        GRAVITY_Z, ROBOT_FRICTION, PLANE_FRICTION
    - Demo / filming knobs:
        DEMO_SLEEP_TIME : slow down stepping for visibility (may override SLEEP_TIME in some modes)
        DEMO_STRETCH    : sample motor targets more slowly (bigger = slower apparent motion)
        DEMO_AMP_MULT   : exaggerate amplitude for visibility (bigger = bigger swings)
    - GUI camera:
        CAMERA_* : follow-camera defaults for PyBullet GUI

Precedence:
    Many scripts also allow environment variable overrides (e.g., MAX_FORCE, SLEEP_TIME,
    GAIT_FREQ_HZ, GAIT_AMPLITUDE). Treat these constants as sensible defaults.
"""

# --- Simulation length / logging cadence ---
SIM_STEPS = 4000
PRINT_EVERY = 10

# --- Motor authority / target generation ---
MAX_FORCE = 150.0     # motor force limit (Newtons)
RANDOM_TARGETS = False

# A legacy "sleep" knob. Some code paths may instead use DEMO_SLEEP_TIME for filming.
SLEEP_TIME = 1/60

# --- Timing / target-angle series (from earlier refactor) ---
DT = 1/240  # physics timestep in seconds
RNG_SEED = 0
TARGET_RANGE = 1.5707963267948966  # pi/2
SINE_CYCLES = 3
KICK_START = 200
KICK_END = 350

# --- Physics parameters / gait defaults ---
GRAVITY_Z = -9.8
MOTOR_FREQ_HZ = 1.0     # legacy/alternate base frequency knob
GAIT_AMPLITUDE = 0.55
GAIT_FREQ_HZ = 1.3
BACK_OFFSET = -0.20
FRONT_OFFSET = 0.15
ROBOT_FRICTION = 2.5
PLANE_FRICTION = 1.5

# --- Demo / recording visibility knobs (safe to tweak) ---
DEMO_SLEEP_TIME = 1/15  # slower for human eyes
DEMO_STRETCH = 6        # sample motor targets slower (bigger = slower apparent motion)
DEMO_AMP_MULT = 3.0     # exaggerate swing size (try 2.0 if too wild)

# --- Camera follow (PyBullet GUI) ---
CAMERA_FOLLOW = True
CAMERA_DISTANCE = 3.0
CAMERA_YAW = 60.0
CAMERA_PITCH = -25.0
