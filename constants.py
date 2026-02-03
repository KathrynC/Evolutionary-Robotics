SIM_STEPS = 4000
PRINT_EVERY = 10

MAX_FORCE = 150.0     # more authority to reach targets
RANDOM_TARGETS = False

SLEEP_TIME = 1/60  # tweak later for filming

# Added from simulate.py during refactor
DT = 1/240  # physics timestep
RNG_SEED = 0
TARGET_RANGE = 1.5707963267948966  # pi/2
SINE_CYCLES = 3
KICK_START = 200
KICK_END = 350

# Added during refactor
GRAVITY_Z = -9.8
MOTOR_FREQ_HZ = 1.0     # base frequency; other motor will be half
GAIT_AMPLITUDE = 0.55
GAIT_FREQ_HZ = 1.3
BACK_OFFSET = -0.20
FRONT_OFFSET = 0.15
ROBOT_FRICTION = 2.5
PLANE_FRICTION = 1.5

# --- Demo/recording visibility knobs (safe to tweak) ---
DEMO_SLEEP_TIME = 1/15  # slower for human eyes
DEMO_STRETCH = 6       # sample motor targets slower (bigger=slower)
DEMO_AMP_MULT = 3.0    # BIG swings for visibility (try 2.0 if too wild)

# Camera follow (PyBullet GUI)
CAMERA_FOLLOW = True
CAMERA_DISTANCE = 3.0
CAMERA_YAW = 60.0
CAMERA_PITCH = -25.0
