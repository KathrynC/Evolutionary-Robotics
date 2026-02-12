# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synapse Gait Zoo: a catalog of 116 discovered gaits for a 3-link PyBullet robot, each defined by fixed neural network weights producing distinct locomotion behaviors. The robot has 3 rigid links (Torso, BackLeg, FrontLeg), 2 hinge joints (Torso_BackLeg, Torso_FrontLeg), 3 touch sensors, and 2 motors. Simulations are fully deterministic (PyBullet DIRECT mode, zero variance across trials).

## Environment Setup

```bash
conda activate er
# Environment defined in environment.yml (Python 3.11, pybullet 3.25, numpy, matplotlib, fastapi, uvicorn)
```

## Key Commands

```bash
# Generate robot body and brain files (must run before first simulation)
python3 generate.py          # produces body.urdf, brain.nndf, world.sdf

# Run a simulation (GUI by default)
python3 simulate.py

# Run headless
HEADLESS=1 python3 simulate.py

# Run with telemetry
HEADLESS=1 TELEMETRY=1 TELEMETRY_VARIANT_ID=my_gait TELEMETRY_RUN_ID=run0 python3 simulate.py

# Record videos for configured gaits (offscreen render via ffmpeg)
python3 record_videos.py

# Random-search gait optimizer (independent of NN pipeline)
python3 optimize_gait.py --trials 800 --seconds 10 --seed 2

# Batch-run zoo variants with telemetry
python3 tools/zoo/run_zoo.py --variants_dir artifacts/rules/zoo --headless 1

# Plot sensor traces
python3 analyze.py
```

## Architecture

### Simulation Pipeline

`simulate.py` instantiates `SIMULATION` which owns the full episode lifecycle:

1. **SIMULATION.__init__** (`simulation.py`): Connects PyBullet (GUI/DIRECT via `HEADLESS` env var), sets gravity/timestep, creates WORLD and ROBOT
2. **WORLD** (`world.py`): Loads `plane.urdf` ground plane; optionally loads `world.sdf` if it contains a `<world>` element
3. **ROBOT** (`robot.py`): Loads `body.urdf` + `brain.nndf`, calls `pyrosim.Prepare_To_Simulate()`, creates SENSOR and MOTOR objects for each link/joint
4. **Per-step loop**: `Sense(t)` -> `Think()` (NN update) -> `Act(t)` (motor commands) -> `p.stepSimulation()`

### Two Control Paths

- **Neural network path** (default): `brain.nndf` defines sensor/motor/hidden neurons and weighted synapses. ROBOT.Act() reads motor neuron outputs and sends position targets to joints. Synapse weights in `synapse_gait_zoo.json` define all 116 gaits.
- **GAIT_MODE direct drive**: When `GAIT_MODE=1` + `GAIT_VARIANT_PATH=<json>`, simulation.py bypasses the NN pipeline and drives joints directly with sine waves parameterized from the variant JSON.

### Neural Network Topologies

- **Standard 6-synapse**: 3 sensors (neurons 0-2) -> 2 motors (neurons 3-4), 6 weights
- **Crosswired 10-synapse**: Standard 6 + motor-to-motor connections (w34, w43, w33, w44). Cross-wiring enables CPG oscillation and crab walking.
- **Hidden layer**: Arbitrary topology with hidden neurons (e.g., half-center oscillator)

Synapse weight naming convention: `wXY` means source neuron X -> target neuron Y (e.g., `w03` = sensor 0 -> motor 3, `w34` = motor 3 -> motor 4, `w33` = motor 3 self-feedback).

### pyrosim Submodule

`pyrosim/` is an external library (git submodule) providing URDF/SDF generation, neural network loading/updating, and PyBullet joint/sensor helpers. Key interfaces:
- `pyrosim.Prepare_To_Simulate(robotId)` — populates `linkNamesToIndices` and `jointNamesToIndices`
- `pyrosim.Set_Motor_For_Joint(bodyIndex, jointName, controlMode, targetPosition, maxForce)` — note: jointName may be bytes or str
- `NEURAL_NETWORK("brain.nndf")` — loads/updates the neural network; motor neurons expose `Get_Value()` and `Get_Joint_Name()`

### Telemetry System

`tools/telemetry/logger.py` records per-step JSONL data (position, orientation, contacts, joint states) sampled every N steps. Produces `telemetry.jsonl` + `summary.json` per run. Controlled by env vars: `TELEMETRY=1`, `TELEMETRY_EVERY`, `TELEMETRY_OUT`, `TELEMETRY_VARIANT_ID`, `TELEMETRY_RUN_ID`.

### Key Data Files

- `synapse_gait_zoo.json` — all 116 gaits with weights, measurements, and category metadata
- `artifacts/gait_taxonomy.json` — structural motifs, behavioral tags, per-gait feature vectors
- `artifacts/telemetry/` — per-step telemetry for all gaits (400 JSONL records + summary each)
- `body.urdf` / `brain.nndf` — robot body and current neural network (brain.nndf is overwritten by scripts like `record_videos.py`)
- `constants.py` — central config: SIM_STEPS=4000, DT=1/240, MAX_FORCE=150, gravity, friction, gait defaults

## Important Environment Variables

| Variable | Purpose |
|---|---|
| `HEADLESS` | `1` for DIRECT mode (no GUI) |
| `GAIT_VARIANT_PATH` | Path to variant JSON for parameterized gaits |
| `GAIT_MODE` | `1` to bypass NN and drive joints directly from variant |
| `SIM_STEPS` | Override simulation length (default 4000) |
| `MAX_FORCE` | Motor force limit override (default 150N) |
| `USE_NN` | `0` to skip neural network motor control |
| `SIM_DEBUG` | `1` for debug prints at startup |

## Conventions

- Joint names are exact strings: `"Torso_BackLeg"`, `"Torso_FrontLeg"`. pyrosim may use bytes (`b"Torso_BackLeg"`); code handles both.
- `brain.nndf` is a shared file overwritten by multiple scripts. `record_videos.py` backs it up and restores it.
- Simulations are deterministic — identical weights produce identical trajectories. Float64 precision matters for evolved gaits (rounding to 6 decimal places can shift behavior by 30%).
- Units: angles in radians, time in seconds, frequency in Hz, force in Newtons.
