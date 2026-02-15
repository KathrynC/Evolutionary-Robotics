# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synapse Gait Zoo: a catalog of 116 discovered gaits for a 3-link PyBullet robot, each defined by fixed neural network weights producing distinct locomotion behaviors. The robot has 3 rigid links (Torso, BackLeg, FrontLeg), 2 hinge joints (Torso_BackLeg, Torso_FrontLeg), 3 touch sensors, and 2 motors. Simulations are fully deterministic (PyBullet DIRECT mode, zero variance across trials).

## Environment Setup

```bash
conda activate er
# Environment defined in environment.yml (Python 3.11, pybullet 3.25, numpy 1.26.4, matplotlib, fastapi, uvicorn)

# LLM-mediated experiments (structured_random_*.py, fisher_metric.py, yoneda_crosswired.py) require:
# Ollama running locally (default http://localhost:11434) with a model like qwen3-coder:30b
```

### Local LLMs via Ollama

Ollama is running on this machine and available for Claude Code to outsource work to. Use the REST API at `http://localhost:11434`. Available models:

| Model | Size | Notes |
|---|---|---|
| `qwen3-coder:30b` | 18 GB | Primary model used by structured_random_*.py experiments |
| `deepseek-r1:8b` | 5.2 GB | Reasoning model, good for chain-of-thought tasks |
| `gpt-oss:20b` | 13 GB | General-purpose |
| `llama3.1:latest` | 4.9 GB | Lightweight general-purpose |

Quick usage from shell:
```bash
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3-coder:30b","prompt":"your prompt here","stream":false}' | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])"
```

From Python (pattern used throughout the codebase, see `structured_random_common.py:ask_ollama()`):
```python
import urllib.request, json
resp = urllib.request.urlopen(urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps({"model": "qwen3-coder:30b", "prompt": "...", "stream": False}).encode(),
    headers={"Content-Type": "application/json"}
))
result = json.loads(resp.read())["response"]
```

These local LLMs can be used for brainstorming, code generation drafts, data analysis narration, weight generation experiments, or any task where offloading to a local model is faster or more appropriate than doing it inline.

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

# Generate full-resolution telemetry for all 116 zoo gaits (4000 records each @ 240 Hz)
python3 generate_telemetry.py              # all gaits
python3 generate_telemetry.py --gait 18_curie  # single gait
python3 generate_telemetry.py --dry-run    # preview without running

# Compute Beer-framework analytics from telemetry → synapse_gait_zoo_v2.json
python3 compute_beer_analytics.py

# Record videos for configured gaits (offscreen render via ffmpeg)
python3 record_videos.py

# Random-search gait optimizer (independent of NN pipeline)
python3 optimize_gait.py --trials 800 --seconds 10 --seed 2

# Batch-run zoo variants with telemetry
python3 tools/zoo/run_zoo.py --variants_dir artifacts/rules/zoo --headless 1

# Plot sensor traces
python3 analyze.py

# Launch Twine interactive story server
cd twine && uvicorn server:app --reload
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

`generate_telemetry.py` is the batch runner that produces full-resolution telemetry (every=1, 4000 records per gait at 240 Hz) for all 116 zoo gaits. It writes brain.nndf per gait, runs a headless simulation, and saves to `artifacts/telemetry/<gait_name>/`. Backs up and restores brain.nndf.

### Beer-Framework Analytics Pipeline

`compute_beer_analytics.py` reads all 116 telemetry JSONL files and computes 4 pillars of metrics per gait:

1. **Outcome** — displacement (dx, dy), yaw, speed stats, work proxy, distance-per-work efficiency
2. **Contact** — per-link duty fractions, 3-bit contact state distribution (8 states), Shannon entropy, 8×8 transition matrix
3. **Coordination** — FFT-based dominant frequency/amplitude per joint, Hilbert-transform phase difference, phase-lock score
4. **Rotation axis** — PCA of angular velocity covariance (axis dominance), axis switching rate, per-axis periodicity

Output: `synapse_gait_zoo_v2.json` — preserves all v1 gait fields but replaces `telemetry` with a comprehensive `analytics` object. Constraint: numpy-only (no scipy/sklearn). Uses FFT-based Hilbert transform instead of `scipy.signal.hilbert`.

### Research Campaign Scripts

High-budget simulation campaigns (hundreds to thousands of sims each) that investigate the weight-space landscape and behavioral dynamics:

- **walker_competition.py** — 5 optimization algorithms (Hill Climber, Ridge Walker, Cliff Mapper, Novelty Seeker, Ensemble Explorer) compete with 1,000-evaluation budget each
- **causal_surgery.py** — Mid-simulation brain transplants (weight switching at specific timesteps), single-synapse ablation, rescue experiments (~600 sims)
- **behavioral_embryology.py** — Tracks gait emergence during the first 500+ steps: when does locomotion start? When do coordination metrics stabilize?
- **gait_interpolation.py** — Linear interpolation in 6D weight space between champion pairs; maps fitness landscape smoothness and intermediate super-gaits
- **resonance_mapping.py** — Bypasses NN entirely, drives joints with sinusoidal sweeps across frequency/phase/amplitude to find the body's mechanical transfer function (~2,150 sims)
- **atlas_cliffiness.py** — Spatial atlas of cliffiness: gradient reconstruction from 500 base points, 2D slice heatmaps, cliff anatomy profiles (~6,400 sims)
- **cliff_taxonomy.py** — Adaptive probing of top 50 cliffiest points with gradient/perpendicular profiles and multi-scale classification
- **analyze_dark_matter.py** — Studies "dead" gaits (|DX| < 1m) from 500 random trials; classifies as spinners, rockers, vibrators, circlers, or inert
- **random_search_500.py** / **random_search_cliffs.py** — Large-scale random weight sampling and cliff detection

### Categorical Structure & Formal Validation Scripts

Scripts that empirically validate the categorical structure of the Sem→Wt→Beh pipeline (LLM-generated weights → robot behavior):

- **categorical_structure.py** — Core validation: functor F (Sem→Wt), map G (Wt→Beh), composition G∘F, sheaf structure, information geometry. Produces 8 figures + JSON results. Pure computation on existing data (~5 sec).
- **structured_random_compare.py** — Statistical comparison of LLM conditions (verbs, theorems, bible, places) vs baseline. Mann-Whitney U tests, PCA, behavioral clustering.
- **structured_random_common.py** — Shared utilities: `ask_ollama()` (Ollama REST API), `run_trial_inmemory()` (headless PyBullet sim → Beer analytics), `write_brain()`, `compute_all()`.
- **structured_random_{verbs,theorems,bible,places,baseline}.py** — Per-condition LLM weight generation scripts (100 trials each via Ollama).
- **fisher_metric.py** — Calls Ollama 10× per seed to measure LLM output variance. Builds statistical manifold. Interruptible (saves after each seed). Requires Ollama running locally.
- **perturbation_probing.py** — Directly measures cliffiness at LLM-generated weight vectors using 6-direction perturbation protocol (~259 PyBullet sims). Compares to atlas-interpolated values.
- **yoneda_crosswired.py** — Tests whether 10-synapse (crosswired) topology increases functor faithfulness for collapsed seed clusters. Calls Ollama for motor-to-motor weights.
- **hilbert_formalization.py** — Hilbert space analysis: L² Gram matrix of 121 zoo gait trajectories, RKHS kernel regression of cliffiness, behavioral spectral gaps. Pure computation.
- **llm_seeded_evolution.py** — Tests whether LLM weights are a launchpad or trap for evolution. Hill climber from LLM seeds vs random seeds, 500 evals each.

These scripts produce JSON artifacts in `artifacts/` and matplotlib visualizations. Each is self-contained (defines its own simulation harness, typically using headless PyBullet).

### Twine Interactive Server

`twine/server.py` is a FastAPI bridge between a browser-based SugarCube story (`twine/experiment.html`) and the PyBullet simulation. Users commit gait parameters through 36 interpretive personas/lenses. The server writes variant JSON, spawns a headless simulation, reads the telemetry summary, and returns results for narrative interpretation.

- `GET /` serves the interactive story
- `POST /simulate` accepts gait parameters, runs a sim, returns the summary

### Tools Directory

- `tools/telemetry/logger.py` — Core telemetry logger (JSONL per-step recording)
- `tools/gait_zoo.py` — Utilities for enumerating/expanding gait variants, rotating bins, swapping motors
- `tools/replay_trace.py` — Load and inspect telemetry traces (neuron values, motor outputs)
- `tools/zoo/run_zoo.py` — Batch runner for gait variants with summary metric collection
- `tools/zoo/collect_summaries.py` / `regen_scores.py` — Aggregate and regenerate gait scores

### Key Data Files

- `synapse_gait_zoo.json` — v1 zoo: all 116 gaits with weights, measurements, category metadata, and telemetry summaries
- `synapse_gait_zoo_v2.json` — v2 zoo: same gait data but with Beer-framework `analytics` object replacing `telemetry` per gait
- `artifacts/gait_taxonomy.json` — structural motifs, behavioral tags, per-gait feature vectors
- `artifacts/telemetry/<gait_name>/telemetry.jsonl` — full-resolution telemetry (4000 records at 240 Hz per gait, all 116 gaits)
- `artifacts/categorical_structure_results.json` — formal categorical validation: functor F, map G, composition, sheaf, info geometry
- `artifacts/fisher_metric_results.json` — LLM output variance: 30 seeds × 10 repeats, covariance matrices
- `artifacts/perturbation_probing_results.json` — directly measured cliffiness at 37 LLM weight vectors
- `artifacts/yoneda_crosswired_results.json` — 10-synapse faithfulness test across 4 collapsed clusters
- `artifacts/hilbert_formalization_results.json` — L² Gram matrix, RKHS regression, spectral analysis
- `artifacts/llm_seeded_evolution_results.json` — evolution from LLM seeds vs random
- `artifacts/structured_random_{verbs,theorems,bible,places,baseline}.json` — 495 trials with weights + behavioral metrics
- `artifacts/atlas_cliffiness.json` — 500 probe points with cliffiness/gradient data
- `artifacts/cliff_taxonomy.json` — 50 cliff profiles with type classification
- `body.urdf` / `brain.nndf` — robot body and current neural network (brain.nndf is overwritten by scripts like `record_videos.py` and `generate_telemetry.py`)
- `constants.py` — central config: SIM_STEPS=4000, DT=1/240, MAX_FORCE=150, gravity, friction, gait defaults
- `artifacts/unified_framework_synthesis.md` — unified categorical framework connecting all 3 projects (Spot a Cat, Gait Zoo, AI Seances)
- `CONTINUATION_PLAN.md` — comprehensive research continuation plan with Parts A-F, all results tables

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
| `TELEMETRY` | `1` to enable per-step telemetry recording |
| `TELEMETRY_EVERY` | Sampling interval for telemetry (default 10; use 1 for full-resolution) |
| `TELEMETRY_OUT` | Base output directory for telemetry (default `artifacts/telemetry`) |
| `TELEMETRY_VARIANT_ID` | Gait name used in telemetry output path |
| `TELEMETRY_RUN_ID` | Run identifier used in telemetry output path |

## Conventions

- Joint names are exact strings: `"Torso_BackLeg"`, `"Torso_FrontLeg"`. pyrosim may use bytes (`b"Torso_BackLeg"`); code handles both.
- `brain.nndf` is a shared file overwritten by multiple scripts. `record_videos.py` and `generate_telemetry.py` back it up and restore it.
- Simulations are deterministic — identical weights produce identical trajectories. Float64 precision matters for evolved gaits (rounding to 6 decimal places can shift behavior by 30%).
- Units: angles in radians, time in seconds, frequency in Hz, force in Newtons.
- **numpy compatibility**: environment.yml pins numpy 1.26.4; some analysis scripts use `np.trapezoid` (numpy 2.x). Check installed version if you encounter `np.trapz` vs `np.trapezoid` errors.
- Analytics pipeline is numpy-only by design (no scipy, no sklearn). Signal processing (Hilbert transform, FFT) is implemented from scratch via `np.fft`.
- Research campaign scripts are self-contained — each defines its own simulation harness and writes outputs to `artifacts/`. They can take minutes to run (hundreds/thousands of headless sims).
