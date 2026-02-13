# Synapse Gait Zoo

A catalog of 116 discovered gaits for a 3-link PyBullet robot, organized by structural motif, behavioral tag, and attractor dynamics. Each gait is a fixed-weight neural network (no learning at runtime) that produces a distinct locomotion style from the same 3-link body.

## The Robot

```
[FrontLeg]---[Torso]---[BackLeg]
```

- 3 rigid links connected by 2 hinge joints
- 3 touch sensors (one per link), 2 motors (one per joint)
- Neural network maps sensor values to motor commands through weighted synapses
- Simulated in PyBullet (DIRECT mode, deterministic — zero variance across trials)

## Setup

```bash
conda env create -f environment.yml   # Python 3.11, pybullet 3.25, numpy 1.26.4, matplotlib, fastapi
conda activate er
```

## Running a Gait

Write a brain file and simulate:

```python
import json, pybullet as p, pybullet_data

zoo = json.load(open("synapse_gait_zoo.json"))
gait = zoo["categories"]["persona_gaits"]["gaits"]["18_curie"]

# Write brain.nndf from gait weights, then:
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# ... load plane.urdf, body.urdf, run simulation loop
```

See `record_videos.py` for a complete example that writes brain files and runs simulations with video capture.

## Recording Videos

```bash
python record_videos.py
```

Renders all configured gaits to `videos/` using offscreen PyBullet rendering piped to ffmpeg. 89 videos currently recorded.

## The Zoo

**116 gaits across 11 categories, 13 structural motifs, 22 behavioral tags, 112 unique motif-tag profiles.**

All gaits and their weights are stored in `synapse_gait_zoo.json` (v1) and `synapse_gait_zoo_v2.json` (v2, with Beer-framework analytics per gait). Full taxonomy (motifs, behavioral tags, per-gait features) is in `artifacts/gait_taxonomy.json`. Full-resolution telemetry (4000 records/gait at 240 Hz) is in `artifacts/telemetry/`. Videos are in `videos/`.

### Categories

| Category | Gaits | Architecture | Description |
|---|---|---|---|
| persona_gaits | 74 | standard 6-synapse / crosswired | Named after scientists/thinkers/artists. 18 thematic groups + 20 originals. |
| cross_wired_cpg | 7 | crosswired 10-synapse | Motor-to-motor feedback creates internal central pattern generators |
| market_mathematics | 7 | crosswired 10-synapse | Weight patterns inspired by financial mathematics |
| evolved | 1 | crosswired 10-synapse | Found by evolutionary search |
| time_signatures | 7 | crosswired 10-synapse | Musical meters encoded in synapse topology |
| hidden_neurons | 1 | hidden layer | Half-center oscillator with 2 hidden neurons. All-time DX champion (+50.11) |
| spinners | 4 | various | Gaits that prioritize rotation over translation |
| homework | 4 | standard 6-synapse | Ludobots course assignments |
| pareto_walk_spin | 3 | crosswired 10-synapse | Simultaneously walk AND spin — Pareto frontier of displacement vs rotation |
| bifurcation_gaits | 1 | standard 6-synapse | Configurations at sharp phase transition boundaries |
| crab_walkers | 7 | crosswired 10-synapse | Walk more sideways (Y) than forward (X). Top 3 are evolved. |

### Architectures

**Standard 6-synapse**: 3 sensors to 2 motors (6 weights). The simplest topology.

**Crosswired 10-synapse**: Standard 6 + up to 4 motor-to-motor connections (w34, w43, w33, w44). Cross-wiring enables CPG oscillation, spin torque, and crab walking.

**Hidden layer**: Arbitrary topology with hidden neurons between sensors and motors. The CPG champion uses 2 hidden neurons in a half-center oscillator pattern.

## Leaderboards

### Displacement (|DX|)

| # | Gait | DX | Category |
|---|---|---|---|
| 1 | 43_hidden_cpg_champion | +50.11 | hidden_neurons |
| 2 | 21_noether_cpg | -43.23 | cross_wired_cpg |
| 3 | 22_curie_amplified | +37.14 | cross_wired_cpg |
| 4 | 5_pelton | +34.70 | persona_gaits |
| 5 | 100_kcramer_anthology | +32.32 | persona_gaits |
| 6 | 32_carry_trade | +32.22 | market_mathematics |
| 7 | 69_grunbaum_deflation | -30.48 | persona_gaits |
| 8 | 50_noether_cyclone | -30.16 | pareto_walk_spin |
| 9 | 23_hodgkin_huxley | -29.79 | cross_wired_cpg |
| 10 | 36_take_five | -27.72 | time_signatures |

### Total Distance (displacement)

| # | Gait | Distance | DX | DY | Category |
|---|---|---|---|---|---|
| 1 | 43_hidden_cpg_champion | 50.19 | +50.11 | -2.88 | hidden_neurons |
| 2 | 21_noether_cpg | 43.81 | -43.23 | -7.12 | cross_wired_cpg |
| 3 | 56_evolved_crab_v2 | 41.19 | -6.71 | -40.64 | crab_walkers |
| 4 | 57_evolved_sidewinder | 39.52 | -3.99 | -39.32 | crab_walkers |
| 5 | 58_evolved_crab_positive_v2 | 38.10 | -1.17 | +38.08 | crab_walkers |
| 6 | 52_curie_crab | 37.76 | +24.44 | -28.79 | crab_walkers |
| 7 | 22_curie_amplified | 37.17 | +37.14 | -1.60 | cross_wired_cpg |
| 8 | 5_pelton | 35.27 | +34.70 | +6.32 | persona_gaits |
| 9 | 100_kcramer_anthology | 33.12 | +32.32 | -7.25 | persona_gaits |
| 10 | 50_noether_cyclone | 33.05 | -30.16 | -13.51 | pareto_walk_spin |

### Spin (|YAW|)

| # | Gait | YAW | Turns | Category |
|---|---|---|---|---|
| 1 | 44_spinner_champion | +838 | 2.33 | spinners |
| 2 | 45_spinner_stable | -749 | 2.08 | spinners |
| 3 | 46_spinner_crosswired | -320 | 0.89 | spinners |
| 4 | 1_original | +299 | 0.83 | homework |
| 5 | 2_flipped | +215 | 0.60 | homework |

### Crab Walking (|DY|)

| # | Gait | DY | DX | Heading | Crab Ratio | Origin |
|---|---|---|---|---|---|---|
| 1 | 56_evolved_crab_v2 | -40.64 | -6.71 | -99 | 6.06 | evolved |
| 2 | 57_evolved_sidewinder | -39.32 | -3.99 | -96 | 9.85 | evolved |
| 3 | 58_evolved_crab_positive_v2 | +38.08 | -1.17 | +92 | 32.55 | evolved |
| 4 | 52_curie_crab | -28.79 | +24.44 | -50 | 1.18 | hand-designed |
| 5 | 53_rucker_landcrab | +27.05 | +11.46 | +67 | 2.36 | hand-designed |

Crab ratio = |DY|/|DX|. Values > 1.0 mean the robot walks more sideways than forward. The top 3 are evolved solutions that require full float64 weight precision — rounding to 6 decimal places shifts them to different attractors (see knife-edge sensitivity below).

## Files

### Data and Artifacts

| File | Description |
|---|---|
| `synapse_gait_zoo.json` | v1 catalog: 116 gaits, weights, measurements, telemetry summaries across 11 categories |
| `synapse_gait_zoo_v2.json` | v2 catalog: same gaits with Beer-framework `analytics` object (4 pillars) replacing `telemetry` |
| `artifacts/gait_taxonomy.json` | Taxonomy v2.0: 13 motifs, 22 behavioral tags, per-gait feature vectors |
| `artifacts/telemetry/` | Full-resolution telemetry for all 116 gaits (4000 JSONL records + summary.json each) |
| `artifacts/discovery_dig_full116.txt` | Deep analysis output: 13 digs across the full zoo |
| `artifacts/plots/` | Trajectory maps, phase portraits, stability, speed, torque visualizations |
| `videos/` | MP4 videos of gaits |

### Core Simulation

| File | Description |
|---|---|
| `simulation.py` | Main simulation runner: PyBullet lifecycle, Sense→Think→Act loop, GAIT_MODE direct-drive bypass, telemetry hooks |
| `robot.py` | ROBOT class: loads body.urdf + brain.nndf, manages sensors/motors, executes NN control |
| `motor.py` | MOTOR class: joint control with sine-trajectory support for GAIT_MODE variants |
| `sensor.py` | SENSOR class: touch sensor reading and value recording per link |
| `world.py` | WORLD class: loads ground plane and optional SDF world elements |
| `generate.py` | Generates body.urdf, brain.nndf, and world.sdf from pyrosim primitives |
| `constants.py` | Central physics config: SIM_STEPS=4000, DT=1/240, MAX_FORCE=150, gravity, friction |
| `body.urdf` | Robot body definition (3 rigid links, 2 hinge joints) |
| `brain.nndf` | Current neural network weights (shared file, overwritten by many scripts) |
| `pyrosim/` | External submodule: URDF/SDF generation, NN loading/updating, PyBullet joint/sensor helpers |

### Analytics and Telemetry

| File | Description |
|---|---|
| `compute_beer_analytics.py` | Reads all 116 telemetry JSONL files and produces v2 zoo with 4-pillar Beer-framework metrics (numpy-only, no scipy/sklearn). Implements FFT-based Hilbert transform, 3-bit contact state encoding, Shannon entropy, Markov transition matrices, PCA via eigendecomposition. |
| `generate_telemetry.py` | Batch runner for full-resolution telemetry (every=1, 4000 records/gait at 240 Hz). Writes brain.nndf per gait, runs headless sim, saves to `artifacts/telemetry/`. |
| `record_videos.py` | Video recording infrastructure: writes brain files, runs offscreen PyBullet rendering piped to ffmpeg |

### Research Campaign Scripts

Self-contained simulation campaigns (hundreds to thousands of headless sims each) that investigate weight-space landscape and behavioral dynamics. Each defines its own simulation harness and writes outputs to `artifacts/`.

| File | Description |
|---|---|
| `causal_surgery.py` | Mid-simulation brain transplants (weight switching at specific timesteps), single-synapse ablation, and rescue experiments (~600 sims). Tests whether gaits are initial-condition-dependent or attractor-dependent. |
| `causal_surgery_interpolation.py` | Extension of causal surgery: continuous interpolation between weight vectors during simulation, blending gaits in real-time rather than discrete switching. |
| `gait_interpolation.py` | Linear interpolation in 6D weight space between champion pairs. Maps fitness landscape smoothness, discovers intermediate super-gaits, and tests whether high-performing gaits are connected by high-performing paths. |
| `behavioral_embryology.py` | Tracks gait emergence during the first 500+ steps: when does locomotion start? When do coordination metrics (phase locking, contact entropy, speed CV) stabilize? |
| `resonance_mapping.py` | Bypasses NN entirely, drives joints with sinusoidal sweeps across frequency/phase/amplitude to find the body's mechanical transfer function (~2,150 sims). Maps the body's intrinsic dynamics. |
| `walker_competition.py` | 5 optimization algorithms (Hill Climber, Ridge Walker, Cliff Mapper, Novelty Seeker, Ensemble Explorer) compete with 1,000-evaluation budget each to find high-displacement gaits. |
| `atlas_cliffiness.py` | Spatial atlas of behavioral cliffs: gradient reconstruction from 500 base points, 2D slice heatmaps, cliff anatomy profiles (~6,400 sims). Maps where in weight space behavior changes discontinuously. |
| `cliff_taxonomy.py` | Adaptive probing of top 50 cliffiest points: gradient/perpendicular profiles, multi-scale classification of cliff types (walls, ridges, cusps). |
| `cliff_taxonomy_deep.py` | Deep-dive into cliff structure: expanded perturbation radii, multi-metric cliff profiles, and cross-section visualizations for the sharpest behavioral boundaries. |
| `random_search_500.py` | Large-scale random sampling of 500 weight vectors in 6D standard-topology space. Maps the statistical distribution of gait fitness, displacement, spin, and tilt across random controllers. |
| `random_search_cliffs.py` | Pairwise cliff detection between random-search base points. Identifies which random gaits sit near behavioral discontinuities by measuring DX changes between nearby weight vectors. |
| `random_search_analytics.py` | Post-hoc analysis of random search data: fitness distributions, weight-performance correlations, cliff frequency statistics, and landscape roughness metrics. |
| `timestep_atlas.py` | Timestep Atlas: sweeps all 116 zoo gaits across 7 DT values in coupled and decoupled modes to separate physics-resolution artifacts from controller-sampling-rate artifacts. Adds DT as a 7th gaitspace dimension. |

### Analysis and Visualization

| File | Description |
|---|---|
| `analyze_super_gaits.py` | Investigates interpolation-discovered super-gaits (gaits that outperform both parents). Characterizes what makes intermediate weight vectors exceptional. |
| `analyze_trial3.py` | Detailed analysis of a specific optimization trial: fitness trajectory, weight evolution, behavioral transitions during search. |
| `analyze_dark_matter.py` | Studies "dead" gaits (|DX| < 1m) from random search: classifies as spinners, rockers, vibrators, circlers, or inert. Maps the taxonomy of failure. |
| `analyze_novelty_champion.py` | Characterizes the novelty-search champion from walker_competition.py: what behavioral dimensions make it novel, and how does novelty relate to fitness? |
| `plot_gaitspace.py` | Visualizes the structure of gait space: PCA/t-SNE embeddings of Beer-framework analytics, cluster boundaries, behavioral region maps. |
| `sweep_openloop_legal.py` | Open-loop parameter sweep across legal (constraint-satisfying) gait configurations. Maps how amplitude, frequency, and phase offset affect displacement. |
| `analyze.py` | Basic sensor trace plotter for inspecting individual simulation runs. |

### Optimization

| File | Description |
|---|---|
| `search.py` | Core evolutionary search engine: parallel evolution with parent/child comparison |
| `optimize_gait.py` | Random-search gait optimizer (independent of NN pipeline): samples sine-wave parameters directly |

### Interactive

| File | Description |
|---|---|
| `twine/server.py` | FastAPI bridge between browser-based SugarCube story and PyBullet simulation. Users commit gait parameters through 36 interpretive personas/lenses. |
| `twine/experiment.html` | Interactive Twine story (SugarCube format) for guided gait exploration |

### Tools

| File | Description |
|---|---|
| `tools/telemetry/logger.py` | Core telemetry logger: per-step JSONL recording (position, orientation, contacts, joint states) |
| `tools/gait_zoo.py` | Utilities for enumerating/expanding gait variants, rotating bins, swapping motors |
| `tools/replay_trace.py` | Load and inspect telemetry traces (neuron values, motor outputs) |
| `tools/zoo/run_zoo.py` | Batch runner for gait variants with summary metric collection |
| `tools/zoo/collect_summaries.py` | Aggregate telemetry summaries across gait variant runs |
| `tools/zoo/regen_scores.py` | Regenerate gait scores from collected summaries |

## Further Reading

- [FINDINGS.md](FINDINGS.md) — Scientific analysis and key discoveries
- [PERSONAS.md](PERSONAS.md) — The 18 persona gait themes
- [REFERENCES.md](REFERENCES.md) — Annotated bibliography

## Key References

- **Beer 1995** — Small CTRNN dynamics; our 3-sensor 2-motor network is in the same regime Beer analyzed
- **Beer 2006** — Parameter space bifurcation structure; our bouncer-to-spinner cliff is a physical instance of what Beer predicted
- **McGeer 1990** — Passive dynamic walking and limit cycles; our top-5 displacement gaits being limit cycles validates this principle
- **Ijspeert 2008** — CPG review; our hidden-layer champion is effectively a CPG
- **Cully et al. 2015** — Behavioral repertoires via MAP-Elites; the closest precedent to our zoo concept
- **Sims 1994** — Evolved virtual creatures; the original menagerie of evolved locomotion
