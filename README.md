# Synapse Gait Zoo

A catalog of 59 discovered gaits for a 3-link PyBullet robot, organized by weight motif and behavioral class. Each gait is a fixed-weight neural network (no learning at runtime) that produces a distinct locomotion style from the same 3-link body.

## The Robot

```
[FrontLeg]---[Torso]---[BackLeg]
```

- 3 rigid links connected by 2 hinge joints
- 3 touch sensors (one per link), 2 motors (one per joint)
- Neural network maps sensor values to motor commands through weighted synapses
- Simulated in PyBullet (DIRECT mode, deterministic — zero variance across trials)

## The Zoo

**59 gaits across 11 categories, 12 weight motifs, 3 leaderboards.**

All gaits and their weights are stored in `synapse_gait_zoo.json`. Videos for every gait are in `videos/`.

### Categories

| Category | Gaits | Architecture | Description |
|---|---|---|---|
| persona_gaits | 20 | standard 6-synapse | Named after scientists/thinkers. The original collection. |
| cross_wired_cpg | 7 | crosswired 10-synapse | Motor-to-motor feedback creates internal central pattern generators |
| market_mathematics | 7 | crosswired 10-synapse | Weight patterns inspired by financial mathematics |
| evolved | 1 | crosswired 10-synapse | Found by evolutionary search |
| time_signatures | 7 | crosswired 10-synapse | Musical meters encoded in synapse topology |
| hidden_neurons | 1 | hidden layer | Half-center oscillator with 2 hidden neurons. All-time DX champion (+50.11) |
| spinners | 4 | various | Gaits that prioritize rotation over translation |
| homework | 4 | standard 6-synapse | Ludobots course assignments |
| pareto_walk_spin | 3 | crosswired 10-synapse | Simultaneously walk AND spin — Pareto frontier of displacement vs rotation |
| bifurcation_gaits | 1 | standard 6-synapse | Configurations at sharp phase transition boundaries |
| crab_walkers | 4 | crosswired 10-synapse | Walk more sideways (Y) than forward (X) |

### Architectures

**Standard 6-synapse**: 3 sensors to 2 motors (6 weights). The simplest topology.

**Crosswired 10-synapse**: Standard 6 + up to 4 motor-to-motor connections (w34, w43, w33, w44). Cross-wiring enables CPG oscillation, spin torque, and crab walking.

**Hidden layer**: Arbitrary topology with hidden neurons between sensors and motors. The CPG champion uses 2 hidden neurons in a half-center oscillator pattern.

## Leaderboards

### Displacement (|DX|)

| # | Gait | DX |
|---|---|---|
| 1 | 43_hidden_cpg_champion | +50.11 |
| 2 | 21_noether_cpg | -43.23 |
| 3 | 22_curie_amplified | +37.14 |
| 4 | 5_pelton | +34.70 |
| 5 | 32_carry_trade | +32.22 |

### Spin (|YAW|)

| # | Gait | YAW | Turns |
|---|---|---|---|
| 1 | 44_spinner_champion | +838 | 2.33 |
| 2 | 45_spinner_stable | -749 | 2.08 |
| 3 | 46_spinner_crosswired | -320 | 0.89 |

### Crab Walking (|DY|)

| # | Gait | DY | DX | Heading | Crab Ratio |
|---|---|---|---|---|---|
| 1 | 52_curie_crab | -28.79 | +24.44 | -50 | 1.18 |
| 2 | 53_rucker_landcrab | +27.05 | +11.46 | +67 | 2.36 |
| 3 | 35_evolved_curie | -17.50 | +24.28 | -36 | 0.72 |
| 4 | 10_tesla_3phase | +15.49 | -18.33 | +140 | 0.85 |
| 5 | 54_rucker_sidewinder | +15.19 | +5.70 | +69 | 2.66 |

Crab ratio = |DY|/|DX|. Values > 1.0 mean the robot walks more sideways than forward.

## Weight Motifs

12 structural motifs identified by analyzing weight patterns across all 59 gaits:

| Motif | Members | Signature |
|---|---|---|
| canonical_antisymmetric | 7 | w_i3 = -w_i4 (equal and opposite drive to each motor) |
| curie_asymmetric_drive | 7 | Torso weight differs between motors; stronger front-leg drive |
| noether_involution | 4 | Weights approximately negate under motor swap |
| same_drive_symmetric | 4 | Both motors receive similar-sign drive |
| minimal_wiring | 3 | 3-4 active synapses (most weights zero) |
| cpg_dominant | 1 | Cross-wiring dominates over sensor input |
| half_center_oscillator | 3 | Hidden neurons with mutual inhibition/excitation |
| spin_torque | 4 | Asymmetric cross-wiring creates net rotational torque |
| positive_feedback_cascade | 1 | Self-reinforcing motor feedback |
| walk_and_spin | 3 | Strong base pattern + cross-wiring for simultaneous translation and rotation |
| crab_walk | 7 | Asymmetric drive patterns amplified by cross-wiring for lateral motion |
| bifurcation_boundary | 2 | Configuration at a sharp phase transition; tiny perturbation changes behavior qualitatively |

## Key Discoveries

**All gaits are perfectly deterministic.** PyBullet DIRECT mode produces zero variance across trials (CV=0.000 for all 59 gaits). What matters is sensitivity — how much a gait's behavior changes with small weight perturbations.

**Motor balance ratio predicts direction.** The ratio of total drive to motor 3 vs motor 4 (MB = sum|w_i3| / sum|w_i4|) predicts forward vs backward walking. MB < 1.0 tends forward, MB > 1.0 tends backward.

**Cross-wiring unlocks new behaviors.** The 4 motor-to-motor weights (w34, w43, w33, w44) enable CPG oscillation, spin, and crab walking that are impossible with sensor-to-motor weights alone.

**Lateral motion was a blind dimension.** Before measuring DY, all gaits were characterized only by forward displacement. The average |DY| across the zoo is 3.75m. Many gaits walk at significant diagonal angles.

**Bifurcation boundaries are sharp.** The bouncer configuration [0, +1, -1, 0, -1, +1] is perfectly still (DX=0, YAW=0, tilt=0). Reducing one weight by 10% (w24: 1.0 to 0.9) produces a 188 spin. The sharpest behavioral cliff in the zoo.

## Sensitivity Classes

Gaits fall into three sensitivity classes based on how their behavior changes with small weight perturbations (central-difference gradient, +/-0.05):

| Class | Example | DX Sensitivity | Character |
|---|---|---|---|
| Antifragile | 19_haraway | 32 | Robust to perturbation |
| Knife-edge | 32_carry_trade | 1340 | High performance, high fragility |
| Yaw powder keg | 1_original | 17149 (yaw) | Tiny changes cause massive rotation changes |

## Setup

```bash
conda activate er
pip install pybullet
```

## Running a gait

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

## Recording videos

```bash
python record_videos.py
```

Renders all configured gaits to `videos/` using offscreen PyBullet rendering piped to ffmpeg. 66 videos currently recorded.

## Files

| File | Description |
|---|---|
| `synapse_gait_zoo.json` | Complete catalog: 59 gaits, weights, measurements, motifs, leaderboards |
| `record_videos.py` | Video recording infrastructure (offscreen render to ffmpeg) |
| `simulation.py` | Main simulation runner |
| `body.urdf` | Robot body definition (3-link) |
| `brain.nndf` | Current neural network weights (overwritten by scripts) |
| `constants.py` | Physics parameters (SIM_STEPS=4000, DT=1/240, gravity, friction) |
| `videos/` | 66 MP4 videos of gaits (gitignored) |
| `pyrosim/` | Neural network and simulation utilities (ludobots/pyrosim) |
