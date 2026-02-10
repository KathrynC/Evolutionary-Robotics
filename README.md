# Synapse Gait Zoo

A catalog of 59 discovered gaits for a 3-link PyBullet robot, organized by weight motif, attractor dynamics, and behavioral class. Each gait is a fixed-weight neural network (no learning at runtime) that produces a distinct locomotion style from the same 3-link body.

## The Robot

```
[FrontLeg]---[Torso]---[BackLeg]
```

- 3 rigid links connected by 2 hinge joints
- 3 touch sensors (one per link), 2 motors (one per joint)
- Neural network maps sensor values to motor commands through weighted synapses
- Simulated in PyBullet (DIRECT mode, deterministic — zero variance across trials)

## The Zoo

**59 gaits across 11 categories, 12 weight motifs, 4 attractor types, 3 leaderboards.**

All gaits and their weights are stored in `synapse_gait_zoo.json`. Videos for every gait are in `videos/`. Per-step telemetry (400 records/gait) is in `artifacts/telemetry_full/`.

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

## Attractor Taxonomy

Every gait was instrumented with per-step telemetry (position, orientation, ground contacts, joint position/velocity/torque at every 10th sim step). From joint phase portraits, speed variability, and tilt time series, each gait is classified into one of four dynamical attractor types:

| Type | Count | Character |
|---|---|---|
| **fixed_point** | 1 | Converges to stationary equilibrium. The bouncer: perfectly still, zero displacement, zero tilt. |
| **limit_cycle** | 15 | Stable periodic orbit. Consistent stride, low speed variability (CV < 0.5). The most reliable walkers: CPG champion, curie, noether_cpg, carry_trade, pelton, rubato, etc. |
| **complex** | 37 | Moving but not strictly periodic. Drifting phase, variable speed, or multi-frequency dynamics. Stable but irregular. Most gaits live here. |
| **chaotic/fallen** | 6 | Tips over (tilt > 60). Unbounded phase portrait. Tesla, lamarr, gamma_squeeze, original, gallop, blues_shuffle. |

## Telemetry

Per-step telemetry captures what endpoint measurements miss. Every gait has 400 records (one per 10 sim steps across 4000 total steps), each containing:

- **Base position** (x, y, z) — full trajectory, not just start/end
- **Orientation** (roll, pitch, yaw) — stability time series
- **Ground contacts** — when feet touch, revealing gait cycle
- **Joint states** (position, velocity, torque) — motor dynamics and energetics

### Telemetry-Derived Metrics (per gait)

| Metric | Description | Range across zoo |
|---|---|---|
| `stride_freq_hz` | Zero-crossings of joint velocity / sim duration | 0 (bouncer) to 8.2 Hz |
| `duty_cycle` | Fraction of time with ground contact | 0.002 (dymaxion) to 1.0 (bouncer) |
| `mean_torque` | Average absolute joint torque (energy proxy) | 0.1 (bouncer) to 107.3 (mean_reversion) |
| `transient_frac` | Fraction of sim before reaching 80% of final displacement | 0.015 (lorenz_B) to 1.0 (bouncer) |
| `mean_speed` | Average instantaneous speed (m/s) | 0.0 (bouncer) to 3.33 (CPG champion) |
| `speed_cv` | Speed coefficient of variation (lower = smoother) | 0.0 (bouncer) to 1.81 (original) |
| `attractor_type` | Dynamical classification | fixed_point / limit_cycle / complex / chaotic_fallen |

### Visualization Panels (in `artifacts/plots/`)

- **trajectories.png** — (x, y) path for all 59 gaits, color-coded by time. Shows straight walkers, spirals, diagonals, loops, and the bouncer's single dot.
- **phase_portraits.png** — Joint position vs velocity for both joints. Limit cycles show clean ellipses; chaotic fallers show unbounded spirals; the bouncer is a point.
- **stability_contacts.png** — Tilt (red) and ground contact count (blue) over time. Shows exactly when fallers tip, and how stable gaits maintain low tilt.
- **speed_profiles.png** — Instantaneous speed over time. The CPG champion has remarkably constant high speed. Limit cycles show rhythmic oscillations.
- **torque_profiles.png** — Joint torque over time, showing energetic cost and motor rhythm.

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

**Limit cycles are the best walkers.** The 15 limit-cycle gaits include all top-5 displacement leaders. Clean periodic orbits in joint phase space correlate with high, consistent speed and long displacement. The CPG champion (limit cycle, mean_speed=3.33 m/s, speed_cv=0.42) is both the fastest and one of the smoothest.

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
| `synapse_gait_zoo.json` | Complete catalog: 59 gaits, weights, measurements, motifs, attractors, telemetry metrics |
| `record_videos.py` | Video recording infrastructure (offscreen render to ffmpeg) |
| `simulation.py` | Main simulation runner |
| `body.urdf` | Robot body definition (3-link) |
| `brain.nndf` | Current neural network weights (overwritten by scripts) |
| `constants.py` | Physics parameters (SIM_STEPS=4000, DT=1/240, gravity, friction) |
| `videos/` | 66 MP4 videos of gaits (gitignored) |
| `artifacts/telemetry_full/` | Per-step telemetry JSONL for all 59 gaits (gitignored) |
| `artifacts/plots/` | Trajectory maps, phase portraits, stability, speed, torque visualizations (gitignored) |
| `pyrosim/` | Neural network and simulation utilities (ludobots/pyrosim) |

## Related Work

Annotated references organized by which of our findings they relate to. Papers marked with **[foundational]** established the field; those marked **[closest]** are the nearest precedent to our specific contributions.

### Limit Cycles and Locomotion Dynamics

- **McGeer, "Passive Dynamic Walking" (1990)**, *Int. J. Robotics Research*, 9(2):62-82. **[foundational]** Showed that unpowered bipeds walking down a slope settle into stable limit cycles without any control. The origin of the limit-cycle-as-gait paradigm. Our finding that all top-5 displacement gaits are limit cycles is an evolutionary validation of McGeer's principle.

- **Holmes, Full, Koditschek, and Guckenheimer, "The Dynamics of Legged Locomotion: Models, Analyses, and Challenges" (2006)**, *SIAM Review*, 48(2):207-304. **[foundational]** The definitive review formalizing dynamical systems tools (Poincare maps, limit cycle stability) for legged locomotion. Provides the theoretical framework our attractor taxonomy uses.

- **Ijspeert, "Central Pattern Generators for Locomotion Control in Animals and Robots: A Review" (2008)**, *Neural Networks*, 21(4):642-653. **[foundational]** Reviews how CPGs produce limit cycle oscillations underlying locomotion. Our cross-wired gaits are effectively evolved CPGs; the time-signature gaits encode rhythmic structure directly in synapse topology.

- **Hubicki, Jones, Daley, and Hurst, "Do Limit Cycles Matter in the Long Run?" (2015)**, *IEEE ICRA*. Near-limit-cycle behaviors emerge naturally from task-optimal locomotion planning, supporting our finding that limit cycles are not just analytically convenient but functionally optimal.

- **Collins, Ruina et al., "Efficient Bipedal Robots Based on Passive-Dynamic Walkers" (2005)**, *Science*. Extended McGeer's work to show that minimal actuation added to passive limit-cycle walkers produces highly efficient locomotion.

### Small Neural Network Dynamics and Bifurcations

- **Beer, "On the Dynamics of Small Continuous-Time Recurrent Neural Networks" (1995)**, *Adaptive Behaviour*, 3(4):469-509. **[foundational]** Foundational analysis of small CTRNNs characterizing their dynamical repertoire. Our 3-sensor 2-motor network is in the same regime Beer analyzed theoretically.

- **Beer, "Parameter Space Structure of Continuous-Time Recurrent Neural Networks" (2006)**, *Neural Computation*, 18(12):3009-3051. **[closest]** Systematically computed bifurcation manifolds partitioning CTRNN parameter space into qualitatively different dynamics. Our bouncer-to-spinner bifurcation (10% weight change → 188-degree behavioral shift) is a concrete physical instance of what Beer predicted theoretically.

- **Beer, Chiel, and Gallagher, "Evolution and Analysis of Model CPGs for Walking" (1999)**, *J. Computational Neuroscience*, 7:99-147 (two-part paper). **[closest]** Analyzed populations of evolved CPGs and identified "general principles" holding despite weight variability. The closest precedent to our weight motif taxonomy, though they did not develop named motif categories or a motor balance ratio rule.

- **Beer and Gallagher, "Evolving Dynamical Neural Networks for Adaptive Behavior" (1992)**, *Adaptive Behavior*, 1(1):91-122. Showed that evolved recurrent networks generate locomotion without proprioceptive feedback — recurrence itself produces oscillation. Our cross-wired gaits demonstrate this: motor-to-motor feedback (w34, w43) creates CPG-like oscillation without hidden neurons.

### Network Motifs and Structure-Function Mapping

- **Milo, Shen-Orr, Itzkovitz, Kashtan, Chklovskii, and Alon, "Network Motifs: Simple Building Blocks of Complex Networks" (2002)**, *Science*, 298(5594):824-827. **[foundational]** Introduced network motifs — recurring structural patterns in directed graphs. Operates at the *topological* level (connectivity patterns). Our motif taxonomy operates at the *weight-value* level within a fixed topology, which is a different and more granular analysis.

- **Kashtan and Alon, "Spontaneous Evolution of Modularity and Network Motifs" (2005)**, *PNAS*, 102(39):13773-13778. Modular environments drive spontaneous evolution of network motifs. Relevant to why our weight motifs might cluster: the physics of the 3-link body imposes structure on the space of functional weight patterns.

- **Gaier and Ha, "Weight Agnostic Neural Networks" (2019)**, *NeurIPS*. Showed that network topology alone (without trained weights) can encode solutions. Complements our finding that within a fixed topology, weight *values* cluster into discrete motifs — topology and weight patterns are both channels for encoding behavior.

### Quality-Diversity and Behavioral Repertoires

- **Mouret and Clune, "Illuminating Search Spaces by Mapping Elites" (2015)**, *arXiv:1504.04909*. **[foundational]** Introduced MAP-Elites, which creates maps of high-performing solutions across behavioral space. The algorithmic counterpart to our manually-curated gait zoo.

- **Cully, Clune, Tarapore, and Mouret, "Robots That Can Adapt Like Animals" (2015)**, *Nature*, 521(7553):503-507. **[closest]** Generated ~15,000 six-legged gaits using MAP-Elites, creating a behavioral repertoire before deployment for damage recovery. The most direct precedent to our gait zoo concept, though their focus is engineering (damage adaptation) while ours is scientific (understanding what behaviors are possible and why).

- **Lehman and Stanley, "Abandoning Objectives: Evolution Through the Search for Novelty Alone" (2011)**, *Evolutionary Computation*, 19(2):189-223. Showed that searching for behavioral diversity rather than fitness discovers more solutions. Relevant to why our zoo keeps finding qualitatively new behaviors (crab walkers, walk-and-spin) rather than optimizing a single metric.

- **Sims, "Evolving Virtual Creatures" (1994)**, *SIGGRAPH*. The original work evolving diverse virtual creatures with co-evolved morphologies and neural controllers, producing a menagerie of locomotion styles. Our work restricts morphology to study the full behavioral repertoire of a single body.

### Robustness, Sensitivity, and the Reality Gap

- **Jin and Branke, "Evolutionary Optimization in Uncertain Environments — A Survey" (2005)**, *IEEE Trans. Evolutionary Computation*, 9(3):303-317. **[foundational]** Establishes that there is "usually a tradeoff between the quality and robustness of the solution." Our knife-edge vs antifragile classification may show something stronger: a strict correlation rather than a tradeoff, where *all* top performers are fragile.

- **Jakobi, Husbands, and Harvey, "Noise and the Reality Gap: The Use of Simulation in Evolutionary Robotics" (1995)**, *ECAL*. **[foundational]** The standard approach: inject noise during evolution to produce robust controllers. Our approach is the inverse — keep simulation deterministic, use post-hoc sensitivity analysis to predict real-world variability. A different lens on the same problem.

- **Jakobi, "Evolutionary Robotics and the Radical Envelope-of-Noise Hypothesis" (1997)**, *Adaptive Behavior*. Formalizes the principle that controllers evolved with sufficient noise transfer to reality. Our sensitivity analysis could identify which gaits *need* noise-robust evolution and which are naturally antifragile.

### Lateral Locomotion and Crab Walking

- **Kinsey et al., "Sideways Crab-Walking Is Faster and More Efficient Than Forward Walking for a Hexapod Robot" (2022)**, *Bioinspiration & Biomimetics*. Showed that designed crab walking can be 75% faster with 40% lower cost of transport than forward walking in hexapods. Our crab walkers emerged spontaneously from cross-wiring search rather than being designed, and in a biped rather than hexapod.

- **Nelson, Grant, Barber, and Fagg, "Fitness Functions in Evolutionary Robotics: A Survey and Analysis" (2009)**, *Robotics and Autonomous Systems*. Catalogs fitness function pitfalls. Using DX-only fitness (missing DY) is a known class of fitness design error, but it is usually discussed as a mistake to avoid rather than as a window into behavioral diversity.

### Dynamical Systems Classification

- **Strogatz, *Nonlinear Dynamics and Chaos* (1994, 2nd ed. 2015)**. **[foundational]** The textbook providing the standard attractor taxonomy (fixed points, limit cycles, tori, strange attractors) that our classification builds on.

- **Ren et al., "Experimental Study of Limit Cycle and Chaotic Controllers for the Locomotion of Centipede Robots" (2006)**, *IEEE*. **[closest]** Directly compared limit-cycle vs chaotic controllers for locomotion, finding limit cycles more stable and efficient. The closest precedent to our attractor-based gait classification, though they compared only two types rather than classifying an entire population.

### Evolved Locomotion Controllers

- **Nolfi and Floreano, *Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines* (2000)**. **[foundational]** The textbook establishing that different network topologies (feedforward, recurrent, fully connected) produce qualitatively different behavioral repertoires. Our progression from standard 6-synapse to crosswired 10-synapse to hidden-layer architectures traces this trajectory.

- **Reil and Husbands, "Evolution of Central Pattern Generators for Bipedal Walking in a Real-Time Physics Environment" (2002)**, *IEEE Trans. Evolutionary Computation*, 6(2):159-168. Evolved CPGs for bipedal walking in physics simulation. Gait transitions in their work are implicitly bifurcation phenomena, related to our bifurcation analysis.

- **Tonelli and Mouret, "Modularity and Sparsity: Evolution of Neural Net Controllers in Physically Embodied Robots" (2016)**, *Frontiers in Robotics and AI*. Studied evolved controllers in physical robots with ternary weights and found correlations between modularity, sparsity, and performance. Related to our weight motif analysis, though at a coarser resolution.
