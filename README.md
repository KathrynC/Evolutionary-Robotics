# Synapse Gait Zoo

A catalog of 77 discovered gaits for a 3-link PyBullet robot, organized by weight motif, attractor dynamics, and behavioral class. Each gait is a fixed-weight neural network (no learning at runtime) that produces a distinct locomotion style from the same 3-link body.

## The Robot

```
[FrontLeg]---[Torso]---[BackLeg]
```

- 3 rigid links connected by 2 hinge joints
- 3 touch sensors (one per link), 2 motors (one per joint)
- Neural network maps sensor values to motor commands through weighted synapses
- Simulated in PyBullet (DIRECT mode, deterministic — zero variance across trials)

## The Zoo

**77 gaits across 11 categories, 12 weight motifs, 4 attractor types (with 5 complex subtypes), 3 leaderboards.**

All gaits and their weights are stored in `synapse_gait_zoo.json`. Videos for every gait are in `videos/`. Per-step telemetry (400 records/gait) is in `artifacts/telemetry_full/`.

### Categories

| Category | Gaits | Architecture | Description |
|---|---|---|---|
| persona_gaits | 35 | standard 6-synapse / crosswired | Named after scientists/thinkers. Includes Fibonacci, Cage, Womack, Grünbaum, Gallagher. |
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

### Persona Gait Themes

The 35 persona gaits include five thematic groups added after the original 20:

- **Fibonacci (3 gaits)**: Golden ratio proportions in weight structure. `fibonacci_self` uses phi^-2 self-feedback for stable diagonal walking. `fibonacci_phyllotaxis` maps the golden angle (137.5°) to successive weights — chaotic but covers 10m while spinning 595°. `fibonacci_spiral` scales weights by phi (0.382, 0.618, 1.0) for near-pure lateral motion.

- **John Cage (3 gaits)**: `cage_433` is near-silence (weights at ±0.01, no visible motion — the ambient physics IS the performance). `cage_prepared` takes the curie pattern and flips two weights (like inserting bolts into piano strings), transforming a forward walker into a crab walker (DY=15.39). `cage_iching` uses chance operations (random.seed(1952)) for all 10 weights — randomness produces the strongest backward walker in the group (DX=-19.95).

- **Jack Womack (3 gaits)**: Austerity and survival. `womack_random_acts` uses only 2 sensor synapses + self-feedback — institutional feedback loops drive backward locomotion with the lowest tilt (10°) relative to displacement in the zoo. `womack_ambient` has only 2 active synapses (the absolute minimum for locomotion). `womack_terraplane` operates at 20-40% of normal weight magnitude — existence at the poverty line.

- **Branko Grünbaum (3 gaits)**: Tilings and patterns. `grunbaum_penrose` uses sin/cos of the Penrose tile angles (36° and 72°) as weights — pure aperiodic geometry produces a 23m backward walker. `grunbaum_deflation` cascades through phi^-1 levels (1.0, 0.618, 0.382, 0.236) like a Penrose tiling deflation rule, with self-feedback at phi^-3 (31m, tilt 27°). `grunbaum_defect` starts with perfect p6m symmetry (uniform ±0.7) and breaks it with a single cross-wire — a third intentional near-fixed-point alongside bouncer and cage_433.

- **Patrick X. Gallagher (3 gaits)**: Analytic number theory. `gallagher_multiplicative` uses products of prime pairs from {2,3,5,7} — multiplicative (not additive) weight construction. `gallagher_gaps` encodes consecutive prime differences [1,2,2,4,2,4] as weights — very stable (tilt 10°). `gallagher_sieve` places weights only at prime-indexed positions (indices 2,3,5,7), zeroing composites — only 4 active synapses. The sparsity IS the sieve.

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
| 5 | 32_carry_trade | +32.22 | market_mathematics |
| 6 | 69_grunbaum_deflation | -30.48 | persona_gaits |
| 7 | 50_noether_cyclone | -30.16 | pareto_walk_spin |
| 8 | 23_hodgkin_huxley | -29.79 | cross_wired_cpg |
| 9 | 36_take_five | -27.72 | time_signatures |
| 10 | 52_curie_crab | +24.44 | crab_walkers |

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
| 9 | 50_noether_cyclone | 33.05 | -30.16 | -13.51 | pareto_walk_spin |
| 10 | 32_carry_trade | 32.29 | +32.22 | -2.07 | market_mathematics |

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

### Persona Gaits (by total displacement)

| # | Gait | Distance | DX | DY | Tilt | Notable |
|---|---|---|---|---|---|---|
| 1 | 5_pelton | 35.27 | +34.70 | +6.32 | 32° | All-time persona champion |
| 2 | 69_grunbaum_deflation | 31.27 | -30.48 | -6.99 | 27° | Penrose deflation cascade |
| 3 | 10_tesla_3phase | 24.00 | -18.33 | +15.49 | 186° | Fallen but far |
| 4 | 18_curie | 23.73 | +23.73 | +0.05 | 52° | Purest forward walker |
| 5 | 68_grunbaum_penrose | 23.03 | -22.70 | +3.90 | 33° | Tile angle geometry |
| 6 | 15_noether | 21.66 | -21.65 | -0.48 | 37° | Near-pure backward |
| 7 | 16_franklin | 20.27 | -17.49 | +10.24 | 56° | Strong diagonal |
| 8 | 63_cage_prepared | 18.34 | -9.97 | +15.39 | 34° | Disrupted curie → crab |
| 9 | 60_fibonacci_phyllotaxis | 16.51 | +9.72 | +13.34 | 186° | Golden angle, fallen |
| 10 | 4_sayama | 15.26 | -14.56 | -4.58 | 53° | Complex attractor |

## Attractor Taxonomy

Every gait was instrumented with per-step telemetry (position, orientation, ground contacts, joint position/velocity/torque at every 10th sim step). From joint phase portraits, speed variability, and tilt time series, each gait is classified into one of four dynamical attractor types:

| Type | Count | Character |
|---|---|---|
| **fixed_point** | 1 | Converges to stationary equilibrium. The bouncer: perfectly still, zero displacement, zero tilt. |
| **limit_cycle** | 15 | Stable periodic orbit. Consistent stride, low speed variability (CV < 0.5). The most reliable walkers: CPG champion, curie, noether_cpg, carry_trade, pelton, rubato, etc. |
| **complex** | 37 | Moving but not strictly periodic. Subdivides into 5 subtypes (see below). Most gaits live here. |
| **chaotic/fallen** | 6 | Tips over (tilt > 60). Unbounded phase portrait. Tesla, lamarr, gamma_squeeze, original, gallop, blues_shuffle. |

### Complex Attractor Subtypes

The 37 complex gaits are not one category — they split into 5 dynamically distinct subtypes based on FFT spectral content, speed autocorrelation, trajectory curvature, and transient analysis:

| Subtype | Count | Mean Disp | Character |
|---|---|---|---|
| **multi_frequency** | 23 | 14.0m | Multiple competing frequencies in speed signal. Spectrally rich. The largest subgroup. |
| **drifter** | 5 | 24.6m | High displacement with irregular rhythm. Rivals limit cycles. Produced by amplified curie_asymmetric_drive motif. |
| **quasi_periodic** | 5 | 8.4m | Nearly periodic (autocorrelation 0.4–0.78). At the limit-cycle boundary — closest to reclassification. |
| **transient_decay** | 3 | 5.5m | Activity changes dramatically over time (speed CV > 1.0). Not in steady state. Includes bifurcation-boundary gaits. |
| **wobbler** | 1 | 0.8m | Active joint oscillation with high tilt but no displacement. Rocking without locomotion. |

The `canonical_antisymmetric` motif spans 3 subtypes (multi_frequency, quasi_periodic, transient_decay) — structurally simple but dynamically versatile. The `curie_asymmetric_drive` motif splits cleanly: multi_frequency when moderate, drifter when amplified.

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

- **trajectories.png** — (x, y) path for all gaits, color-coded by time. Shows straight walkers, spirals, diagonals, loops, and the bouncer's single dot.
- **phase_portraits.png** — Joint position vs velocity for both joints. Limit cycles show clean ellipses; chaotic fallers show unbounded spirals; the bouncer is a point.
- **stability_contacts.png** — Tilt (red) and ground contact count (blue) over time. Shows exactly when fallers tip, and how stable gaits maintain low tilt.
- **speed_profiles.png** — Instantaneous speed over time. The CPG champion has remarkably constant high speed. Limit cycles show rhythmic oscillations.
- **torque_profiles.png** — Joint torque over time, showing energetic cost and motor rhythm.

## Weight Motifs

12 structural motifs identified by analyzing weight patterns across all gaits:

| Motif | Members | Signature |
|---|---|---|
| canonical_antisymmetric | 7 | w_i3 = -w_i4 (equal and opposite drive to each motor) |
| curie_asymmetric_drive | 7 | Torso weight differs between motors; stronger front-leg drive |
| noether_involution | 4 | Weights approximately negate under motor swap |
| same_drive_symmetric | 4 | Both motors receive similar-sign drive |
| minimal_wiring | 6 | 3-4 active synapses (most weights zero) |
| cpg_dominant | 1 | Cross-wiring dominates over sensor input |
| half_center_oscillator | 3 | Hidden neurons with mutual inhibition/excitation |
| spin_torque | 4 | Asymmetric cross-wiring creates net rotational torque |
| positive_feedback_cascade | 1 | Self-reinforcing motor feedback |
| walk_and_spin | 3 | Strong base pattern + cross-wiring for simultaneous translation and rotation |
| crab_walk | 7 | Asymmetric drive patterns amplified by cross-wiring for lateral motion |
| bifurcation_boundary | 2 | Configuration at a sharp phase transition; tiny perturbation changes behavior qualitatively |

## Key Discoveries

**All gaits are perfectly deterministic.** PyBullet DIRECT mode produces zero variance across trials (CV=0.000 for all gaits). What matters is sensitivity — how much a gait's behavior changes with small weight perturbations.

**Motor balance ratio predicts direction.** The ratio of total drive to motor 3 vs motor 4 (MB = sum|w_i3| / sum|w_i4|) predicts forward vs backward walking. MB < 1.0 tends forward, MB > 1.0 tends backward.

**Cross-wiring unlocks new behaviors.** The 4 motor-to-motor weights (w34, w43, w33, w44) enable CPG oscillation, spin, and crab walking that are impossible with sensor-to-motor weights alone.

**Lateral motion was a blind dimension.** Before measuring DY, all gaits were characterized only by forward displacement. The average |DY| across the zoo is 3.75m. Many gaits walk at significant diagonal angles.

**Bifurcation boundaries are sharp.** The bouncer configuration [0, +1, -1, 0, -1, +1] is perfectly still (DX=0, YAW=0, tilt=0). Reducing one weight by 10% (w24: 1.0 to 0.9) produces a 188 spin. The sharpest behavioral cliff in the zoo.

**Limit cycles are the best walkers — and we know why.** The 15 limit-cycle gaits include all top-5 displacement leaders. Three mechanisms explain this:
- *Directional efficiency*: Limit cycles convert 74% of path length into net displacement vs 54% for complex gaits (1.4x). They walk straighter — less motion wasted on non-locomotory oscillation.
- *Speed consistency*: 1.8x lower speed variability (CV 0.41 vs 0.72). Uniform stride wastes less energy on acceleration/deceleration.
- *Joint asymmetry*: Limit cycles use joints more asymmetrically (0.20 vs 0.13), creating directed thrust rather than symmetric rocking.

**Complex:drifters are the hidden contenders.** Five "complex" gaits classified as drifters actually rival limit cycles in displacement (mean 24.6m vs 23.5m) and directional efficiency (0.756 vs 0.740). They walk far despite lacking clean periodicity. The curie_asymmetric_drive motif produces drifters when amplified — these are "almost-limit-cycles" with enough core periodicity to walk straight.

**Efficiency is a property of structure.** The torque-displacement Pareto frontier contains only 6 of 58 moving gaits. Motif determines efficiency: `half_center_oscillator` (0.414) and `minimal_wiring` (0.230) dominate; `spin_torque` (0.017) is dead last. Efficiency correlates strongly with directional efficiency (r=0.70) — walking straight IS being efficient. Displacement barely correlates with torque (r=0.21) — high torque does not guarantee high displacement.

**Evolution finds knife-edge solutions.** Evolving crab walkers with |DY| fitness produced a verified champion (|DY|=40.64, crab ratio 6.06), beating the hand-designed record by 41%. But the first attempt stored weights at 6 decimal places, and the rounding shifted the champion from DY=-36.34 to DY=-25.68 — a 30% performance drop from ~1e-7 weight change. Full float64 precision is required for reproducibility. This is the reality gap operating through weight precision rather than environmental noise.

**Heading is controllable but the landscape has cliffs.** Sweeping 83 configurations (MB ratio × cross-wiring asymmetry) achieves full 360° heading coverage, but the control surface is nonlinear and non-monotonic (CW-heading correlation = -0.14). The heading distribution is bimodal: a forward cluster (-30° to +30°) and a backward cluster (150°-180°). Pure lateral headings (60°-120°) are rare — structurally difficult for this body plan. Consistent with bifurcation findings: nearby parameters can produce wildly different headings.

## Efficiency Frontier

Only 6 gaits are Pareto-optimal (no other gait beats them on both displacement AND torque):

| Gait | Displacement | Mean Torque | Efficiency | Attractor |
|---|---|---|---|---|
| 43_hidden_cpg_champion | 50.03m | 120.78 | 0.414 | limit_cycle |
| 30_garch | 24.64m | 91.89 | 0.268 | complex:multi_frequency |
| 39_bulgarian | 17.39m | 77.72 | 0.224 | complex:multi_frequency |
| 42_polyrhythm | 11.41m | 69.65 | 0.164 | complex:multi_frequency |
| 17_lamarr | 5.03m | 47.17 | 0.107 | chaotic/fallen |
| 7_fuller_dymaxion | 2.45m | 4.08 | 0.602 | limit_cycle |

The frontier is bookended by two limit cycles: fuller_dymaxion (minimum energy, sparse wiring) and the CPG champion (maximum performance, hidden-layer oscillator). The middle is complex:multi_frequency — moderate displacement at moderate cost.

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

Renders all configured gaits to `videos/` using offscreen PyBullet rendering piped to ffmpeg. 89 videos currently recorded.

## Files

| File | Description |
|---|---|
| `synapse_gait_zoo.json` | Complete catalog: 77 gaits, weights, measurements, motifs, attractors, telemetry metrics |
| `record_videos.py` | Video recording infrastructure (offscreen render to ffmpeg) |
| `simulation.py` | Main simulation runner |
| `body.urdf` | Robot body definition (3-link) |
| `brain.nndf` | Current neural network weights (overwritten by scripts) |
| `constants.py` | Physics parameters (SIM_STEPS=4000, DT=1/240, gravity, friction) |
| `videos/` | 89 MP4 videos of gaits (gitignored) |
| `artifacts/telemetry_full/` | Per-step telemetry JSONL for all gaits (gitignored) |
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
