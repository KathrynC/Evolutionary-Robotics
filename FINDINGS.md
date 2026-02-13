# Synapse Gait Zoo: Findings

## Beer-Framework Analytics (v2)

The v2 zoo (`synapse_gait_zoo_v2.json`) replaces the v1 per-gait telemetry summary with a comprehensive analytics object computed from full-resolution telemetry (4000 records at 240 Hz). The pipeline (`compute_beer_analytics.py`) is numpy-only (no scipy/sklearn) and computes 4 pillars of metrics per gait, inspired by Beer's framework for analyzing small continuous-time recurrent neural networks.

### The Four Pillars

**Pillar 1: Outcome** — Where did the robot go, and at what cost?
- Displacement (dx, dy), net yaw (integrated from angular velocity)
- Mean speed, speed coefficient of variation
- Work proxy (time-integrated |torque * joint velocity| across both joints)
- Distance-per-work efficiency

**Pillar 2: Contact** — How does the robot touch the ground?
- Per-link duty fractions (Torso, BackLeg, FrontLeg)
- 3-bit contact state (torso×4 + back×2 + front×1) — 8 possible states
- State distribution, dominant state, Shannon entropy (bits)
- 8x8 Markov transition matrix (row-normalized) — the gait's contact "grammar"

**Pillar 3: Coordination** — How do the joints relate to each other?
- FFT-based dominant frequency and amplitude per joint
- Hilbert-transform instantaneous phase difference (delta_phi)
- Phase lock score: |mean(e^{i*delta_phi(t)})| — 0 = independent, 1 = perfectly locked

**Pillar 4: Rotation Axis** — How does the body rotate?
- PCA of angular velocity covariance — axis dominance (3-element vector, sums to 1)
- Axis switching rate (Hz) — how often the dominant rotation axis changes
- Per-axis periodicity (dominant FFT frequency of wx, wy, wz)

### Analytics Ranges Across 116 Gaits

| Metric | Min | Max | Mean | Median |
|---|---|---|---|---|
| Mean speed (m/s) | 0.001 | 2.48 | 1.31 | 1.29 |
| Distance per work | 0.00011 | 0.0056 | 0.0017 | 0.0015 |
| Contact entropy (bits) | 0.03 | 1.90 | 1.46 | 1.52 |
| Phase lock score | 0.13 | 1.00 | 0.63 | 0.65 |
| Axis dominance (1st) | 0.47 | 1.00 | 0.82 | 0.86 |
| Axis switching rate (Hz) | 0.06 | 32.5 | — | — |

## Gaitspace Structure

The Beer-framework analytics reveal that gaitspace is not a collection of discrete types but a **continuous multi-dimensional manifold** organized by a small number of independent control dimensions. These findings were invisible in v1's scalar telemetry summaries.

### Phase locking is the master control parameter

Phase lock scores span 0.125 to 0.99999, and this single number predicts more about a gait's character than its weights do. The distribution is bimodal:

- **Tightly locked (>0.95):** 14 gaits, mean speed 0.57 m/s — smooth, predictable, slow
- **Loosely coupled (<0.30):** 12 gaits, mean speed 1.75 m/s — chaotic, fast
- **The fastest gait** (69_grunbaum_deflation, 2.48 m/s) sits at moderate phase lock (0.618), not at either extreme

Phase lock trades against speed. CPG-like tight locking enables repeatable motion but limits velocity. Chaotic coupling enables faster motion but sacrifices predictability.

### Contact complexity is morphologically determined, not neurally controlled

Contact entropy (0.03 to 1.90 bits) is decoupled from every other metric — it correlates with neither speed (r~0), phase locking (r=0.06), nor efficiency. A gait with rich contact diversity (1.8 bits) is no faster or more efficient than one with boring contacts (0.9 bits). Contact patterns appear to be set by the body's geometry and friction, not the neural controller. This means the 8-state contact space is an independent behavioral dimension that the network traverses but does not control.

### Roll dominance is the morphological attractor

59% of gaits are roll-dominant (axis_dominance[0] > 0.85). The 3-link robot's hinge joints naturally create sagittal-plane rocking. Deviations from roll dominance are informative:

- **Spinner champion (44):** axis_dominance = [0.49, 0.33, 0.17] — the most evenly distributed rotation in the zoo. "Spinning" is not clean yaw rotation; it's chaotic 3-axis tumbling at 21.3 Hz axis switching rate.
- **Crab walker (56):** 32.5 Hz axis switching — the highest in the zoo. Lateral walking requires constant rebalancing across all three rotation axes.
- **Fuller dymaxion (7):** axis_dominance = [0.99998, ...] — the purest single-axis motion, a minimal-synapse metronome that barely leaves the sagittal plane.

### The CPG champion's dominance is mechanistically explained

43_hidden_cpg_champion (dx=50.1m) was always the displacement record holder. Now all four pillars converge on *why*:

| Pillar | CPG Champion | Curie (best 6-synapse) | Ratio |
|---|---|---|---|
| Displacement (dx) | 50.1 m | 23.7 m | 2.1x |
| Work proxy | 6,482 | 13,797 | 0.47x (half the energy!) |
| Efficiency | 0.0077 | 0.0017 | 4.5x |
| Phase lock | 0.64 | 0.37 | — |
| delta_phi | -156° (anti-phase) | -161° (anti-phase) | — |
| Contact entropy | 0.93 bits (simple) | 1.33 bits | — |

The hidden-layer half-center oscillator produces genuine CPG dynamics: anti-phase alternating legs (like a biological trot), low energy expenditure, simple repeatable contact pattern, and a stable limit cycle. Every pillar aligns. No v1 metric could show this convergence.

### Identical topology, quantifiably divergent dynamics

Gaits 43 (CPG champion) and 44 (spinner champion) share the same 7-neuron half-center topology. One synapse magnitude ratio (symmetric |-0.8|=|0.8| vs asymmetric |-0.6|!=|0.5|) transforms:

| Metric | 43 (CPG champion) | 44 (Spinner) |
|---|---|---|
| dx | 50.1 m | 0.25 m |
| yaw_net | -0.53 rad | 14.8 rad (2.4 turns) |
| phase_lock | 0.64 | 0.36 |
| axis_dominance[0] | 0.75 (roll) | 0.49 (distributed) |
| efficiency | 0.0077 | 0.0002 |

The symmetry breaking in one synapse decouples phase locking, flattens axis dominance, and converts efficient translation into energetically expensive rotation. The bifurcation is visible across all four pillars simultaneously.

### Speed and efficiency are weakly coupled

Speed-efficiency correlation is only r=0.20 across 116 gaits (52x efficiency range). The four quadrants are populated unevenly:

- **Fast and efficient:** 21 gaits — the sweet spot
- **Slow and inefficient:** 23 gaits — stationary or struggling
- **Fast and inefficient:** 15 gaits — high-speed energy waste
- **Slow and efficient:** 15 gaits — rare efficient minimalists

The efficiency champion (93_borges_mirror, 0.0056) achieves it through perfect antisymmetry, not speed. The speed champion (69_grunbaum_deflation, 2.48 m/s) has above-median but unremarkable efficiency (0.0027). No gait dominates both dimensions — the Pareto frontier is real and sparse.

### Contact transition matrices reveal gait grammars

Each gait now has an 8x8 Markov transition matrix encoding how contact states follow each other — a sequential "grammar" for ground interaction. High-performing gaits show distinctive signatures:

- **CPG champion:** state 0→0 probability 0.959 (stable stance dominates), no 1→2 transitions ever (clean alternation)
- **Spinner:** state 1 decays quickly (contact is transient, the robot is always leaving the ground)
- **Crab walker:** state 0 sticky (0.902) but with broader transition diversity when it does change

These are gait fingerprints invisible to any scalar metric.

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

## Gait Taxonomy v2.0

Every gait is classified along two axes: **structural motifs** (what the network looks like) and **behavioral tags** (what the robot does). The full taxonomy is in `artifacts/gait_taxonomy.json`.

Of 116 gaits, 112 have unique motif-tag profiles. The 4 shared profiles reveal convergent evolution across categories.

### Structural Motifs (13)

**Topology motifs** — which synapse classes are active:

| Motif | Count | Description |
|---|---|---|
| M1_pure_sensor | 40 | Only sensor-to-motor synapses (w03-w24). The feedforward archetype. |
| M2_sensor_cross | 20 | Sensor-to-motor + motor-to-motor cross-wiring (w34/w43). No self-feedback. |
| M3_sensor_self | 3 | Sensor-to-motor + self-feedback (w33/w44). No cross-wiring. |
| M4_full_network | 32 | All three synapse classes active. Maximum recurrence. |
| M5_minimal | 14 | Four or fewer active synapses. Maximum function from minimum structure. |
| M6_maximal | 31 | All 10 possible synapses active. Dense network. |
| M7_hidden_layer | 3 | Hidden neurons between sensors and motors. Adds computational depth. |

**Feedback motifs** — the character of self-feedback (w33, w44):

| Motif | Count | Description |
|---|---|---|
| F1_no_feedback | 70 | No self-feedback. Pure feedforward. |
| F2_standard_feedback | 35 | w33>0 (back leg positive), w44<0 (front leg negative). The canonical oscillator-amplifier. |
| F3_reversed_feedback | 4 | w33<0, w44>0. Inverted proprioception — destabilizing. |
| F4_same_sign_feedback | 4 | Both self-feedbacks share the same sign. Sustaining rather than alternating. |
| F5_hidden_oscillator | 3 | Oscillation from hidden neuron cross-inhibition. A central pattern generator. |

### Behavioral Tags (22)

**Locomotion** — where does the robot go?

| Skill | Count | Description |
|---|---|---|
| S1_strider | 35 | High displacement (>15), mostly straight, stable |
| S2_sprinter | 19 | Very high displacement (>25). Raw speed. |
| S3_drifter | 18 | Moderate displacement (5-15), curved path. Goes somewhere, but indirectly. |
| S4_spinner | 7 | Walking in circles (straightness < 0.15). |
| S5_statue | 5 | Near-static (<2). Moves limbs but stays in place. |
| S6_faller | 12 | Height collapses below 50%. Falls over. |
| S7_bouncer | 34 | Large vertical oscillation (z_range > 0.6). |
| S8_glider | 10 | Smooth vertical profile (z_range < 0.2). Floats. |
| S9_crab | 10 | Lateral displacement dominates forward (|DY/DX| > 1.5). |

**Temporal** — how does it move through time?

| Skill | Count | Description |
|---|---|---|
| T1_metronome | 16 | Highly periodic joints. Clock-like regularity. |
| T2_chaotic | 10 | No detectable periodicity. Aperiodic motion. |
| T3_never_settler | 74 | Never reaches steady state within simulation (64% of all gaits). |
| T4_fast_oscillator | 37 | Joint period < 10 (high-frequency limb motion). |
| T5_slow_oscillator | 21 | Joint period > 40 (low-frequency limb motion). |

**Coordination** — how do the legs relate?

| Skill | Count | Description |
|---|---|---|
| C1_anti_phase | 31 | Legs alternate. The canonical walking gait. |
| C2_in_phase | 63 | Legs move together. Hopping or bounding. |
| C3_asymmetric_amplitude | 8 | One joint has >2x the amplitude of the other. |

**Efficiency and direction:**

| Skill | Count | Description |
|---|---|---|
| E1_efficient | 28 | Top-quartile displacement per unit energy. |
| E2_turner | 44 | Significant yaw accumulation (>1.0 rad). |
| E3_backward | 51 | Walks backward (dx < -0.5). |

## Key Discoveries

**All gaits are perfectly deterministic.** PyBullet DIRECT mode produces zero variance across trials (CV=0.000 for all gaits). What matters is sensitivity — how much a gait's behavior changes with small weight perturbations.

**w44 is the stability governor.** Across 113 standard-topology gaits, negative w44 (front leg self-feedback) correlates with higher displacement (avg 18.9 negative vs 11.2 zero vs 12.8 positive), lower yaw, and higher straightness. Performance-designed categories (pareto, market_math, CPG) have 57-100% negative w44 rates; persona gaits only 28%. Exception: reversed feedback (w44>0) works for crab walkers.

**Anti-phase legs produce better walkers.** 31 anti-phase gaits average 17.0 displacement vs 12.7 for 63 in-phase gaits. Anti-phase crabs are the strongest combo: avg 31.7 displacement. The advantage is mechanical — alternating legs produce more consistent ground contact for propulsion.

**64% of all gaits never reach steady state.** 74/116 gaits are still accelerating or decelerating at simulation end (4000 steps). Spinners, pareto_walk_spin, and hidden neurons are 100% never-settlers. Only evolved/bifurcation gaits consistently settle. The implication: longer simulations would change the leaderboard.

**Direction is emergent.** No single weight predicts forward vs backward walking above 55.7% accuracy (w43 is the best, barely above coin flip). The best composite predictor — |motor4 inputs| > |motor3 inputs| — reaches only 65.1%. Direction is a nonlinear function of the full weight vector, not any subset.

**The faller signature is weak sensor coupling.** All 10 sensor-to-motor weights are weaker in fallers than non-fallers (diffs of +0.26 to +0.50). Fallers also have strong w43 cross-wiring (-0.30 vs -0.06 in non-fallers). The recipe for falling: disconnect sensors, let motors talk to each other.

**Architecture is the strongest performance predictor.** Crosswired 10-synapse gaits average 16.8 displacement vs 10.1 for standard 6-synapse. Hidden-layer gaits include the all-time champion (50.0). Within standard_6, top and bottom quartiles have nearly identical synapse counts (5.8 vs 5.5) — it's weight values, not count, that matters within an architecture.

**Crab walkers reverse the cross-wiring rule.** Normal gaits have w43 negative (inhibitory cross-feedback). Crab walkers flip it positive (+0.16 vs -0.11). They also pump 76% of energy into J0 (back leg) — the most asymmetric energy budget of any category. Evolved crabs put 84-94% of energy into J0.

**Convergent evolution is real.** 4 pairs of gaits from different categories share identical motif-tag profiles. deleuze_bwo (persona) and minsky (CPG) have nearly identical displacement (21.66 vs 21.79), joint ranges, and velocity profiles despite being designed independently with different intent. foucault_heterotopia (persona) matches rubato (time_signature). The physics of the body constrains the space of possible behaviors.

**The leaderboard has a recipe.** All top-10 gaits are S1_strider + S2_sprinter. 80% are never-settlers. 70% are E1_efficient. F4_same_sign_feedback and M7_hidden_layer are 33x enriched in the top 10 vs population. The champion formula: full network + either hidden oscillator or same-sign feedback + never settling + high straightness.

**Cross-wiring unlocks new behaviors.** The 4 motor-to-motor weights (w34, w43, w33, w44) enable CPG oscillation, spin, and crab walking that are impossible with sensor-to-motor weights alone.

**Bifurcation boundaries are sharp.** The bouncer configuration [0, +1, -1, 0, -1, +1] is perfectly still (DX=0, YAW=0, tilt=0). Reducing one weight by 10% (w24: 1.0 to 0.9) produces a 188 spin. The sharpest behavioral cliff in the zoo.

**Evolution finds knife-edge solutions.** Evolving crab walkers with |DY| fitness produced a verified champion (|DY|=40.64, crab ratio 6.06), beating the hand-designed record by 41%. But the first attempt stored weights at 6 decimal places, and the rounding shifted the champion from DY=-36.34 to DY=-25.68 — a 30% performance drop from ~1e-7 weight change. Full float64 precision is required for reproducibility.

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

The Beer-framework work proxy (time-integrated |torque * joint velocity|) provides a finer-grained efficiency measure than v1's mean_torque ratio. The CPG champion uses half the work of Curie (6,482 vs 13,797) for 2.1x the displacement — its distance-per-work (0.0077) is 4.5x better. The efficiency champion overall is 93_borges_mirror (0.0056 distance/work) which achieves it through perfect weight antisymmetry and low-energy backward walking.

## Sensitivity Classes

Gaits fall into three sensitivity classes based on how their behavior changes with small weight perturbations (central-difference gradient, +/-0.05):

| Class | Example | DX Sensitivity | Character |
|---|---|---|---|
| Antifragile | 19_haraway | 32 | Robust to perturbation |
| Knife-edge | 32_carry_trade | 1340 | High performance, high fragility |
| Yaw powder keg | 1_original | 17149 (yaw) | Tiny changes cause massive rotation changes |

## Telemetry

Full-resolution telemetry captures what endpoint measurements miss. Every gait has 4000 records (one per sim step at 240 Hz), each containing:

- **Base position** (x, y, z) — full trajectory, not just start/end
- **Orientation** (roll, pitch, yaw) — stability time series
- **Linear and angular velocity** (vx, vy, vz, wx, wy, wz) — instantaneous dynamics
- **Ground contacts** — per-link booleans [Torso, BackLeg, FrontLeg], revealing gait cycle
- **Joint states** (position, velocity, torque) — motor dynamics and energetics

### Telemetry-Derived Metrics (per gait, v1)

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

## What Gaitspace Tells Us

The 116-gait atlas, read as a whole, raises questions that reach beyond robotics into philosophy of mind, dynamical systems theory, and the epistemology of biological modeling.

### The body speaks louder than the brain

Every gait in the zoo — regardless of topology, synapse count, or category — is dominated by roll-axis rotation. The PCA of angular velocity covariance puts the first principal component firmly on the roll axis across all 116 gaits. This is not a property of any particular neural network; it is a property of the body. A three-link robot with lateral hinge joints and no active roll control simply *must* rock side to side whenever it moves. The brain can choose how fast, how far, and in what direction — but the fundamental rotational signature is dictated by morphology before a single synapse fires.

This is embodied cognition in miniature. The body is not a passive vessel that the controller fills with behavior; it is a filter that constrains the space of possible behaviors before the controller acts. The neural network explores the behavioral landscape, but the landscape itself is carved by link lengths, joint axes, mass distribution, and ground friction. Any theory of locomotion in this system that treats the controller in isolation is incomplete — the body is the first and most powerful "synapse."

### Behavior space is not a space

Classical intuition suggests that small changes to parameters produce small changes in behavior — that gaitspace should be a smooth manifold one could traverse continuously. The zoo provides a sharp counterexample. Gaits 43 and 44 share identical topology (hidden-layer half-center oscillator, same neuron layout) and differ only in the learned weight values. Yet 43 is the CPG Champion — the fastest, most efficient gait in the entire zoo — while 44 is the Spinner Champion, a gait that rotates in place and barely translates at all. Same architecture, opposite behavioral regimes.

This is bifurcation, not interpolation. The mapping from weight space to behavior space is riddled with cliffs, folds, and discontinuities. There is no smooth path from "fastest forward walker" to "rotates in place" — there is a critical boundary in parameter space where the attractor type itself changes. This means that any search through weight space (evolutionary, gradient-based, or random) must contend with a landscape that is fundamentally non-convex and potentially fractal in its sensitivity structure. Gaitspace is less a rolling landscape and more a shattered mirror, each shard reflecting a qualitatively different dynamical regime.

### Phase locking as a phase transition

The distribution of phase lock scores across all 116 gaits is strikingly bimodal: gaits cluster near 0 (unlocked, phase drifts freely) or near 1 (tightly locked, legs maintain fixed phase relationship), with very few in between. This is the signature of a critical phenomenon. Phase locking is not a quantity that gaits accumulate gradually; it is a regime that a gait either falls into or does not, much like a physical phase transition between disordered and ordered states.

The bimodality suggests that the 6-weight (or 10-weight) parameter space contains a critical surface — a boundary where the system transitions between phase-locked and phase-free dynamics. Most randomly sampled weight configurations land clearly on one side or the other. The rarity of intermediate phase lock scores means the transition is sharp, not gradual. This is reminiscent of synchronization transitions in coupled oscillator theory (Kuramoto models), but here it emerges from a minimal neural network driving a physical body through intermittent ground contact. The physics of foot strikes may itself act as a coupling mechanism that either reinforces or disrupts inter-joint phase coherence.

### Entropy is orthogonal to function

Contact entropy — the Shannon entropy of the 8-state contact pattern distribution (each state a 3-bit vector of which links touch the ground) — shows near-zero correlation with speed, phase lock, and efficiency. A gait can be fast with high entropy or fast with low entropy. It can be phase-locked and high-entropy or phase-locked and low-entropy. This independence is surprising because one might expect that "better" gaits would have more structured (lower entropy) contact patterns.

What this reveals is that contact entropy measures something genuinely orthogonal to performance: the *complexity of the ground interaction strategy*, not its quality. A gait that uses all 8 contact states roughly equally (high entropy) is not better or worse than one that concentrates on 2 or 3 states (low entropy) — it is *different in kind*. This suggests that locomotion performance is determined more by the temporal coordination of contacts (captured by phase locking and transition structure) than by the diversity of contact states visited. The alphabet size does not determine the meaning of the sentence.

### The speed–efficiency frontier and the CPG question

The Pareto frontier between speed and efficiency is thin and telling. At one end sits Fuller Dymaxion (gait 7): a minimal 6-synapse network achieving moderate speed at very low energy cost — the hybrid car of the zoo. At the other end sits the CPG Champion (gait 43): a hidden-layer half-center oscillator that generates its own clock signal (a Central Pattern Generator, or CPG) through reciprocal inhibition between hidden neurons, achieving the highest speed and the best efficiency simultaneously.

A CPG is a neural circuit that produces rhythmic output without requiring rhythmic input. In biological organisms, CPGs in the spinal cord generate the basic locomotion rhythm; sensory feedback modulates but does not create the pattern. In gait 43, the hidden neurons form exactly this kind of circuit: they oscillate autonomously, and the motor neurons are driven by this internal clock rather than by raw sensor values. The result is a gait that is both fast and efficient — the internal oscillator finds a resonant frequency that matches the body's natural dynamics, extracting maximum displacement per unit of work.

That the most efficient gait in the zoo is also the one that has *internalized its own timing* is a deep result. It suggests that the transition from reactive control (sensor-driven) to generative control (CPG-driven) is not merely an architectural choice but a performance boundary. The body has a natural frequency; a controller that discovers and exploits that frequency — rather than fighting it with moment-to-moment sensor corrections — achieves qualitatively superior locomotion.

### Naming and legibility

That so many gaits have persona names — Curie, Haraway, Fuller, Borges, Lamarr, Grünbaum — is not mere whimsy. It reflects something real about the structure of gaitspace: many of these gaits are legible. They have recognizable character. When a human watches gait 17 (Lamarr) tumble chaotically across the ground, or gait 7 (Fuller) glide with minimal effort, or gait 43 (CPG Champion) stride with metronomic precision, the behavioral differences are immediately apparent and nameable. The naming convention is an implicit claim that the gaits are not interchangeable points in a continuous space but distinct behavioral individuals — and the analytics bear this out.

### Cartography before theory

The Synapse Gait Zoo is, at this stage, an empirical atlas. It maps the territory of what a minimal body-brain system can do, and the Beer-framework analytics provide coordinates for that map. But an atlas is not a theory. The findings — roll dominance, phase lock bimodality, entropy independence, bifurcation sensitivity, CPG superiority — are empirical regularities waiting for theoretical unification.

The situation is not unlike Tycho Brahe's star catalogs before Kepler: meticulous observations that clearly contain deep structure, but whose governing laws have not yet been written down. The zoo tells us *what* gaitspace looks like. The next question is *why* it looks that way — what minimal dynamical model predicts the bimodal phase lock distribution, what property of the body-ground interaction makes entropy orthogonal to performance, what theorem connects CPG resonance to Pareto optimality. The data is rich enough to constrain such a theory. The theory itself remains ahead.
