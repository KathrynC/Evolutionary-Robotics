NOTE: THIS README CONTAINS ERRORS AND IS IN REVISION

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

## The Zoo

**116 gaits across 11 categories, 13 structural motifs, 22 behavioral tags, 112 unique motif-tag profiles.**

All gaits and their weights are stored in `synapse_gait_zoo.json`. Full taxonomy (motifs, behavioral tags, per-gait features) is in `artifacts/gait_taxonomy.json`. Per-step telemetry (400 records/gait) is in `artifacts/telemetry/`. Videos are in `videos/`.

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

### Persona Gait Themes

The 74 persona gaits include eighteen thematic groups added after the original 20:

- **Fibonacci (3 gaits)**: Golden ratio proportions in weight structure. `fibonacci_self` uses phi^-2 self-feedback for stable diagonal walking. `fibonacci_phyllotaxis` maps the golden angle (137.5°) to successive weights — chaotic but covers 10m while spinning 595°. `fibonacci_spiral` scales weights by phi (0.382, 0.618, 1.0) for near-pure lateral motion.

- **John Cage (3 gaits)**: `cage_433` is near-silence (weights at ±0.01, no visible motion — the ambient physics IS the performance). `cage_prepared` takes the curie pattern and flips two weights (like inserting bolts into piano strings), transforming a forward walker into a crab walker (DY=15.39). `cage_iching` uses chance operations (random.seed(1952)) for all 10 weights — randomness produces the strongest backward walker in the group (DX=-19.95).

- **Jack Womack (3 gaits)**: Austerity and survival. `womack_random_acts` uses only 2 sensor synapses + self-feedback — institutional feedback loops drive backward locomotion with the lowest tilt (10°) relative to displacement in the zoo. `womack_ambient` has only 2 active synapses (the absolute minimum for locomotion). `womack_terraplane` operates at 20-40% of normal weight magnitude — existence at the poverty line.

- **Branko Grünbaum (3 gaits)**: Tilings and patterns. `grunbaum_penrose` uses sin/cos of the Penrose tile angles (36° and 72°) as weights — pure aperiodic geometry produces a 23m backward walker. `grunbaum_deflation` cascades through phi^-1 levels (1.0, 0.618, 0.382, 0.236) like a Penrose tiling deflation rule, with self-feedback at phi^-3 (31m, tilt 27°). `grunbaum_defect` starts with perfect p6m symmetry (uniform ±0.7) and breaks it with a single cross-wire — a third intentional near-fixed-point alongside bouncer and cage_433.

- **Patrick X. Gallagher (3 gaits)**: Analytic number theory. `gallagher_multiplicative` uses products of prime pairs from {2,3,5,7} — multiplicative (not additive) weight construction. `gallagher_gaps` encodes consecutive prime differences [1,2,2,4,2,4] as weights — very stable (tilt 10°). `gallagher_sieve` places weights only at prime-indexed positions (indices 2,3,5,7), zeroing composites — only 4 active synapses. The sparsity IS the sieve.

- **Philip K. Dick (3 gaits)**: Reality instability. `dick_valis` adds tiny cross-wiring noise to a curie base — the noise IS the information, producing 28m (second-strongest persona gait). `dick_replicant` is an exact curie copy with one weight sign-flipped — the Voigt-Kampff failure point that collapses a 24m walker to 4m. `dick_ubik` reverses all curie signs — entropy runs backward into 353° of spinning chaos.

- **Max Ernst (3 gaits)**: Surrealist process. `ernst_collage` splices curie's motor-3 weights with noether's motor-4 weights — incompatible sources produce a 16m diagonal walker. `ernst_celebes` uses near-maximal weights (0.85-0.95) + CPG for a mechanical beast with only 5° tilt. `ernst_decalcomania` mirrors weights across motors with a tiny asymmetric smear (0.3→0.35) — form emerges from imperfect reflection.

- **Gilles Deleuze (3 gaits)**: Philosophy of difference. `deleuze_fold` uses symmetric base + strong self-feedback (w33=0.7, w44=-0.7) — the interior folds onto the exterior for 23m at 16° tilt. `deleuze_bwo` (Body without Organs) gives every weight identical magnitude (0.55) — no hierarchy, yet 22m of locomotion. `deleuze_rhizome` activates all 10 weights with no zeros and no dominant path.

- **Michel Foucault (3 gaits)**: Power and control. `foucault_heterotopia` whispers sensor input (±0.1) while cross-wiring dominates (w34=0.8) — motors govern themselves, walking 15m at only 4° tilt. `foucault_madness` traps the system in extreme self-feedback (w33=0.9, w44=-0.9) yet still produces 14m. `foucault_panopticon` applies equal surveillance (all weights ±0.5) — total symmetric control produces near-paralysis (2m).

- **André Breton (3 gaits)**: Surrealist manifesto. `breton_nadja` maps digits of sqrt(2) to weights — the irrational wanderer walks 12m. `breton_chance` combines Fibonacci ratios with prime gaps — objective chance as meaningful coincidence (12m diagonal). `breton_automatic` uses pure random weights (seed=1924, year of the first Surrealist Manifesto) — falls over but the automatic text is real.

- **Gertrude Stein (3 gaits)**: Repetition and the continuous present. `stein_rose` repeats one magnitude (0.65) six times — "a rose is a rose is a rose" as neural architecture (2m lateral drift). `stein_continuous_present` uses same-sign self-feedback (w33=0.8, w44=0.8) — both motors sustain rather than alternate, producing 6m of lateral drift at 124° yaw. `stein_tender_buttons` encodes ASCII values of "TENDER BUT" as weights — language as material, producing 12m of locomotion from text.

- **Jorge Luis Borges (3 gaits)**: Labyrinths and mirrors. `borges_labyrinth` whispers sensor input (0.1-0.3) while cross-wiring dominates (w34=0.7, w43=-0.9) — the labyrinth has no center, producing 11m at only 16° tilt. `borges_mirror` achieves perfect antisymmetry — every weight to motor 3 is the exact negation of motor 4 — Tlön reflected, walking 22m backward. `borges_aleph` maps digits of e (2.71828...) to weights — the point containing all points produces 9m of diagonal motion.

- **Shirley Jackson (3 gaits)**: Hidden violence and isolation. `jackson_lottery` takes the curie base (village walker) and flips one weight to -1.0 — one synapse drawn, one stone thrown, collapsing 24m to 5m. `jackson_hill_house` reverses self-feedback signs (w33=-0.6, w44=+0.6) — the house looks right outside but the angles are wrong inside (49° tilt). `jackson_castle` activates only 2 of 10 synapses (w13, w24) — total isolation, each leg speaking only to itself, walking 6m diagonally in seclusion.

- **Kathryn Cramer (3 gaits)**: Editing, hypertext, anthology-making. `kcramer_hypertext` activates all 10 weights at different moderate levels — a navigable network where every node connects, nothing dominates, the reader chooses the route (19m backward). `kcramer_salvage` starts from panopticon-like symmetry (±0.5, near-paralysis) but adds asymmetric cross-wiring to break the deadlock — the editorial intervention that rescues a failed text (2m). `kcramer_anthology` curates weights from three champions — w13 from pelton, w04 from curie, w24 from fibonacci — the editor's art of combination (33m, the persona displacement champion).

- **John G. Cramer (3 gaits)**: Transactional interpretation of quantum mechanics. `jgcramer_transactional` propagates offer waves (strong negative motor-3 weights) and confirmation waves (positive motor-4 weights) with the handshake cross-wiring (w34=0.5, w43=-0.5) creating the observable event (5m, falls at 121° tilt). `jgcramer_handshake` makes cross-wiring dominant (w34=0.8, w43=-0.8) — motors exchange signals stronger than their connection to sensors (13m at 28° tilt). `jgcramer_absorber` uses near-maximal self-feedback (w33=0.9, w44=-0.9) with minimal sensor input — the response of the universe that completes the transaction (16m backward).

- **Joseph Papp (3 gaits)**: Democratic theater, public access. `papp_public` uses moderate, accessible weights — nothing extreme, nothing exclusive, the democratic gait anyone could discover through methodical exploration (19m backward at only 12° tilt). `papp_shakespeare` performs the classical curie pattern outdoors with ambient noise — tiny cross-wiring (±0.1) represents wind and traffic (4m). `papp_chorus_line` gives every sensor-motor weight identical magnitude (0.7) — the ensemble moves as one, cross-wiring provides choreographic coordination (6m).

- **Al Pacino (3 gaits)**: Intensity and controlled commitment. `pacino_heat` places near-maximal weights on dominant channels (w13=-0.95, w04=0.95, w24=0.95) with restrained supporting weights — the professional who is disciplined until the moment of action (26m). `pacino_godfather` keeps every weight below 0.65 yet produces directed motion through controlled asymmetry — quiet authority at conversational volume (4m). `pacino_hooah` fires all sensor-motor weights at 0.9 with cross-wiring at 0.5 and self-feedback at 0.4 — the full Pacino, maximum intensity across the board (5m, the question is whether passion produces motion or chaos).

- **Simone Biles (3 gaits)**: Power, proprioception, and the courage to stop. `biles_yurchenko` uses near-maximal uniform sensor drive (0.95) with strong cross-wiring (0.6) but no self-feedback — pure explosive coordination, the vault that changed the code of points (6m displacement, 100% upright). `biles_twisties` inverts self-feedback signs (w33=-0.7, w44=+0.7) with weak sensors (0.2) — the internal model is wrong, proprioception says down is up, the bravest thing she did was stop (8m, 100% upright — the body moves but the map is inverted). `biles_floor` balances asymmetric sensor weights with moderate cross-wiring (0.4) and light self-feedback (0.2) — the floor routine that shows mastery is maximum control, not maximum force (11m).

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

### Persona Gaits (by total displacement)

| # | Gait | Distance | DX | DY | Tilt | Notable |
|---|---|---|---|---|---|---|
| 1 | 5_pelton | 35.27 | +34.70 | +6.32 | 32° | Persona champion |
| 2 | 100_kcramer_anthology | 33.12 | +32.32 | -7.25 | 40° | Curated from 3 champions |
| 3 | 69_grunbaum_deflation | 31.27 | -30.48 | -6.99 | 27° | Penrose deflation cascade |
| 4 | 74_dick_valis | 28.19 | +23.74 | +15.19 | 37° | Divine static as signal |
| 5 | 107_pacino_heat | 26.33 | +23.71 | -11.44 | 32° | Precision and commitment |
| 6 | 10_tesla_3phase | 24.00 | -18.33 | +15.49 | 186° | Fallen but far |
| 7 | 18_curie | 23.73 | +23.73 | +0.05 | 52° | Purest forward walker |
| 8 | 80_deleuze_fold | 23.51 | -21.93 | +8.47 | 16° | Self-feedback locomotion |
| 9 | 68_grunbaum_penrose | 23.03 | -22.70 | +3.90 | 33° | Tile angle geometry |
| 10 | 93_borges_mirror | 21.82 | -21.61 | +3.00 | 17° | Perfect antisymmetry, Tlön reflected |
| 11 | 81_deleuze_bwo | 21.74 | -21.61 | +2.40 | 20° | Uniform magnitude, no hierarchy |
| 12 | 15_noether | 21.66 | -21.65 | -0.48 | 37° | Near-pure backward |
| 13 | 16_franklin | 20.27 | -17.49 | +10.24 | 56° | Strong diagonal |
| 14 | 104_papp_public | 19.40 | -19.40 | -0.35 | 12° | Democratic theater, 12° tilt |
| 15 | 98_kcramer_hypertext | 19.29 | -18.72 | -4.66 | 37° | All 10 weights navigable |

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

## Gait Taxonomy (v2.0)

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
| `synapse_gait_zoo.json` | Complete catalog: 116 gaits, weights, measurements across 11 categories |
| `artifacts/gait_taxonomy.json` | Taxonomy v2.0: 13 motifs, 22 behavioral tags, per-gait feature vectors |
| `artifacts/telemetry/` | Per-step telemetry for all 116 gaits (400 JSONL records + summary.json each) |
| `artifacts/discovery_dig_full116.txt` | Deep analysis output: 13 digs across the full zoo |
| `record_videos.py` | Video recording infrastructure (offscreen render to ffmpeg) |
| `simulation.py` | Main simulation runner |
| `body.urdf` | Robot body definition (3-link) |
| `brain.nndf` | Current neural network weights (overwritten by scripts) |
| `constants.py` | Physics parameters (SIM_STEPS=4000, DT=1/240, gravity, friction) |
| `videos/` | MP4 videos of gaits |
| `artifacts/plots/` | Trajectory maps, phase portraits, stability, speed, torque visualizations |
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
