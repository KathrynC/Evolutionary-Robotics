# Persona Gaits

The 74 persona gaits include twenty originals and eighteen thematic groups of 3 gaits each, named after scientists, thinkers, and artists. Each theme translates a thinker's ideas into neural network weight patterns, producing gaits whose behaviors often resonate with the source material in unexpected ways.

## Persona Leaderboard

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

## Themes

### Fibonacci (3 gaits)

Golden ratio proportions in weight structure. `fibonacci_self` uses phi^-2 self-feedback for stable diagonal walking. `fibonacci_phyllotaxis` maps the golden angle (137.5°) to successive weights — chaotic but covers 10m while spinning 595°. `fibonacci_spiral` scales weights by phi (0.382, 0.618, 1.0) for near-pure lateral motion.

### John Cage (3 gaits)

`cage_433` is near-silence (weights at ±0.01, no visible motion — the ambient physics IS the performance). `cage_prepared` takes the curie pattern and flips two weights (like inserting bolts into piano strings), transforming a forward walker into a crab walker (DY=15.39). `cage_iching` uses chance operations (random.seed(1952)) for all 10 weights — randomness produces the strongest backward walker in the group (DX=-19.95).

### Jack Womack (3 gaits)

Austerity and survival. `womack_random_acts` uses only 2 sensor synapses + self-feedback — institutional feedback loops drive backward locomotion with the lowest tilt (10°) relative to displacement in the zoo. `womack_ambient` has only 2 active synapses (the absolute minimum for locomotion). `womack_terraplane` operates at 20-40% of normal weight magnitude — existence at the poverty line.

### Branko Grünbaum (3 gaits)

Tilings and patterns. `grunbaum_penrose` uses sin/cos of the Penrose tile angles (36° and 72°) as weights — pure aperiodic geometry produces a 23m backward walker. `grunbaum_deflation` cascades through phi^-1 levels (1.0, 0.618, 0.382, 0.236) like a Penrose tiling deflation rule, with self-feedback at phi^-3 (31m, tilt 27°). `grunbaum_defect` starts with perfect p6m symmetry (uniform ±0.7) and breaks it with a single cross-wire — a third intentional near-fixed-point alongside bouncer and cage_433.

### Patrick X. Gallagher (3 gaits)

Analytic number theory. `gallagher_multiplicative` uses products of prime pairs from {2,3,5,7} — multiplicative (not additive) weight construction. `gallagher_gaps` encodes consecutive prime differences [1,2,2,4,2,4] as weights — very stable (tilt 10°). `gallagher_sieve` places weights only at prime-indexed positions (indices 2,3,5,7), zeroing composites — only 4 active synapses. The sparsity IS the sieve.

### Philip K. Dick (3 gaits)

Reality instability. `dick_valis` adds tiny cross-wiring noise to a curie base — the noise IS the information, producing 28m (second-strongest persona gait). `dick_replicant` is an exact curie copy with one weight sign-flipped — the Voigt-Kampff failure point that collapses a 24m walker to 4m. `dick_ubik` reverses all curie signs — entropy runs backward into 353° of spinning chaos.

### Max Ernst (3 gaits)

Surrealist process. `ernst_collage` splices curie's motor-3 weights with noether's motor-4 weights — incompatible sources produce a 16m diagonal walker. `ernst_celebes` uses near-maximal weights (0.85-0.95) + CPG for a mechanical beast with only 5° tilt. `ernst_decalcomania` mirrors weights across motors with a tiny asymmetric smear (0.3→0.35) — form emerges from imperfect reflection.

### Gilles Deleuze (3 gaits)

Philosophy of difference. `deleuze_fold` uses symmetric base + strong self-feedback (w33=0.7, w44=-0.7) — the interior folds onto the exterior for 23m at 16° tilt. `deleuze_bwo` (Body without Organs) gives every weight identical magnitude (0.55) — no hierarchy, yet 22m of locomotion. `deleuze_rhizome` activates all 10 weights with no zeros and no dominant path.

### Michel Foucault (3 gaits)

Power and control. `foucault_heterotopia` whispers sensor input (±0.1) while cross-wiring dominates (w34=0.8) — motors govern themselves, walking 15m at only 4° tilt. `foucault_madness` traps the system in extreme self-feedback (w33=0.9, w44=-0.9) yet still produces 14m. `foucault_panopticon` applies equal surveillance (all weights ±0.5) — total symmetric control produces near-paralysis (2m).

### André Breton (3 gaits)

Surrealist manifesto. `breton_nadja` maps digits of sqrt(2) to weights — the irrational wanderer walks 12m. `breton_chance` combines Fibonacci ratios with prime gaps — objective chance as meaningful coincidence (12m diagonal). `breton_automatic` uses pure random weights (seed=1924, year of the first Surrealist Manifesto) — falls over but the automatic text is real.

### Gertrude Stein (3 gaits)

Repetition and the continuous present. `stein_rose` repeats one magnitude (0.65) six times — "a rose is a rose is a rose" as neural architecture (2m lateral drift). `stein_continuous_present` uses same-sign self-feedback (w33=0.8, w44=0.8) — both motors sustain rather than alternate, producing 6m of lateral drift at 124° yaw. `stein_tender_buttons` encodes ASCII values of "TENDER BUT" as weights — language as material, producing 12m of locomotion from text.

### Jorge Luis Borges (3 gaits)

Labyrinths and mirrors. `borges_labyrinth` whispers sensor input (0.1-0.3) while cross-wiring dominates (w34=0.7, w43=-0.9) — the labyrinth has no center, producing 11m at only 16° tilt. `borges_mirror` achieves perfect antisymmetry — every weight to motor 3 is the exact negation of motor 4 — Tlön reflected, walking 22m backward. `borges_aleph` maps digits of e (2.71828...) to weights — the point containing all points produces 9m of diagonal motion.

### Shirley Jackson (3 gaits)

Hidden violence and isolation. `jackson_lottery` takes the curie base (village walker) and flips one weight to -1.0 — one synapse drawn, one stone thrown, collapsing 24m to 5m. `jackson_hill_house` reverses self-feedback signs (w33=-0.6, w44=+0.6) — the house looks right outside but the angles are wrong inside (49° tilt). `jackson_castle` activates only 2 of 10 synapses (w13, w24) — total isolation, each leg speaking only to itself, walking 6m diagonally in seclusion.

### Kathryn Cramer (3 gaits)

Editing, hypertext, anthology-making. `kcramer_hypertext` activates all 10 weights at different moderate levels — a navigable network where every node connects, nothing dominates, the reader chooses the route (19m backward). `kcramer_salvage` starts from panopticon-like symmetry (±0.5, near-paralysis) but adds asymmetric cross-wiring to break the deadlock — the editorial intervention that rescues a failed text (2m). `kcramer_anthology` curates weights from three champions — w13 from pelton, w04 from curie, w24 from fibonacci — the editor's art of combination (33m, the persona displacement champion).

### John G. Cramer (3 gaits)

Transactional interpretation of quantum mechanics. `jgcramer_transactional` propagates offer waves (strong negative motor-3 weights) and confirmation waves (positive motor-4 weights) with the handshake cross-wiring (w34=0.5, w43=-0.5) creating the observable event (5m, falls at 121° tilt). `jgcramer_handshake` makes cross-wiring dominant (w34=0.8, w43=-0.8) — motors exchange signals stronger than their connection to sensors (13m at 28° tilt). `jgcramer_absorber` uses near-maximal self-feedback (w33=0.9, w44=-0.9) with minimal sensor input — the response of the universe that completes the transaction (16m backward).

### Joseph Papp (3 gaits)

Democratic theater, public access. `papp_public` uses moderate, accessible weights — nothing extreme, nothing exclusive, the democratic gait anyone could discover through methodical exploration (19m backward at only 12° tilt). `papp_shakespeare` performs the classical curie pattern outdoors with ambient noise — tiny cross-wiring (±0.1) represents wind and traffic (4m). `papp_chorus_line` gives every sensor-motor weight identical magnitude (0.7) — the ensemble moves as one, cross-wiring provides choreographic coordination (6m).

### Al Pacino (3 gaits)

Intensity and controlled commitment. `pacino_heat` places near-maximal weights on dominant channels (w13=-0.95, w04=0.95, w24=0.95) with restrained supporting weights — the professional who is disciplined until the moment of action (26m). `pacino_godfather` keeps every weight below 0.65 yet produces directed motion through controlled asymmetry — quiet authority at conversational volume (4m). `pacino_hooah` fires all sensor-motor weights at 0.9 with cross-wiring at 0.5 and self-feedback at 0.4 — the full Pacino, maximum intensity across the board (5m, the question is whether passion produces motion or chaos).

### Simone Biles (3 gaits)

Power, proprioception, and the courage to stop. `biles_yurchenko` uses near-maximal uniform sensor drive (0.95) with strong cross-wiring (0.6) but no self-feedback — pure explosive coordination, the vault that changed the code of points (6m displacement, 100% upright). `biles_twisties` inverts self-feedback signs (w33=-0.7, w44=+0.7) with weak sensors (0.2) — the internal model is wrong, proprioception says down is up, the bravest thing she did was stop (8m, 100% upright — the body moves but the map is inverted). `biles_floor` balances asymmetric sensor weights with moderate cross-wiring (0.4) and light self-feedback (0.2) — the floor routine that shows mastery is maximum control, not maximum force (11m).
