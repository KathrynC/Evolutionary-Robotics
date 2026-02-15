# Synapse Gait Zoo & Motion Gait Dictionary

**Kathryn Cramer** — University of Vermont

A research project in evolutionary robotics that began as a catalog of 116 discovered gaits for a 3-link PyBullet robot and evolved into a systematic investigation of **language-model-mediated robot control**: can LLMs translate human motion concepts into neural network weights that produce recognizable locomotion?

The answer is yes — and the investigation has progressed through four layers: from gait catalogs to weight-space landscapes to LLM-mediated control to a **Transactional Interpretation of Quantum Mechanics (TIQM)**-inspired framework where LLMs choose both the robot's brain AND its physics, and a second LLM evaluates whether the resulting behavior resonates with semantic intent.

## The Robot

```
[FrontLeg]---[Torso]---[BackLeg]
```

- 3 rigid links connected by 2 hinge joints
- 3 touch sensors (one per link), 2 motors (one per joint)
- Neural network maps sensor values to motor commands through weighted synapses
- Simulated in PyBullet (DIRECT mode, deterministic — zero variance across trials)
- 4000 timesteps at 240 Hz (16.67 simulated seconds per run)

## Setup

```bash
conda env create -f environment.yml   # Python 3.11, pybullet 3.25, numpy 1.26.4, matplotlib, fastapi
conda activate er
```

## Project Structure

The project has three layers, each building on the last:

### Layer 1: The Gait Zoo (116 gaits)

A catalog of 116 discovered gaits organized by structural motif, behavioral tag, and attractor dynamics. Each gait is a fixed-weight neural network (no learning at runtime) that produces a distinct locomotion style. See [The Zoo](#the-zoo) below.

### Layer 2: The Weight-Space Landscape (~25,000 simulations)

Research campaigns that mapped the 6D weight space, revealing it to be riddled with behavioral cliffs — non-differentiable, fractal, isotropically chaotic. See [The Weight-Space Landscape](#the-weight-space-landscape-behavioral-cliffs-and-fractal-sensitivity) below.

### Layer 3: LLM-Mediated Robot Control (706+ LLM trials, 58 motion concepts, 365 dictionary entries)

Systematic investigation of using LLMs to translate semantic concepts into neural network weights. This produced the **Motion Gait Dictionary** — a multilingual catalog of 58 motion concepts across 5 languages and 5 LLMs — and led to the central insight about sensory death and nervous system evolution. See [Motion Gait Dictionary](#motion-gait-dictionary) and [The Central Insight](#the-central-insight-sensory-death-and-the-case-for-proprioception) below.

### Layer 4: TIQM Framework & Alternate Physics (8,000+ character trials, 500+ probe seeds)

Inspired by Cramer's Transactional Interpretation of Quantum Mechanics: the LLM emits an "offer wave" (6 weights + 6 physics parameters), PyBullet propagates it through simulated physics, and a second LLM (the "absorber") evaluates semantic resonance. The gait is the standing wave that forms when semantic intent and physical realization agree.

Key discoveries:
- **Death encoding**: LLMs encode whether fictional characters die through a 6-number channel. Dying characters get heavier gravity, higher restitution (bouncier falls). Survivors get Earth gravity (9.81 m/s²). Horatio — the faithful friend who survives Hamlet — walks the furthest.
- **Physics-level narrative encoding**: qwen3-coder chose gravity=20.0, friction=0.1, max_force=50N for ALL Romeo & Juliet characters — a crushing, low-friction tragic world.
- **12D collapse elimination**: Expanding the output channel from 6D (weights only) to 12D (weights + physics) eliminates weight collapse entirely.
- **Embodiment-readiness**: LLMs already contain embodied knowledge from absorbing all human writing about bodies and movement. The engineering problem is building adequate output devices, not teaching LLMs about bodies.

See [TIQM Framework](#tiqm-framework--alternate-physics), [Character & Death Experiments](#character--death-experiments), and [Semantic Probe Taxonomy](#semantic-probe-taxonomy) below.

---

## Motion Gait Dictionary

**58 motion concepts, 365 entries, 5 LLMs, 5 languages.**

Each entry is a motion word (e.g., "crawl", "Hüpfen", "蹒跚") translated by an LLM into 6 synapse weights, simulated for 4000 steps, and evaluated with Beer-framework analytics. The dictionary maps the semantic-to-behavioral channel: how faithfully can a robot execute a human motion concept?

### The Five LLMs

| Model | Source |
|---|---|
| gpt-4.1-mini | OpenAI API |
| qwen3-coder:30b | Ollama (local) |
| deepseek-r1:8b | Ollama (local) |
| llama3.1:latest | Ollama (local) |
| gpt-oss:20b | Ollama (local) |

### The Five Languages

English, German, Finnish, French, Chinese (Mandarin)

### The 58 Concepts

amble, backward_walk, bounce, charge, circle, crab_walk, crawl, creep, dash, drag, drift, fall, forward_walk, freeze, gallop, glide, headstand, hop, limp, lunge, lurch, march, patrol, pivot, plod, prance, prowl, retreat, roam, rock, roll, rush, saunter, scurry, shuffle, skid, skip, slide, spin, sprint, stagger, stand_still, stomp, stride, stumble, sway, tiptoe, trot, turn_left, turn_right, twirl, twist, waddle, walk_and_spin, wander, weave, wobble, zigzag

### Compilation Video

All 58 concepts were recorded as individual simulation videos and compiled into a 3.5-hour presentation:

| Property | Value |
|---|---|
| Duration | 211.7 minutes |
| Concepts | 58 (alphabetical) |
| Total entries | 365 |
| Format | 1280x720 @ 30 fps, H.264 |

Opening title card, per-concept section dividers with descriptions, lower-third captions showing word/language/model for each clip, and closing credits citing all LLMs and key references.

### Key Files

| File | Description |
|---|---|
| `artifacts/motion_gait_dictionary_v2.json` | Complete dictionary: 58 concepts, 365 entries with weights and Beer analytics |
| `artifacts/motion_gait_dictionary_v2.pdf` | Formatted PDF of the dictionary |
| `build_motion_dictionary.py` | Assembles dictionary from motion seed experiment data |
| `motion_seed_experiment_v2.py` | LLM weight generation experiment (365 trials) |
| `record_concept_videos.py` | Per-concept video renderer (PyBullet + PIL + ffmpeg) |
| `compile_concept_video.py` | Compilation assembler with title/section/credits cards |
| `render_dictionary_pdf.py` | PDF renderer with CJK font support |

### Further Reading

- [artifacts/motion_seed_experiment_v2_report.md](artifacts/motion_seed_experiment_v2_report.md) — Full experiment report
- [artifacts/motion_gait_dictionary_compilation_notes.md](artifacts/motion_gait_dictionary_compilation_notes.md) — Video production notes and key insights

---

## The Central Insight: Sensory Death and the Case for Proprioception

The most important observation from watching hundreds of simulated gaits across all 58 concepts:

### The Problem

Many gaits "die" mid-simulation. The robot lands in a configuration where no body link contacts the ground. With only 3 touch sensors (all exteroceptive), the neural network receives all-zero input, motor outputs flatline, and the robot freezes permanently. This is **sensory death**: loss of environmental contact produces total sensory blackout.

Evidence: "freeze" has 22 dictionary entries — the most of any concept. A large fraction of weight configurations produce gaits that eventually land wrong and lose all sensory feedback.

### The Survivors

The most interesting surviving gaits are the **hoppers** — where the "torso" link is repurposed as a limb and a leg becomes a de facto "head." These work because the torso-as-limb keeps cycling through ground contact, keeping touch sensors alive. They've accidentally solved the sensory death problem by finding a geometry that maintains feedback throughout the gait cycle.

### The Fix

The robot needs sensors that can't go dark. Proprioceptive sensors — joint angles, angular velocity, accelerometers — always produce non-zero signal regardless of ground contact. Biological organisms have both exteroception (touch) and interoception (proprioception). Proprioception is the sense that never goes dark.

### The Reframe: Communication, Not Optimization

The Motion Gait Dictionary establishes a **call and response** between humans and robots:

1. Human provides a semantic concept ("crawl", "hop", "retreat")
2. LLM translates it into neural network weights
3. Robot executes the weights as behavior
4. Human evaluates whether behavior matches intent

This is a **communication channel**. Its fidelity is bottlenecked not by the weights (365 examples show LLMs can generate diverse weights) and not by the vocabulary (58 concepts across 5 languages) — it's bottlenecked by the **nervous system's expressive bandwidth**. Three touch sensors and two motors can't faithfully distinguish "saunter" from "amble."

**The next evolutionary pressure should be on the nervous system topology itself** — not to make the robot faster, but to increase the fidelity of the semantic-to-behavior channel. We're not optimizing for locomotion. We're optimizing for communication.

See [artifacts/motion_gait_dictionary_compilation_notes.md](artifacts/motion_gait_dictionary_compilation_notes.md) for the full argument.

---

## The Zoo

**116 gaits across 11 categories, 13 structural motifs, 22 behavioral tags, 112 unique motif-tag profiles.**

All gaits and their weights are stored in `synapse_gait_zoo.json` (v1) and `synapse_gait_zoo_v2.json` (v2, with Beer-framework analytics). Full taxonomy in `artifacts/gait_taxonomy.json`. Full-resolution telemetry (4000 records/gait at 240 Hz) in `artifacts/telemetry/`.

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

### Leaderboards

#### Displacement (|DX|)

| # | Gait | DX | Category |
|---|---|---|---|
| 1 | 43_hidden_cpg_champion | +50.11 | hidden_neurons |
| 2 | 21_noether_cpg | -43.23 | cross_wired_cpg |
| 3 | 22_curie_amplified | +37.14 | cross_wired_cpg |
| 4 | 5_pelton | +34.70 | persona_gaits |
| 5 | 100_kcramer_anthology | +32.32 | persona_gaits |

#### Crab Walking (|DY|)

| # | Gait | DY | Crab Ratio | Origin |
|---|---|---|---|---|
| 1 | 56_evolved_crab_v2 | -40.64 | 6.06 | evolved |
| 2 | 57_evolved_sidewinder | -39.32 | 9.85 | evolved |
| 3 | 58_evolved_crab_positive_v2 | +38.08 | 32.55 | evolved |

#### Spin (|YAW|)

| # | Gait | YAW | Turns |
|---|---|---|---|
| 1 | 44_spinner_champion | +838 | 2.33 |
| 2 | 45_spinner_stable | -749 | 2.08 |

---

## The Weight-Space Landscape: Behavioral Cliffs and Fractal Sensitivity

Three research campaigns (~12,000 headless simulations) mapped the 6D weight space. The central finding: the fitness landscape is **riddled with behavioral cliffs** — regions where a tiny weight change (r=0.05) causes the robot to shift to a completely different locomotion regime.

- **80%** of random points have a cliff (>5m displacement shift) at r=0.05
- The landscape is **formally non-differentiable**: derivatives diverge as ~1/r with no smoothness floor
- The fractal structure comes from **contact dynamics**: binary foot strikes turn smooth weight changes into fractal behavioral changes
- **Gradient descent is impossible** at any scale; evolutionary algorithms work because they don't assume local smoothness

### Cliff Taxonomy

Five empirical cliff types: Canyon (38%), Step (30%), Precipice (26%), Slope (6%), Fractal.

See [FINDINGS.md](FINDINGS.md) for the full analysis, [artifacts/cliff_taxonomy_commentary.md](artifacts/cliff_taxonomy_commentary.md) for philosophical implications.

---

## Resonance Mapping

~2,150 simulations bypassing the NN, driving joints with pure sine waves to map the body's mechanical transfer function.

- Body has broad resonance across 1–4 Hz, not a single frequency
- **Amplitude is the chaos gateway**: below 0.8 rad smooth, above 0.8 rad fractal
- **Neural networks far exceed the open-loop ceiling**: best sine wave = 32.7m; best NN = 60.2m (Novelty Champion). The NN senses contact events in real time and adjusts timing — closed-loop feedback is worth +27.5m over open-loop.

See [artifacts/resonance_mapping_summary.md](artifacts/resonance_mapping_summary.md).

---

## Structured Random Search: The LLM as Weight-Space Sampler

Five conditions, 100 trials each (495 total), testing whether LLMs produce structured weight distributions:

| Condition | Dead% | Median |DX| | Max |DX| | Phase Lock |
|---|---|---|---|---|
| **baseline** | 8% | **6.64m** | 27.79m | 0.613 |
| **verbs** | 5% | 1.55m | 25.12m | 0.850 |
| **theorems** | 8% | 2.79m | 9.55m | **0.904** |
| **bible** | **0%** | 1.55m | **29.17m** | 0.908 |
| **places** | **0%** | 1.18m | 5.64m | 0.884 |

The LLM is a *conservative* sampler: it avoids both death and greatness, clustering in a tight behavioral subspace with high coordination but modest displacement.

### The Triptych

Three gaits from the Bible condition where meaning transfers across substrates:

- **The Pale Horse** (Revelation 6:8) — DX = +29.17m, overall champion. *Death rides fast.*
- **The Whirling Wind** (Ecclesiastes 1:6) — Efficiency = 0.00495, most efficient gait. *The eternal wind cycles with minimal waste.*
- **The Conservation Law** (Noether's Theorem) — DX = 0.031m, nearly dead. Every weight pair exactly equal and opposite: perfect cancellation. *A conservation law conserves.*

See [artifacts/structured_random_triptych.md](artifacts/structured_random_triptych.md).

---

## Celebrity & Archetypometrics Experiments

### Celebrities (132 names → 4 gaits)

132 celebrity names from tokenization lexicons — politicians, Kardashians, tech billionaires, musicians, actors, athletes, authors, historical figures — collapse into exactly 4 robot gaits. The 4-archetype structure cuts across every domain boundary: Donald Trump, LeBron James, and Beyonce share one gait; Julian Assange, OJ Simpson, and Billie Eilish share a backward-walking gait. The LLM encodes **narrative role** (assertive, default, contrarian, transgressor), not domain knowledge.

See [artifacts/structured_random_celebrities_analysis.json](artifacts/structured_random_celebrities_analysis.json) and [artifacts/multimodel_celebrity_findings.md](artifacts/multimodel_celebrity_findings.md).

### Archetypometrics (2000 fictional characters from 341 stories)

2000 fictional characters seeded through the LLM pipeline. Characters cluster by narrative archetype, not by source work. The weight-space is structured by storytelling convention.

See [artifacts/archetypometrics_findings.md](artifacts/archetypometrics_findings.md).

---

## Character & Death Experiments

### Multi-Model Character Experiment (8,000+ trials)

The largest experiment to date: 8,000+ fictional characters from hundreds of literary works, seeded through 4 local LLMs (qwen3-coder:30b, deepseek-r1:8b, llama3.1:latest, gpt-oss:20b). Each character becomes a set of 6 synapse weights and a PyBullet simulation.

**Weight collapse**: 66% of trials (1,293/1,943 in early checkpoints) collapse into shared weight vectors. The top cluster contained 111 identical weight vectors assigned to characters as diverse as Applejack (My Little Pony), Saul Goodman (Breaking Bad), and Claire Standish (The Breakfast Club). Predominantly qwen3-coder:30b and gpt-oss:20b.

### Death Encoding

Characters who die in their stories are encoded differently from characters who survive — through a 6-number channel that knows nothing about plot:

| Character | Dies? | DX | Speed | Notes |
|---|---|---|---|---|
| Horatio (Hamlet) | No | +8.70 | 1.44 | Walks the furthest — the faithful witness |
| Hermione (Harry Potter) | No | -1.39 | 1.01 | Steady, purposeful movement |
| Dumbledore (Harry Potter) | Yes | +3.54 | 1.00 | Fastest of the dying — goes out actively |
| Hamlet (Hamlet) | Yes | +0.49 | 0.68 | Slow, deliberate — "the rest is silence" |
| Gatsby (Great Gatsby) | Yes | -1.22 | 0.47 | Drifts backward |
| Boromir (Lord of the Rings) | Yes | 0.00 | 0.00 | Completely still — already fallen |

See [artifacts/tiqm_death_test_phase2.json](artifacts/tiqm_death_test_phase2.json), [artifacts/the_seance_observation.md](artifacts/the_seance_observation.md).

---

## TIQM Framework & Alternate Physics

Mapping John G. Cramer's Transactional Interpretation of Quantum Mechanics to the LLM-robot pipeline:

| TIQM Concept | Pipeline Element |
|---|---|
| Emitter | LLM generating weights + physics params |
| Offer wave | The 12D vector (6 weights + 6 physics) |
| Transaction medium | PyBullet simulation |
| Absorber | Second LLM (VLM) evaluating the gait |
| Confirmation wave | VLM's semantic resonance score |
| Standing wave | The resulting gait — when intent and physics agree |

### Alternate Physics as Expressive Channel

The LLM chooses not just the robot's brain (6 synapse weights) but also its world (6 physics parameters):

| Parameter | Range | What it means |
|---|---|---|
| Gravity | 0.0 – 20.0 m/s² | Weight of existence |
| Friction | 0.1 – 2.5 | World's grip on the body |
| Restitution | 0.0 – 0.8 | Bounciness — does the world forgive falls? |
| Damping | 0.0 – 2.0 | How fast energy dissipates |
| Max force | 50 – 200 N | Motor strength |
| Mass ratio | 1.0 – 3.0 | Limb heaviness relative to torso |

### Experimental Protocols

- **Phase 1**: Single extended offer — LLM chooses 12 parameters at once
- **Phase 2**: Offer + confirmation — second LLM rates semantic resonance of the resulting behavior
- **Iterative TIQM**: Multiple rounds of offer/confirmation until resonance converges
- **Multi-offer**: Multiple competing transactions, highest resonance wins

### Key Files

| File | Description |
|---|---|
| `tiqm_experiment.py` | Full TIQM implementation: all 4 protocols |
| `artifacts/tiqm_framework.md` | Theoretical framework and design rationale |
| `artifacts/tiqm_romeo_juliet_phase2.json` | 10 R&J characters: physics-level tragic world |
| `artifacts/tiqm_death_test_phase2.json` | 12 die/survive characters: death encoding in 12D |
| `artifacts/the_seance_observation.md` | The spookiness of embodiment-readiness |

---

## Semantic Probe Taxonomy

Beyond motion words and fictional characters, we developed a curated taxonomy of semantic probes — diverse seed types that stress-test the LLM→behavior channel in different ways. Each dataset is curated by one LLM and cross-verified by a different LLM to avoid self-confirmation bias.

| Probe Type | Seeds | Source | Curator | Status |
|---|---|---|---|---|
| Motion words | 58 | Multilingual motion vocabulary | Manual | Complete (365 trials) |
| Fictional characters | 8,000+ | 341 literary works | Manual + LLM | Running |
| Celebrities | 132 | Tokenization lexicons | Manual | Complete (4 gaits) |
| OEIS sequences | 396 | Integer sequence encyclopedia | qwen3-coder:30b | Running |
| Mathematicians | 104 | History of mathematics | qwen3-coder:30b | Curated + verified |
| TV Tropes | 185 | Narrative trope taxonomy | qwen3-coder:30b | Curated + verified |
| Stith Thompson motifs | 192 | Folklore motif index | qwen3-coder:30b | Curated + verified |

### Curation & Verification Pipeline

Each curated dataset goes through a three-stage pipeline:

1. **Curation** (qwen3-coder:30b): LLM generates structured entries with metadata (descriptions, categories, energy levels, movement associations)
2. **Verification** (llama3.1:latest): Different LLM cross-checks facts, flags hallucinations, corrects errors
3. **Decollapse** (optional): When weight vectors collapse, a third LLM generates enriched descriptions to re-prompt the original model

| Script | Description |
|---|---|
| `curate_mathematicians.py` | 104 mathematicians with birth/death years, fields, movement metaphors |
| `curate_tv_tropes.py` | 185 tropes across 9 categories with energy/valence annotations |
| `curate_stith_thompson.py` | 192 folklore motifs across 22 categories |
| `verify_curated_data.py` | Cross-model verification of all curated datasets |
| `decollapse_weights.py` | Re-prompts collapsed weight clusters with enriched context |

---

## Agent Infrastructure

### Experiment Runners

| Script | Description |
|---|---|
| `character_seed_experiment.py` | 8,000+ character trials across 4 models |
| `oeis_seed_experiment.py` | OEIS integer sequence trials |
| `motion_seed_experiment_v2.py` | Motion Gait Dictionary trials |
| `tiqm_experiment.py` | TIQM alternate-physics trials |

### Analysis & Quality

| Script | Description |
|---|---|
| `pest_control.py` | Static analysis: finds inconsistencies across all experiment data |
| `verify_curated_data.py` | Cross-model fact-checking of curated datasets |
| `decollapse_weights.py` | Enrichment-based retry of collapsed weight vectors |

---

## Categorical Structure & Formal Validation

Scripts that empirically validate the categorical structure of the Sem→Wt→Beh pipeline:

| Script | What it tests |
|---|---|
| `categorical_structure.py` | Functor F (Sem→Wt), map G (Wt→Beh), composition G∘F, sheaf structure, information geometry |
| `fisher_metric.py` | LLM output variance: 22/30 seeds fully deterministic, 8/30 show binary mode switching |
| `perturbation_probing.py` | Directly measured cliffiness at 37 LLM weight vectors |
| `yoneda_crosswired.py` | 10-synapse topology increases faithfulness for run/jump verbs (5x improvement) but not walk/crawl |
| `hilbert_formalization.py` | L² Gram matrix of 121 gait trajectories, RKHS kernel regression, spectral analysis |
| `llm_seeded_evolution.py` | LLM weights as launchpad for evolution: reaches 85.09m vs 48.41m from random seeds (+76%) |

See [artifacts/unified_framework_synthesis.md](artifacts/unified_framework_synthesis.md) for the unified categorical framework connecting this project to two other projects (Spot a Cat, AI Seances).

---

## Paper

A draft paper is available at [artifacts/paper_draft.md](artifacts/paper_draft.md) (PDF: [artifacts/paper_draft.pdf](artifacts/paper_draft.pdf)):

> **"Reality Is What Doesn't Go Away When You Change the Physics Engine: Structural Transfer from Language Models Through Physical Substrates"**
>
> 706 LLM-mediated trials across 7 semantic conditions, plus ~25,000 supporting simulations. Key finding: LLM-seeded evolution reaches 85.09m displacement — 76% better than random-seeded evolution — because the LLM's conservatism places weights in smooth, evolvable regions of parameter space.

The project has since expanded to 8,000+ character trials, TIQM alternate-physics experiments, and a curated probe taxonomy of 500+ seeds across 7 categories. A second paper focusing on death encoding and the TIQM framework is in development.

---

## Running a Gait

```bash
# Generate robot body and brain files
python3 generate.py          # produces body.urdf, brain.nndf, world.sdf

# Run a simulation (GUI)
python3 simulate.py

# Run headless
HEADLESS=1 python3 simulate.py

# Run with telemetry
HEADLESS=1 TELEMETRY=1 TELEMETRY_VARIANT_ID=my_gait python3 simulate.py

# Generate all 116 telemetry files
python3 generate_telemetry.py

# Record concept videos
python3 record_concept_videos.py

# Compile the full compilation video
python3 compile_concept_video.py
```

## Core Simulation Files

| File | Description |
|---|---|
| `simulation.py` | Main simulation runner: PyBullet lifecycle, Sense→Think→Act loop |
| `robot.py` | ROBOT class: loads body.urdf + brain.nndf, manages sensors/motors |
| `motor.py` | MOTOR class: joint control |
| `sensor.py` | SENSOR class: touch sensor reading per link |
| `world.py` | WORLD class: loads ground plane |
| `generate.py` | Generates body.urdf, brain.nndf, world.sdf |
| `constants.py` | Central physics config: SIM_STEPS=4000, DT=1/240, MAX_FORCE=150 |
| `pyrosim/` | External submodule: URDF/SDF generation, NN loading, PyBullet helpers |

## Research Campaign Scripts

Self-contained simulation campaigns (hundreds to thousands of headless sims each). Each defines its own harness and writes to `artifacts/`.

| Script | Sims | Description |
|---|---|---|
| `walker_competition.py` | ~5,000 | 5 optimization algorithms compete |
| `causal_surgery.py` | ~600 | Mid-simulation brain transplants and ablation |
| `behavioral_embryology.py` | ~500 | Tracks gait emergence in first 500 steps |
| `gait_interpolation.py` | ~1,000 | Linear interpolation between champion pairs |
| `resonance_mapping.py` | ~2,150 | Open-loop sine sweeps (body's transfer function) |
| `atlas_cliffiness.py` | ~6,400 | Spatial atlas of behavioral cliffs |
| `cliff_taxonomy.py` | ~500 | Adaptive probing of 50 cliffiest points |
| `random_search_500.py` | 500 | Random weight sampling |
| `analyze_dark_matter.py` | — | Classifies "dead" gaits (spinners, rockers, vibrators) |
| `timestep_atlas.py` | ~800 | DT sensitivity across 7 timestep values |
| `structured_random_*.py` | 495+ | LLM-mediated weight generation (5 conditions) |
| `motion_seed_experiment_v2.py` | 365 | Motion Gait Dictionary data generation |
| `character_seed_experiment.py` | 8,000+ | Multi-model fictional character trials |
| `oeis_seed_experiment.py` | 396 | OEIS integer sequence trials |
| `tiqm_experiment.py` | 50+ | TIQM alternate physics + confirmation waves |
| `categorical_structure.py` | — | Functor/sheaf/info geometry validation |
| `fisher_metric.py` | — | LLM output variance (300 Ollama calls) |
| `perturbation_probing.py` | ~259 | Cliffiness at LLM weight vectors |
| `hilbert_formalization.py` | — | L² Gram matrix, RKHS regression |
| `llm_seeded_evolution.py` | ~1,000 | Evolution from LLM seeds vs random |

## Further Reading

### Top-Level Documents

- [FINDINGS.md](FINDINGS.md) — Scientific analysis and key discoveries
- [PERSONAS.md](PERSONAS.md) — The 18 persona gait themes
- [REFERENCES.md](REFERENCES.md) — Annotated bibliography
- [CONTINUATION_PLAN.md](CONTINUATION_PLAN.md) — Research continuation plan (Parts A-F)

### Key Artifacts

- [artifacts/paper_draft.md](artifacts/paper_draft.md) — Draft paper: structural transfer from LLMs through physical substrates
- [artifacts/unified_framework_synthesis.md](artifacts/unified_framework_synthesis.md) — Unified categorical framework (3 projects)
- [artifacts/motion_gait_dictionary_compilation_notes.md](artifacts/motion_gait_dictionary_compilation_notes.md) — Compilation video notes and sensory death insight
- [artifacts/motion_seed_experiment_v2_report.md](artifacts/motion_seed_experiment_v2_report.md) — Motion seed experiment v2 report
- [artifacts/structured_random_triptych.md](artifacts/structured_random_triptych.md) — The Triptych: Revelation, Ecclesiastes, Noether
- [artifacts/archetypometrics_findings.md](artifacts/archetypometrics_findings.md) — 2000 fictional characters analysis
- [artifacts/multimodel_celebrity_findings.md](artifacts/multimodel_celebrity_findings.md) — Multi-model celebrity experiment
- [artifacts/resonance_mapping_summary.md](artifacts/resonance_mapping_summary.md) — Body's mechanical transfer function
- [artifacts/dark_matter_analysis.md](artifacts/dark_matter_analysis.md) — Taxonomy of dead gaits
- [artifacts/walker_competition_analysis.md](artifacts/walker_competition_analysis.md) — 5 optimization algorithms compared
- [artifacts/gamified_progressive_search_review.md](artifacts/gamified_progressive_search_review.md) — Review of QD/MAP-Elites gamified search proposal
- [artifacts/persona_effectiveness_theory.md](artifacts/persona_effectiveness_theory.md) — Theory of persona-to-weight structural transfer
- [artifacts/cliff_taxonomy_commentary.md](artifacts/cliff_taxonomy_commentary.md) — Philosophical implications of behavioral cliffs
- [artifacts/tiqm_framework.md](artifacts/tiqm_framework.md) — TIQM theoretical framework: offer waves, confirmation, alternate physics
- [artifacts/the_seance_observation.md](artifacts/the_seance_observation.md) — The seance: why character experiments are spooky, and embodiment-readiness
- [artifacts/technical_report_vacc_proposal.md](artifacts/technical_report_vacc_proposal.md) — Technical report and VACC scaling proposal

## Key References

- **Beer 1996** — "Toward the evolution of dynamical neural networks for minimally cognitive behavior." Proc. Simulation of Adaptive Behavior.
- **Sims 1994** — "Evolving virtual creatures." Proc. SIGGRAPH '94, pp. 15–22.
- **Bongard 2013–2024** — Ludobots: An Introduction to Evolutionary Robotics. University of Vermont / Reddit r/ludobots.
- **Cully et al. 2015** — Behavioral repertoires via MAP-Elites; closest precedent to the zoo concept.
- **McGeer 1990** — Passive dynamic walking and limit cycles.
- **Ijspeert 2008** — CPG review; our hidden-layer champion is effectively a CPG.
- **Cramer 1986** — "The Transactional Interpretation of Quantum Mechanics." Reviews of Modern Physics 58(3). The offer/confirmation wave framework adapted for LLM→physics→VLM transactions.
