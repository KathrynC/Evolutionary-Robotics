# Experiment Design Notes: Expanding the Semantic Probe Taxonomy

**Date:** 2026-02-15
**Context:** The Synapse Gait Zoo pipeline (LLM → 6 weights → PyBullet simulation → Beer analytics) has now been tested with multiple probe types: motion verbs, fictional characters, integer sequences (OEIS), theorems, Bible verses, and place names. This document designs the next probes and the infrastructure needed to run them at scale.

## The Emerging Framework

Each probe type tests a different layer of what can pass through a 6-weight communication channel between an LLM and a physics engine:

| Probe Type | What It Tests | Cultural Load | Scale | Status |
|---|---|---|---|---|
| Motion verbs | Direct movement semantics | Low | ~100 verbs × 4 models × 5 languages | Done |
| Random baseline | Null hypothesis | None | 100 trials | Done |
| Theorems | Abstract structure | Low-Medium | 100 trials | Done |
| Bible verses | Narrative/moral weight | Medium | 100 trials | Done |
| Place names | Geographic/cultural identity | Medium | 100 trials | Done |
| OEIS sequences | Mathematical structure/aura | Low→High (fame) | 99 seqs × 4 models | Running |
| Character names | Identity, fate, personality | Very High | 2000 chars × 4 models | Running |
| Mathematicians | Person × mathematical style | High | ~200 mathematicians × 4 models | Planned |
| Stith Thompson motifs | Narrative mechanics | Medium | ~200 motifs × 4 models | Planned |
| TV Tropes | Cultural/narrative shorthand | Very High | ~200 tropes × 4 models | Planned |

The key question across all probes: **Is the LLM→weight channel a functor (structure-preserving map) or a lossy compression that erases semantic distinctions?** Different probe types stress-test different aspects of this.

---

## Experiment: Mathematicians

### Rationale

Mathematicians are a hybrid between fictional characters and OEIS sequences. They are *people* — with biographies, personalities, life trajectories, causes of death — but their identities are inseparable from their mathematical work. The LLM's representation of "Euler" fuses the person (blind, prolific, Swiss, worked until his last day) with the mathematics (e, graph theory, infinite series, everything). "Ramanujan" carries divine inspiration, notebooks full of formulas without proofs, tuberculosis, death at 32.

This probe type sits at the intersection of three others:
- **Characters** — mathematicians are real people with life stories and fates
- **OEIS sequences** — each mathematician is associated with specific mathematical structures
- **Theorems** — the existing theorem probe tested abstract results; this tests the people behind them

### Key Hypotheses

1. **Death encoding (again):** Do mathematicians who died young or tragically produce more falls?
   - Dies young/tragically: Galois (20, duel), Ramanujan (32, TB), Turing (41, suicide), Abel (26, TB), Hypatia (murdered), Noether (53, surgery complications), Riemann (39, TB)
   - Long productive life: Euler (76), Erdos (83), Gauss (77), Hilbert (81), Hardy (70), Poincaré (58), Gödel (71, though starved himself)

2. **Mathematical style → movement style:** Does the character of someone's mathematics map to gait character?
   - **Systematic builders** (Euler, Hilbert, Bourbaki): steady, structured gaits?
   - **Wild intuitionists** (Ramanujan, Cantor, Grothendieck): erratic, surprising movement?
   - **Applied/physical** (Newton, Archimedes, Fourier): functional locomotion?
   - **Pure abstractionists** (Gödel, Cohen, Noether): minimal movement? The robot body can't express abstraction?

3. **Fame gradient:** Euler, Gauss, and Newton are household names. Noether, Ramanujan, and Galois are famous within mathematics. Wedderburn, Zorn, and Dilworth are known mainly to specialists. Does fame level correlate with gait distinctiveness (more training data = richer representation)?

### Curated List (~200 mathematicians)

Organized by era and mathematical flavor, with metadata for analysis:

**Ancient & Classical (~20)**
- Euclid, Archimedes, Pythagoras, Hypatia, Eratosthenes
- Diophantus, Apollonius, Thales, Brahmagupta, Al-Khwarizmi
- Omar Khayyam, Fibonacci (Leonardo of Pisa), Liu Hui, Aryabhata
- Bhaskara II, Hero of Alexandria, Ptolemy, Nicomachus
- Zu Chongzhi, Madhava of Sangamagrama

**Renaissance & Early Modern (~25)**
- Descartes, Fermat, Pascal, Leibniz, Newton
- Euler, Bernoulli (Jacob), Bernoulli (Johann), Bernoulli (Daniel)
- Cardano, Tartaglia, Vieta, Napier, Kepler
- Huygens, De Moivre, Stirling, Maclaurin, Taylor
- L'Hôpital, Lagrange, Laplace, Legendre, Monge, Fourier

**19th Century — Golden Age (~50)**
- Gauss, Cauchy, Abel, Galois, Jacobi
- Dirichlet, Riemann, Weierstrass, Dedekind, Kronecker
- Cantor, Klein, Lie, Poincaré, Hilbert
- Ramanujan, Hardy, Littlewood, Noether, Minkowski
- Boole, De Morgan, Cayley, Sylvester, Hamilton (W.R.)
- Chebyshev, Kovalevskaya, Hermite, Jordan, Frobenius
- Peano, Frege, Bolzano, Hausdorff, Lebesgue
- Hadamard, Borel, Baire, Volterra, Mittag-Leffler
- Cartan (Élie), Burnside, Schur, Wedderburn, Dickson

**Early 20th Century (~40)**
- Gödel, Turing, von Neumann, Church, Post
- Banach, Kolmogorov, Wiener, Shannon, Ulam
- Weil, Cartan (Henri), Eilenberg, Mac Lane, Zariski
- Ramsey, Erdos, Szemerédi, Turán, Lovász
- Mandelbrot, Lorenz, Smale, Milnor, Thom
- Kodaira, Serre, Atiyah, Singer, Bott
- Grothendieck, Deligne, Langlands, Tate, Iwasawa
- Nash, Arrow, Shapley, Blackwell, Kakutani

**Late 20th Century & Living (~40)**
- Thurston, Perelman, Hamilton (R.), Yau, Donaldson
- Wiles, Taylor (Richard), Faltings, Shimura, Taniyama
- Tao, Scholze, Mirzakhani, Venkatesh, Zhang (Yitang)
- Conway, Knuth, Penrose, Gowers, Villani
- Drinfeld, Kontsevich, Witten, Voevodsky, Lurie
- Lovelace, Hopper, Goldwasser, Dwork (Cynthia)
- Turing (again as CS), Dijkstra, Knuth, Cook, Karp
- Hawking, Penrose, Dyson, 't Hooft (mathematical physicists as boundary cases)

**Mathematical Collectives & Pseudonyms (~5)**
- Bourbaki (Nicolas), Polymath (D.H.J.), Bletchley Park (as entity)
- Pythagorean Brotherhood, Kerala School

### Metadata to Record Per Mathematician

For post-hoc analysis, annotate each with:

| Field | Values | Purpose |
|---|---|---|
| `era` | ancient/renaissance/19th/early20th/late20th/living | Historical context |
| `died_young` | bool (died before 45) | Death encoding test |
| `cause_of_death` | natural/tragic/violent/suicide/unknown/alive | Death encoding granularity |
| `age_at_death` | int or null | Continuous death variable |
| `primary_field` | algebra/analysis/geometry/topology/logic/combinatorics/number_theory/applied/CS | Mathematical style |
| `style` | systematic/intuitive/visual/algebraic/computational/applied | Working style |
| `fame_level` | household/math_famous/specialist/obscure | Cultural embedding level |
| `associated_sequences` | list of OEIS IDs | Cross-reference with OEIS experiment |

This metadata enables:
- Death age vs fall rate regression
- Mathematical field → gait cluster analysis
- Fame level → weight vector diversity
- Cross-referencing a mathematician's gait with their associated sequence's gait (e.g., does Fibonacci-the-person move like Fibonacci-the-sequence?)

### Prompt Template

```
You are designing a neural controller for a 3-link walking robot...

[standard robot description and weight semantics]

I want you to translate the essence of a mathematician into movement.

Mathematician: {name} ({birth_year}–{death_year})
Known for: {brief_description}

Think about how this person thought, worked, and moved through the world
of ideas. A systematic builder of vast theories might produce steady,
structured locomotion. A wild intuitionist who pulled results from thin air
might produce surprising, unpredictable movement. A person who worked
feverishly and died young might produce intense, urgent motion.

[standard weight grid and output format]
```

### Cross-Experiment Analyses

This experiment enables uniquely powerful cross-references:

1. **Fibonacci-the-person vs Fibonacci-the-sequence**: Do the gaits correlate? If so, the LLM's association is dominant. If not, the mathematical structure and the biographical narrative activate different weight regions.

2. **Euler-the-person vs e-the-constant (A001113)**: Euler is associated with e, but also with graph theory, infinite series, and dozens of other concepts. Does the person produce a richer gait than any single associated sequence?

3. **Ramanujan vs Hardy**: Collaborators with opposite personalities. Ramanujan was intuitive, mystical, died young. Hardy was systematic, atheist, lived long. Do the LLMs distinguish them?

4. **Galois vs Gauss**: Both foundational algebraists. Galois died at 20 in a duel. Gauss lived to 77 as the "Prince of Mathematicians." Maximum contrast on the death axis, similar on the math axis.

5. **Turing**: appears in both the character experiment (as a historical figure) and here as a mathematician. Do the gaits match?

---

## Experiment: Stith Thompson Motif Index

### Source

The Stith Thompson Motif-Index of Folk-Literature classifies recurring narrative elements across world folklore into a hierarchical scheme with ~46,000 motifs. The top-level categories are:

- **A** — Mythological Motifs (Creator, creation, world elements)
- **B** — Animals (mythical, speaking, helpful, marriage to)
- **C** — Tabu (forbidden acts, looking, eating, speaking)
- **D** — Magic (transformation, enchantment, objects, powers)
- **E** — The Dead (resuscitation, ghosts, revenants)
- **F** — Marvels (fairies, otherworld, marvelous creatures)
- **G** — Ogres (witches, devils, cannibals)
- **H** — Tests (recognition, riddles, tasks, quests)
- **J** — The Wise and the Foolish (wisdom, cleverness, fools)
- **K** — Deceptions (tricksters, disguises, escapes)
- **L** — Reversal of Fortune (victorious youngest, unpromising hero)
- **M** — Ordaining the Future (bargains, vows, prophecies)
- **N** — Chance and Fate (luck, gambling, accidents)
- **P** — Society (royalty, warriors, customs)
- **Q** — Rewards and Punishments
- **R** — Captives and Fugitives (abduction, rescue, escape)
- **S** — Unnatural Cruelty (murder, mutilation, abandonment)
- **T** — Sex (love, marriage, birth)
- **U** — The Nature of Life (justice, truth, world order)
- **V** — Religion (worship, saints, religious orders)
- **W** — Traits of Character (favorable, unfavorable)
- **Z** — Miscellaneous (formulae, humor, symbolism)

### Data Acquisition

The full motif index is in the public domain (published 1955-1958). Digital versions exist:

1. **Uther's ATU index** (Aarne-Thompson-Uther) is the modernized tale-type system, but the original Thompson motif index is what we want for granularity.
2. A structured digital version is available at various folklore archives. The motif IDs are hierarchical strings like `D1421.1.3` with natural language descriptions.
3. For a curated experiment, select ~200 motifs spanning all 23 top-level categories, biased toward motifs with vivid movement/action implications.

### Curated Selection Strategy (~200 motifs)

Select ~8-10 motifs per top-level category, prioritizing:
- **Movement-rich motifs**: B211.1 "Speaking horse," R211 "Escape by flying," D671 "Transformation flight"
- **Stasis motifs**: D1960 "Sleeping beauty," C961 "Transformation to stone," E481 "Land of dead"
- **Conflict motifs**: G11.2 "Cannibal giant," S110.3 "Princess thrown from tower," K1810 "Deception by disguise"
- **Category contrasts**: pair motifs from opposite categories (e.g., Q10 "Reward" vs Q411 "Punishment: death")

### Prompt Template

```
You are designing a neural controller for a 3-link walking robot...

[standard robot description and weight semantics]

I want you to translate a narrative motif from world folklore into movement.

Motif: {motif_id} — "{motif_description}"
Category: {category} — {category_description}

This motif appears in folk tales worldwide. Think about the physical action,
energy, and emotional register of this narrative element. A motif about flight
might produce quick, light movement. A motif about death or transformation to
stone might produce stillness. A motif about trickery might produce erratic,
unpredictable movement.

[standard weight grid and output format]
```

### What This Tests

- **Narrative mechanics → movement**: Do motifs about similar actions (flight, chase, transformation) cluster in weight/behavior space?
- **Category structure**: Does the Thompson hierarchy (A→Z) map to any structure in gait space? Are all the "D: Magic" motifs in one behavioral neighborhood?
- **Cross-cultural universals**: These motifs recur across hundreds of cultures. If the LLM produces consistent gaits for universal motifs, that's evidence for deep structure in narrative → movement mapping.
- **Death/stasis probe**: Category E (The Dead) and category C (Tabu) motifs provide another test of the death-encoding hypothesis, independent of character identity.

---

## Experiment: TV Tropes

### Source

TV Tropes (tvtropes.org) catalogs recurring narrative conventions across all media. Tropes are named with punchy, self-explanatory labels that carry enormous cultural baggage. There are ~30,000 trope pages, but many are niche. The "Universal Tropes" and "Omnipresent Tropes" indices provide a curated core.

### Data Acquisition

TV Tropes has no official API, but:
1. The site is wiki-structured with predictable URL patterns
2. A curated list of ~200 tropes can be hand-selected from well-known indices
3. Each trope has a laconic (one-line) description that makes a perfect prompt seed

### Curated Selection Strategy (~200 tropes)

Organize by narrative function, selecting tropes with strong movement/physicality associations and pairing with their inversions:

**Movement tropes:**
- The Slow Walk, Dramatic Chase Opening, Walk and Talk, Unflinching Walk
- Running Gag (literal movement), Le Parkour, Roof Hopping
- Stumbling, Death by Falling, Taking the Bullet

**Character archetype tropes:**
- The Hero, The Mentor, The Dragon, Big Bad, Damsel in Distress
- Byronic Hero, Action Girl, Gentle Giant, Dark Lord
- Plucky Comic Relief, The Stoic, Berserker

**Plot mechanic tropes:**
- Chekhov's Gun, Deus Ex Machina, Red Herring, Plot Twist
- Darkest Hour, Heroic Sacrifice, The Reveal, Cliffhanger
- Happy Ending, Downer Ending, Bittersweet Ending

**Tone/energy tropes:**
- Mood Whiplash, Cerebus Syndrome, Breather Episode
- Rule of Cool, Rule of Funny, Nightmare Fuel
- Heartwarming Moments, Tear Jerker

**Meta/structural tropes:**
- Lampshade Hanging, Breaking the Fourth Wall, Genre Savvy
- Foreshadowing, Flashback, Montage

### Prompt Template

```
You are designing a neural controller for a 3-link walking robot...

[standard robot description and weight semantics]

I want you to translate a narrative trope into movement.

Trope: "{trope_name}"
Description: "{laconic_description}"

This is a recurring storytelling pattern across all media. Think about what
this trope feels like — its energy, its rhythm, its emotional register.
"The Slow Walk" might produce deliberate, measured movement. "Heroic Sacrifice"
might produce a forward lunge. "The Stoic" might produce minimal, controlled
motion. "Mood Whiplash" might produce erratic shifts.

[standard weight grid and output format]
```

### What This Tests

- **Maximum cultural load**: TV Tropes names are designed for instant recognition. They compress entire narrative patterns into 2-4 words. The LLM will activate its richest associative networks.
- **Movement-about-movement**: Tropes like "The Slow Walk" and "Dramatic Chase Opening" are literally about movement patterns. Does the LLM produce gaits that match the described movement?
- **Abstract → concrete**: Tropes like "Chekhov's Gun" or "Deus Ex Machina" have no inherent movement. What does "foreshadowing" look like as a gait?
- **Ending valence**: Happy Ending vs Downer Ending vs Bittersweet Ending — do these produce different gaits? Does Downer Ending produce falls (death encoding again)?
- **Inversion pairs**: The Hero vs The Dragon. Gentle Giant vs Berserker. Do paired inversions produce opposed weight vectors?

---

## Scaling to UVM VACC

### What Is VACC

The Vermont Advanced Computing Core is UVM's shared HPC cluster. Key specs (typical configuration):
- SLURM job scheduler
- Mix of CPU and GPU nodes
- GPU nodes with NVIDIA A100/V100 for ML workloads
- Shared filesystem (home dirs, scratch space)
- Module system for software (Python, CUDA, etc.)
- Job submission via `sbatch`, interactive via `salloc`

### What Needs to Scale

The current Mac Mini setup handles:
- ✅ LLM inference via local Ollama (qwen3-coder:30b, deepseek-r1:8b, llama3.1, gpt-oss:20b)
- ✅ Headless PyBullet simulation (DIRECT mode, no GPU needed)
- ✅ Video generation via ffmpeg + PIL (CPU-bound)
- ✅ Beer-framework analytics (numpy-only, fast)

What needs VACC:
- **Larger LLMs**: Running 70B+ parameter models, or multiple models in parallel
- **Massive parallelism**: 8000 character trials took ~16 hours serial; could be <1 hour with 100 parallel workers
- **VLM back-propagation**: Using vision-language models (Gemini, GPT-4V, or open alternatives) to score videos and feed back into weight generation — this requires GPU inference
- **Video rendering at scale**: Rendering 2000+ character videos for the full dataset

### Architecture: Three-Tier Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Tier 1: LLM Weight Generation (GPU nodes)              │
│                                                         │
│  SLURM array job: 1 task per (probe_seed, model) pair   │
│  Each task: prompt LLM → parse weights → write JSON     │
│  GPU requirement: 1× A100 per LLM instance              │
│  Software: vLLM or Ollama, model weights on scratch     │
│  Parallelism: 4-8 concurrent LLM instances              │
│  Output: weights_batch_NNN.json → shared scratch        │
└──────────────────────┬──────────────────────────────────┘
                       │ weights JSON
┌──────────────────────▼──────────────────────────────────┐
│  Tier 2: Simulation + Analytics (CPU nodes)             │
│                                                         │
│  SLURM array job: 1 task per weight vector              │
│  Each task: load weights → PyBullet DIRECT → analytics  │
│  CPU only, ~17s per trial, embarrassingly parallel      │
│  Parallelism: 100-500 concurrent tasks                  │
│  Output: results_batch_NNN.json → shared scratch        │
└──────────────────────┬──────────────────────────────────┘
                       │ results JSON
┌──────────────────────▼──────────────────────────────────┐
│  Tier 3: Video + VLM Scoring (GPU nodes)                │
│                                                         │
│  SLURM array job: 1 task per character/sequence         │
│  Each task: replay sim → render frames → encode video   │
│            → (optional) VLM score video → record score  │
│  GPU needed for: VLM inference, faster rendering        │
│  Output: videos/ + vlm_scores.json                      │
└─────────────────────────────────────────────────────────┘
```

### VACC Setup Checklist

#### 1. Environment

```bash
# On VACC login node
module load python/3.11
module load cuda/12.x  # for GPU nodes

# Create conda env (or virtualenv)
conda create -n gait_zoo python=3.11
conda activate gait_zoo
pip install pybullet numpy pillow

# For LLM serving
pip install vllm  # or build Ollama from source
# Download model weights to $SCRATCH
```

#### 2. Repository Setup

```bash
cd $HOME
git clone https://github.com/KathrynC/Evolutionary-Robotics.git
cd Evolutionary-Robotics
git checkout reframing
git submodule update --init  # pyrosim
```

#### 3. LLM Serving on VACC

Two options:

**Option A: vLLM (recommended for VACC)**
- vLLM serves models with optimized GPU inference and OpenAI-compatible API
- Can serve multiple models on different ports
- Supports tensor parallelism across multiple GPUs for large models

```bash
# SLURM job to start vLLM server
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

The experiment scripts would need a small adapter to call the OpenAI-compatible API instead of Ollama. Add a `--backend vllm` flag or configure via environment variable:

```python
# In structured_random_common.py or a new vacc_common.py
BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # "ollama" or "vllm"
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions")
```

**Option B: Ollama on VACC**
- Simpler setup, matches local dev environment exactly
- Less GPU-efficient than vLLM but zero code changes
- Install Ollama binary, pull models to scratch

#### 4. SLURM Job Templates

**Tier 1: Weight generation (GPU)**
```bash
#!/bin/bash
#SBATCH --job-name=gait_weights
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --array=0-99  # 100 batches
#SBATCH --output=logs/weights_%a.out

module load python/3.11 cuda/12.x
conda activate gait_zoo

# Each array task processes a batch of seeds
python generate_weights_batch.py \
    --batch-id $SLURM_ARRAY_TASK_ID \
    --batch-size 20 \
    --model llama3.1:latest \
    --backend vllm \
    --vllm-url http://gpu-node:8000/v1/completions
```

**Tier 2: Simulation (CPU)**
```bash
#!/bin/bash
#SBATCH --job-name=gait_sim
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --array=0-999
#SBATCH --output=logs/sim_%a.out

module load python/3.11
conda activate gait_zoo

python simulate_batch.py \
    --batch-id $SLURM_ARRAY_TASK_ID \
    --input-dir $SCRATCH/weights/ \
    --output-dir $SCRATCH/results/
```

**Tier 3: Video + VLM scoring (GPU)**
```bash
#!/bin/bash
#SBATCH --job-name=gait_video
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/video_%a.out

module load python/3.11 cuda/12.x ffmpeg
conda activate gait_zoo

python render_and_score_batch.py \
    --batch-id $SLURM_ARRAY_TASK_ID \
    --results-dir $SCRATCH/results/ \
    --video-dir $SCRATCH/videos/ \
    --vlm-model llava-v1.6  # or Gemini API
```

### 5. VLM Back-Propagation Design

The "Ludobot Video Critic" concept: use a vision-language model to watch the robot videos and score them, then use those scores to guide weight generation.

#### Architecture

```
LLM generates weights → simulate → render video
                                        │
                                        ▼
                                 VLM watches video
                                        │
                                        ▼
                              VLM scores: movement quality,
                              concept embodiment, aesthetic
                                        │
                                        ▼
                              Score feeds back to LLM as
                              few-shot context for next generation
```

#### VLM Options on VACC

| Model | Size | Notes |
|---|---|---|
| LLaVA-NeXT (open) | 7B-34B | Good baseline, runs on single A100 |
| InternVL2 (open) | 8B-76B | Strong video understanding |
| Qwen2-VL (open) | 7B-72B | Excellent video QA |
| Gemini API (cloud) | N/A | Best video understanding, requires API key + egress |

#### VLM Scoring Prompt

```
Watch this 30-second video of a 3-link walking robot.

The robot was given neural network weights generated by an LLM
trying to express: "{seed_concept}"

Score the following on a 1-10 scale:
1. Movement quality: Does the robot move at all, or is it frozen/fallen?
2. Concept embodiment: Does the movement evoke the concept?
3. Gait complexity: Is the movement pattern simple or complex?
4. Aesthetic interest: Is this movement interesting to watch?

For each score, give a one-sentence justification.
Output as JSON: {"movement": N, "embodiment": N, "complexity": N, "aesthetic": N, "notes": "..."}
```

#### Feedback Loop

The VLM scores become part of the few-shot context for the next round of LLM weight generation:

```
Previous attempts for this concept:
  Attempt 1: weights={...}, VLM scored movement=2/10 (robot fell over), embodiment=1/10
  Attempt 2: weights={...}, VLM scored movement=7/10 (steady walk), embodiment=4/10

Now generate weights that improve on these attempts.
```

This is gradient descent through natural language — the VLM provides the error signal, the LLM provides the update rule, and the physics engine is the forward pass.

### 6. Data Management on VACC

```
$HOME/Evolutionary-Robotics/          # Code (git repo)
$SCRATCH/gait_zoo/                    # Large data
    weights/                          # LLM-generated weight JSONs
    results/                          # Simulation results + analytics
    videos/                           # Rendered videos (GB scale)
    vlm_scores/                       # VLM scoring results
    models/                           # LLM/VLM model weights
    oeis_cache/                       # OEIS sequence cache
```

$SCRATCH is typically large (TB) but ephemeral (purged after 30-60 days). Important results should be copied to $HOME or pushed to git.

### 7. New Scripts Needed for VACC

| Script | Purpose |
|---|---|
| `vacc_common.py` | Shared utilities: vLLM/Ollama adapter, SLURM-aware batch splitting, scratch path management |
| `generate_weights_batch.py` | Tier 1: Process a batch of seeds through an LLM, write weight JSONs |
| `simulate_batch.py` | Tier 2: Load weight JSONs, run headless sims, write results |
| `render_and_score_batch.py` | Tier 3: Render videos from results, optionally score with VLM |
| `collect_results.py` | Aggregate batch results into single experiment JSON |
| `submit_experiment.py` | Orchestrate: submit Tier 1 → wait → submit Tier 2 → wait → submit Tier 3 |

### 8. Scale Estimates

| Experiment | Trials | Tier 1 (GPU-hr) | Tier 2 (CPU-hr) | Tier 3 (GPU-hr) |
|---|---|---|---|---|
| OEIS (99 seqs × 4 models) | 396 | ~0.5 | ~2 | ~1 |
| Characters (2000 × 4 models) | 8,000 | ~10 | ~40 | ~20 |
| Stith Thompson (~200 × 4 models) | 800 | ~1 | ~4 | ~2 |
| TV Tropes (~200 × 4 models) | 800 | ~1 | ~4 | ~2 |
| VLM feedback loop (1000 × 5 rounds) | 5,000 | ~6 | ~25 | ~30 |
| **Total** | **~15,000** | **~19** | **~75** | **~55** |

This is modest by VACC standards. The bottleneck is LLM inference (Tier 1), not simulation.

### 9. Migration Path

**Phase 1 (now):** Run everything on Mac Mini. Prove the pipeline works. Generate initial results for papers/presentations.

**Phase 2 (VACC CPU-only):** Move simulation (Tier 2) to VACC. Keep LLM inference on Mac Mini or use Ollama on a VACC GPU node. This immediately parallelizes the bottleneck (16-hour serial runs → <1 hour).

**Phase 3 (VACC full):** Deploy vLLM on GPU nodes. Run all three tiers on VACC. Add VLM scoring loop.

**Phase 4 (VLM feedback):** Close the loop — VLM scores feed back into LLM prompts. This is the "gradient descent through natural language" endgame.

---

## Source

- OEIS: https://oeis.org (API: /search?q=id:AXXXXXX&fmt=json)
- Stith Thompson Motif-Index: public domain, digital versions at various folklore archives
- TV Tropes: tvtropes.org (curated selection, no bulk scraping)
- VACC: https://www.uvm.edu/vacc
- VLM survey: artifacts/video_parsing_tools_notes.md
