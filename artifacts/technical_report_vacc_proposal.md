# LLM-Mediated Locomotion: Current Results and VACC Scaling Proposal

**To:** Chris Danforth
**From:** Garden of Computation
**Date:** 2026-02-15
**Re:** Semantic probing of LLM→physics channel; infrastructure bottlenecks; VACC resource request

---

## 1. What We're Doing

We have built a pipeline that asks large language models to translate *semantic concepts* into neural network weights for a simulated walking robot:

```
Semantic seed → LLM → 6 synaptic weights → PyBullet simulation → Beer-framework gait analytics
```

The robot is a 3-link body (torso + two legs) with 2 hinge joints, 3 touch sensors, and 6 weighted synapses connecting sensors to motors. The weight space is small ([-1,1]^6) but behaviorally rich — our "Synapse Gait Zoo" catalogs 116 qualitatively distinct gaits discovered through evolutionary search.

The core question: **When an LLM translates "Hamlet" or "Fibonacci sequence" or "the Pythagorean theorem" into 6 numbers, does the resulting robot behavior preserve meaningful semantic structure?** If so, the LLM→weight channel is acting as a *functor* — a structure-preserving map from a semantic category to a behavioral category. If not, it's lossy compression that erases the distinctions we care about.

## 2. Current Results (Mac Mini M-series, Local Ollama)

### 2.1 Experiments Completed

| Experiment | Trials | Models | Key Finding |
|-----------|--------|--------|-------------|
| Motion verbs | 100 | 1 (qwen3-coder:30b) | Verbs like "sprint" vs "crawl" produce statistically distinct gaits. Semantically similar verbs cluster in weight space. |
| Random baseline | 100 | 1 | Uniform random weights produce a different behavioral distribution than LLM-mediated weights (p < 0.001, Mann-Whitney). |
| Theorems | 95 | 1 | Abstract mathematical theorems generate non-trivial weight structure; Pythagorean theorem consistently produces forward motion. |
| Bible verses | 100 | 1 | Narrative/moral content translates to weight patterns; verses about movement produce faster gaits than verses about stillness. |
| Place names | 100 | 1 | Geographic/cultural identity maps to behavioral signatures. |
| Celebrities | 132 | 1 | Public figures produce distinctive weight patterns. |
| Politics | 79 | 1 | Political figures and concepts produce ideologically-correlated behavioral clusters. |
| Archetypometrics (characters) | 2,000 | 1 | Fictional characters from 341 stories. Led to "death encoding" discovery (below). |

**Total completed trials:** ~2,800 (single-model conditions)

### 2.2 Experiments In Progress

| Experiment | Done/Total | Models | ETA |
|-----------|-----------|--------|-----|
| Character seed (multi-model) | 2,050/8,000 | 4 | ~12 hours |
| OEIS sequences | 200/396 | 4 | ~45 min |

### 2.3 Experiments Queued (curated, ready to run)

| Probe Type | Seeds Curated | Planned Trials | Description |
|-----------|--------------|---------------|-------------|
| Mathematicians | 104 | 416 (×4 models) | Real people whose identities fuse biography with mathematical structure |
| TV Tropes | 185 | 740 (×4 models) | Narrative/cultural tropes (HeroicSacrifice, LeParkour, Slapstick) |
| Stith Thompson Motif Index | 192 | 768 (×4 models) | Folk-literature motifs (D10: transformation to animal, K800: killing by deception) |
| Decollapse retries | ~1,300 | ~1,300 | Re-running collapsed weight vectors with enriched prompts |
| Deepseek retries | ~100 | ~100 | Parsing failures from reasoning model |

### 2.4 Key Findings So Far

**The Death Encoding.** Characters who die in their stories fall down more often in simulation. This emerged from the Romeo & Juliet cast:

| Character | Dies in Play | Falls (torso_duty > 0.3) | Models |
|-----------|-------------|-------------------------|--------|
| Juliet | Yes | 3/4 (75%) | qwen3, deepseek, gpt-oss |
| Romeo | Yes | 2/3 (67%) | llama3.1, gpt-oss |
| Mercutio | Yes | 2/4 (50%) | llama3.1, gpt-oss |
| Friar Laurence | No | 0/4 (0%) | — |
| Nurse | No | 0/4 (0%) | — |
| Benvolio | No | 0/3 (0%) | — |

LLM-annotated death labels across the full 2,000-character dataset (1,520 survive, 471 die) will allow statistical testing of this hypothesis at scale once the 8,000-trial experiment completes.

**Zero Sequence → Stillness.** When given OEIS A000004 (the all-zeros sequence), 3 of 4 models produce all-zero weights, yielding a perfectly still robot. The LLMs understand that "zero" means "nothing." Conversely, the Kolakoski sequence (self-describing, irregular) produces the fastest gaits across all models.

**Weight Collapse.** The most significant confound: 66% of successful character trials collapse into shared weight vectors. The top cluster — a single weight vector — is used 111 times for characters as diverse as Applejack, Saul Goodman, and Claire Standish. This is predominantly a problem with gpt-oss:20b (which produced the same weights for 69 different characters in one cluster) and qwen3-coder:30b. The 9-point weight grid and the small 6D space contribute to this, but the core issue is that the LLMs default to "generic reasonable walking weights" rather than encoding character-specific information. We have built a "decollapse" agent that retries collapsed trials with enriched prompts, but this needs compute time to run.

### 2.5 Supporting Infrastructure

Beyond the experiments themselves, we have built:

- **116-gait Synapse Gait Zoo** with full Beer-framework analytics (displacement, contact patterns, coordination metrics, rotation axis analysis)
- **Full-resolution telemetry** for all 116 gaits (4,000 timesteps each at 240 Hz)
- **Research campaigns:** walker competition (5 algorithms, 5,000 evals), causal surgery (brain transplants, 600 sims), behavioral embryology (gait emergence timing), gait interpolation, resonance mapping (2,150 sims), atlas of cliffiness (6,400 sims), cliff taxonomy, dark matter analysis
- **Categorical/formal structure:** functor validation (Sem→Wt→Beh), Fisher information metric, perturbation probing, Hilbert space formalization, Yoneda crosswired topology test
- **Automated agents:** LLM curation, cross-model verification, pest control (code quality), death annotation, decollapse

**Total simulations run to date: ~21,000**

## 3. The Bottleneck

Everything runs on a single Mac Mini with 4 local Ollama models:

| Model | Size | Role |
|-------|------|------|
| qwen3-coder:30b | 18 GB | Primary weight generator |
| gpt-oss:20b | 13 GB | Weight generator |
| deepseek-r1:8b | 5.2 GB | Weight generator (reasoning, high failure rate) |
| llama3.1:latest | 4.9 GB | Weight generator + verification/curation |

**The constraint is LLM inference throughput.** Ollama serves one request at a time. Each trial needs 1-3 seconds of LLM time. With 4 models per trial and model-swapping overhead:

- The 8,000-trial character experiment takes **~18 hours** wall-clock
- The 396-trial OEIS experiment takes **~1.5 hours**
- Concurrent experiments compete for the single inference queue

PyBullet simulation is cheap (~0.1s per trial in headless DIRECT mode). The LLM is 10-30× slower than the physics.

### What we cannot do on current hardware

1. **Scale to full probe taxonomy.** The planned experiments (mathematicians + TV Tropes + Stith Thompson + decollapse retries) add ~3,300 trials. At current throughput, that's another 2-3 days of continuous LLM inference, during which no other experiments can run.

2. **Run more/larger models.** Larger models (70B+) would likely reduce weight collapse but won't fit in memory alongside others. Cloud API models (GPT-4, Claude) would add model diversity but at significant cost for thousands of trials.

3. **Video generation and VLM scoring.** The next phase of the project involves generating videos of robot gaits and using vision-language models to score/describe them — closing the loop between LLM output and visual behavior. This requires GPU-accelerated rendering and VLM inference, neither of which is feasible on the Mac Mini.

4. **Evolutionary optimization from LLM seeds.** Our `llm_seeded_evolution.py` showed that LLM-generated starting points can accelerate evolutionary search. Scaling this to thousands of seeds requires parallel population evaluations.

## 4. VACC Resource Request

### 4.1 Architecture

The pipeline has three distinct compute profiles:

| Stage | Compute | Memory | I/O | Parallelism |
|-------|---------|--------|-----|-------------|
| **LLM inference** (weight generation) | GPU | 20-40 GB VRAM | Low | Batch (vLLM) |
| **Physics simulation** (PyBullet) | CPU only | < 1 GB | Minimal | Embarrassingly parallel |
| **VLM scoring** (future) | GPU | 20-40 GB VRAM | Video frames | Batch |

### 4.2 Proposed VACC Deployment

**Phase 1: CPU Simulation Farm (immediate value)**

Move PyBullet simulations to VACC CPU nodes via SLURM array jobs. Keep LLM inference on Mac Mini (or simple API calls).

```
Mac Mini (Ollama)  →  weight JSON files  →  VACC CPU array job  →  Beer analytics
```

- SLURM array of N simulation workers, each processing a batch of weight vectors
- Eliminates simulation time from the critical path entirely
- Requires: minimal VACC setup (Python + PyBullet, no GPU)
- Speedup: simulation is already fast (0.1s/trial), but removes it from the LLM contention

**Phase 2: vLLM on GPU Node**

Serve models via vLLM on a VACC GPU node. vLLM supports continuous batching — multiple concurrent requests to the same model with near-linear throughput scaling.

```
VACC GPU (vLLM: qwen3, llama3, etc.)  ←→  VACC CPU array (PyBullet sims)
```

- Single A100/H100 can serve a 30B model at ~50-100 tokens/sec with batching
- Multiple models can be served on separate GPUs or time-shared
- Estimated throughput: **10-50× current Mac Mini**, depending on GPU count
- The 8,000-trial character experiment drops from 18 hours to **~30-60 minutes**

**Phase 3: Full Pipeline (video + VLM)**

```
vLLM (weights) → CPU farm (PyBullet) → GPU render (video) → VLM (scoring/description) → feedback loop
```

- Offscreen PyBullet rendering on GPU nodes for video generation
- VLM (e.g., LLaVA, GPT-4V) scores/describes the resulting gait
- VLM descriptions become input for the next round of weight generation
- This closes the perception loop: LLM → physics → vision → LLM

### 4.3 Resource Estimates

For the immediate planned experiments (~15,000 trials across all probe types):

| Resource | Quantity | Duration | Notes |
|----------|----------|----------|-------|
| GPU node (A100 40GB) | 1 | ~19 hours total | vLLM serving, 4 models rotating |
| CPU cores | 32-64 | ~2 hours total | PyBullet array jobs (embarrassingly parallel) |
| Storage | ~5 GB | Persistent | Weight vectors, telemetry, analytics JSONs |

For the VLM feedback loop (future, ~15,000 videos):

| Resource | Quantity | Duration | Notes |
|----------|----------|----------|-------|
| GPU node (rendering) | 1 | ~25 hours | Offscreen PyBullet + ffmpeg, 6s video each |
| GPU node (VLM) | 1 | ~50 hours | Scoring + description at ~12s per video |

Total GPU budget for the full experimental program: **~100 GPU-hours** (modest by current standards).

### 4.4 Software Requirements

- Python 3.11 with PyBullet, numpy, matplotlib
- vLLM (for serving local models on GPU)
- Ollama model weights (qwen3-coder, llama3.1, deepseek-r1, gpt-oss) — can convert to vLLM format
- ffmpeg (for video encoding)
- No internet access needed during runs (all models local)

### 4.5 What This Enables

With VACC resources, the experimental program expands from the current ~3,000 completed multi-model trials to **~15,000+** within days rather than weeks. More importantly, it enables:

1. **Statistical power for the death encoding hypothesis.** With 8,000 character trials and 2,000 death-annotated characters, we can test whether narrative fate systematically maps to physical behavior across multiple LLM architectures.

2. **The full probe taxonomy.** Ten different semantic probe types, each testing a different layer of the LLM→physics channel, with enough trials per probe for statistical analysis.

3. **Model comparison at scale.** Four+ models generating weights for the same inputs, revealing which architectures preserve semantic structure and which collapse.

4. **The VLM feedback loop.** The most theoretically interesting direction: can a vision-language model, watching the robot walk, recover the semantic seed that generated its gait? If so, the Sem→Wt→Beh→Vision→Sem loop is a measurable channel with computable information capacity.

## 5. Summary

| | Current (Mac Mini) | With VACC |
|---|---|---|
| LLM throughput | ~1 trial/3s, sequential | ~10-50 trials/3s, batched |
| Simulation throughput | ~10 trials/s (limited by LLM queue) | ~100+ trials/s (parallel CPU) |
| Models available | 4 local (8B-30B) | 4+ local (8B-70B+) |
| Video generation | Not feasible | ~600/hour |
| VLM scoring | Not feasible | ~300/hour |
| Full experiment program | ~2-3 weeks | ~2-3 days |
| Total planned simulations | ~15,000 | ~15,000 (same science, faster) |

The Mac Mini has been remarkably productive for prototyping — 21,000 simulations, 54 artifact files, a catalog of discovered phenomena. But the bottleneck is now clear: we need parallel LLM inference to scale beyond proof-of-concept, and GPU rendering to close the vision loop. The VACC is the natural next step.
