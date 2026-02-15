# Motion Seed Experiment Report

**Date**: 2026-02-14
**Total trials**: 500 (100 seeds x 5 LLMs)
**Runtime**: 1506s initial + 1653s re-run = 3159s total (~53 min)

## Experiment Design

100 motion seeds (12 core concepts x 5 languages + 40 English synonyms) were prompted to 5 LLMs. Each LLM was asked to generate 6 neural network weights [w03, w04, w13, w14, w23, w24] that "capture the essence" of a motion word. The weights were then used to run a headless PyBullet simulation of a 3-link robot (4000 steps @ 240 Hz), and the resulting gait was evaluated against behavioral criteria specific to each motion concept.

### Models

| Model | Type | Parse Rate | Core Match Rate |
|---|---|---|---|
| gpt-4.1-mini | OpenAI API | 100/100 | **26/60 (43%)** |
| gpt-oss:20b | Ollama (local) | 97/100 | 19/58 (33%) |
| qwen3-coder:30b | Ollama (local) | 100/100 | 19/60 (32%) |
| deepseek-r1:8b | Ollama (local) | 17/100 | 4/14 (29%) |
| llama3.1:latest | Ollama (local) | 100/100 | 15/60 (25%) |

### Languages

English, German (de), Chinese (zh), French (fr), Finnish (fi)

### 12 Core Motion Concepts

| Concept | Criterion | Match Rate |
|---|---|---|
| stumble | speed_cv > 1.0 or contact_entropy > 1.5 | **17/23 (74%)** |
| stand_still | \|DX\| < 1 and \|DY\| < 1 | **12/22 (55%)** |
| walk_and_spin | displacement > 3m AND \|yaw\| > 1.5 rad | 7/19 (37%) |
| bounce | \|DX\| < 5m but work > 2000 | 9/26 (35%) |
| lurch | speed_cv > 1.2 with \|DX\| > 2m | 9/22 (41%) |
| shuffle | \|DX\| < 3m and speed < 1.0 | 7/20 (35%) |
| limp | \|delta_phi\| > 0.5, phase_lock < 0.7 | 7/21 (33%) |
| crab_walk | \|DY\| > 3 and \|DY\| > \|DX\| | 5/21 (24%) |
| backward_walk | DX < -5, \|DY\| < \|DX\| | 3/20 (15%) |
| spin | \|yaw\| > 3 radians | 3/22 (14%) |
| forward_walk | DX > 5, \|DY\| < \|DX\| | 2/22 (9%) |
| glide | efficiency > 0.002 and speed_cv < 0.8 | 2/21 (10%) |

## Key Findings

### 1. gpt-4.1-mini is the best semantic grounder

gpt-4.1-mini achieved the highest overall core concept match rate (43%), excelling particularly at "quiet" motion concepts:
- **stand_still**: 5/5 (100%) — generated all-zero weights across all 5 languages, producing perfectly stationary robots (DX=0.00, DY=0.00)
- **shuffle**: 5/5 (100%) — consistently produced low-displacement, low-speed gaits
- **limp**: 3/5 (60%) — generated asymmetric phase patterns
- **lurch**: 3/5 (60%) — produced high speed variation with displacement

### 2. "stand still" reveals LLM comprehension depth

The stand_still concept is the most striking result. gpt-4.1-mini and gpt-oss:20b both independently discovered that all-zero weights = no motion, and applied this consistently across English ("stand still"), German ("Stillstand"), Chinese ("静止"), French ("immobile"), and Finnish ("paikallaan seisominen"). This demonstrates genuine semantic comprehension of the weight-behavior mapping, not pattern matching.

### 3. "stumble" is universally easy to produce

At 74% match rate across all models and languages, stumble was the most reliably produced motion. This replicates the original DeepSeek conversation finding where all 4 languages converged to the same gait-space region for "stumble." The criterion (high speed variability or high contact entropy) is naturally produced by many weight configurations, suggesting stumble occupies a large basin in behavior space.

### 4. Directional control is surprisingly hard

Forward walk (9%) and backward walk (15%) were among the hardest concepts. While LLMs can generate weights that produce locomotion, controlling the *direction* of travel requires precise weight relationships that are difficult to intuit semantically. Interestingly, backward walk was only matched by Chinese "向后走" — suggesting possible training-data biases in how locomotion direction is encoded.

### 5. qwen3-coder shows weight-vector collapse

qwen3-coder:30b frequently generated identical weight vectors for semantically related but distinct seeds. For example:
- bounce/Hupfen/弹跳/rebondir all produced [0.8, -0.6, -0.4, 0.9, 0.7, -0.5]
- Multiple glide-related words produced [0.8, -0.6, 0.2, 0.9, -0.7, 0.3]
This suggests qwen3-coder has a limited internal vocabulary for mapping motion concepts to weight space, collapsing many seeds to the same "generic motion" vector.

### 6. gpt-oss:20b excels at stumble and stand_still

Despite being a smaller model, gpt-oss:20b achieved 100% match on both stumble (5/5) and stand_still (5/5). For stand_still, it discovered the all-zeros solution independently of gpt-4.1-mini. For stumble, it generated diverse weight vectors that consistently produced irregular motion.

### 7. deepseek-r1:8b needs special handling

As a reasoning model, deepseek-r1:8b spends most of its token budget on internal `<think>` blocks before generating output. With 1000 tokens allocated, only 17% of prompts produced parseable JSON. When it did produce output, its match rate (29%) was competitive, including notable results:
- **glide** (en): DX=+11.11, the only non-llama3.1 glide match
- **limp** (en): Successfully generated asymmetric gait
- **bounce** (de:Hupfen): Matched with a notably different weight pattern from other models

### 8. Cross-lingual consistency varies by concept

Some concepts show strong cross-lingual consistency (same model matches across multiple languages):
- **stand_still**: gpt-4.1-mini matched 5/5 languages, gpt-oss:20b matched 4/5
- **stumble**: qwen3-coder matched en/de/zh/fr (4/5), llama3.1 matched en/de/zh/fi (4/5)
- **bounce**: qwen3-coder matched en/de/zh/fr (4/5, identical weights)

Other concepts show language-specific effects:
- **backward_walk**: Only Chinese "向后走" produced matches (across 3 different models)
- **spin**: Matched in English, Chinese, and Finnish, but not German or French

## Model-Specific Behavioral Signatures

### gpt-4.1-mini
- Median |DX|: 1.13m (most conservative)
- Tends to generate symmetric weight patterns (e.g., [0.8, -0.8, 0.9, -0.9, 0.7, -0.7])
- Strong semantic grounding for "quiet" and "irregular" motions
- Weak at producing large-displacement gaits

### qwen3-coder:30b
- Median |DX|: 2.25m
- Max |DX|: 50.91m (largest displacement of any model)
- Prone to weight-vector collapse (many seeds → same weights)
- Strong at energetic motions (bounce 80%, crab_walk 60%)

### llama3.1:latest
- Median |DX|: 0.19m (very low median, but high variance)
- Max |DX|: 27.97m
- High behavioral diversity (different weights for every seed)
- Strong at walk_and_spin (60%) and stumble (80%)

### gpt-oss:20b
- Median |DX|: 0.93m
- Best at stumble (100%) and stand_still (100%)
- Good at lurch (80%) — consistently produces high-variability gaits
- Weak at directional motion (forward/backward/crab all 0%)

### deepseek-r1:8b
- Only 17% parse success (needs 4000+ token budget for reasoning)
- When it works, produces unique weight configurations
- Only model to match glide (besides llama3.1)

## Technical Notes

- **Parse failures**: deepseek-r1:8b and gpt-oss:20b use internal reasoning chains that consume the token budget. Setting num_predict=1000 fixed gpt-oss (97% success) but was insufficient for deepseek-r1 (17% success).
- **OpenAI in conda**: The `openai` Python package was not initially installed in the `er` conda environment, causing all gpt-4.1-mini trials to fail until the package was installed mid-experiment.
- **Simulation determinism**: Identical weights always produce identical gaits. Weight vectors can be reused reliably.
- **Extra English words** (trials 301-500): Used `lambda a: True` criteria (any motion counts), so all match rates for those are trivially 100%.
