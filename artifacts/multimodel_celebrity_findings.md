# Multi-LLM Celebrity Experiment — Findings

## Experiment
132 celebrity names from "Revenge of the Androids" (Cramer et al. 2025) run through
all 4 locally available LLMs via Ollama, same prompt, same temperature (1.5), same
deterministic perturbation (±0.05). Perturbation seed includes model name to prevent
artificial convergence.

## Models Tested
| Model | Size | Trials | Success | Notes |
|---|---|---|---|---|
| qwen3-coder:30b | 18 GB | 132 | 132 (100%) | Code-specialized, baseline model |
| llama3.1:latest | 4.9 GB | 132 | 129 (98%) | General-purpose (Meta) |
| gpt-oss:20b | 13 GB | 132 | 1 (1%) | Failed — JSON parsing issues |
| deepseek-r1:8b | 5.2 GB | 132 | 0 (0%) | Failed — chain-of-thought wrapping |

## Key Finding: Two LLMs Have Opposite Theories of Personhood

The two working models disagree on sign pattern **90% of the time**. Only 13/129
shared celebrities received the same sign pattern from both models.

| Metric | qwen3-coder:30b | llama3.1:8b |
|---|---|---|
| Sign patterns used | 4 | **19** |
| Dominant sign | `+-++-+` (66%) | `-+-+-+` (22%) |
| w03 mean | +0.72 (99% positive) | -0.18 (75% negative) |
| w04 mean | -0.46 (99% negative) | +0.38 (74% positive) |
| Mean \|DX\| | 5.83m | 7.86m |
| Median \|DX\| | 3.33m | 6.43m |
| Dead (<1m) | 15.2% | 10.9% |
| Champions (>20m) | 8.3% | 7.0% |
| Cliffiness (median) | 14.2 | 11.5 |

### The w03/w04 prior is completely inverted
Qwen3 has a rigid structural prior: w03 > 0, w04 < 0 for 99% of celebrities.
Llama3.1 inverts this: w03 < 0 (75%), w04 > 0 (74%). The two models have
mirror-image theories of how a torso sensor should drive leg motors.

### DX correlation between models: r = -0.06
Knowing what one model produces for a celebrity tells you *nothing* about what
the other will produce. The mapping from Name → Gait is model-specific, not
a property of the name itself.

### Per-celebrity disagreements
- Albert Einstein: Qwen DX = -20.4m, Llama DX = -31.2m (both walk, different signs)
- Donald Trump: Qwen `+-+-+-` (inert), Llama `-++-+-` (active)
- Cristiano Ronaldo: Qwen DX = -5.0m, Llama DX = +34.6m (champion, opposite direction)
- Ferdinand Marcos: Qwen DX = +24.7m, Llama DX = -21.3m (both champions, opposite direction)

### The 13 celebrities both models agree on (same sign pattern)
Abraham Lincoln, Adam Schiff, Charles Darwin, Chuck Schumer, Elon Musk,
Emmanuel Macron, and 7 others. These represent a "consensus" — celebrities
where different training distributions converge on the same archetypal mapping.

### Llama3.1 explores 4.75x more sign patterns
The smaller model (8B) uses 19 sign patterns where the larger (30B) uses only 4.
This inverts the naive expectation that bigger = more expressive. Possible
explanations: (1) larger models have stronger attractors from more training data,
(2) Qwen3's code-specialization makes it more formulaic, (3) the 30B model
has collapsed its sign-structure representation more aggressively.

## Questions That Emerge

1. **Is the functor F: Sem→Wt well-defined across models?** The categorical
   structure (functor, sheaf) may be a property of a specific model's training
   distribution, not of "language" in general. Two models produce completely
   different weight-space images from the same semantic input.

2. **Can we disentangle structural bias from semantic signal?** The frozen-weight
   prior (w03/w04) is structural, not celebrity-specific. If we subtract each
   model's mean weight vector, do the residuals correlate? This would separate
   "what the model thinks robots should do" from "what the model thinks Trump
   vs Einstein should do."

3. **Why is the smaller model more diverse?** 8B parameters → 19 sign patterns.
   30B parameters → 4 sign patterns. Is this about model capacity, training data
   volume, or architectural differences (code-tuning vs general)?

4. **The cliff landscape is model-independent, but models map to different parts.**
   Qwen3 concentrates at cliffiness 14.4 (2x atlas); Llama3.1 at 12.0 (1.6x).
   Both above baseline but in different neighborhoods.

5. **Test-retest reliability.** The 13 celebrities where both models agree are
   a natural reliability measure. Those names carry signal strong enough to
   override model-specific priors.

6. **Should we run fictional characters through llama3.1?** If the 3-archetype
   collapse is Qwen3-specific and Llama3.1 produces 19 patterns for fiction too,
   that changes the narrative about what the LLM "sees."

7. **deepseek-r1 and gpt-oss failures are data.** The reasoning model wraps output
   in chain-of-thought, breaking JSON extraction. The pipeline is fragile to
   instruction-following capability. Could we adapt the parser, or does the
   failure mode tell us something about how these models process the task?

## Data Files
- `multimodel_celebrities_qwen3-coder_30b.json` — 132 trials
- `multimodel_celebrities_llama3_1_latest.json` — 129 trials
- `multimodel_celebrities_gpt-oss_20b.json` — 1 trial (effectively failed)
- `multimodel_celebrities_deepseek-r1_8b.json` — 0 trials (failed)
- `multimodel_celebrities_v1_4gaits.json` — original 4-gait collapsed run (backup)
- `celebrity_vs_fictional_cliffiness.png` — comparison figure
- `celebrity_fictional_pca_annotated.png` — PCA overlay figure
- `celebrity_fictional_centroids.png` — centroid comparison by sign pattern
