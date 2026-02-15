# Archetypometrics Experiment — Findings

## What We Ran
2,000 fictional characters from 341 stories (UVM Archetypometrics dataset) seeded into
the Sem→Wt→Beh pipeline: character name → Ollama (qwen3-coder:30b, temp=1.5) → 6 synapse
weights → ±0.05 deterministic perturbation → headless PyBullet simulation → Beer analytics.

---

## Key Findings

### 1. The LLM sees fiction through 3 lenses (not 64)

Of 64 possible sign patterns over 6 weights, the LLM used only **3 dominant patterns** for
95.4% of all 2,000 characters:

| Sign pattern | Count | % | Behavioral tendency |
|---|---|---|---|
| `+-++-+` | 891 | 44.5% | Strong leftward walkers (mean DX = -11.3m, 30% champions) |
| `+-+-+-` | 574 | 28.7% | Weak rightward drift (mean DX = +1.2m, 0% champions) |
| `+--++-` | 444 | 22.2% | Mixed/dead (mean DX ≈ 0, 21% dead) |

A 4th pattern (`-++--+`, 2.8%) is rare but interesting — it's the only minority pattern
that actually walks well (mean |DX| = 9.2m).

**Interpretation**: The LLM coarse-grains 2,000 distinct fictional identities into
essentially 3 archetypal "movement personalities." The sign structure determines
*which direction* the robot walks (or whether it walks at all).

### 2. Perturbation proves the landscape is cliff-riddled

Within a single archetype cluster (same LLM output, ±0.05 perturbation only):
- DX ranges span **up to 57m** (from -33.6m to +24.6m within one cluster)
- Within-archetype DX standard deviation reaches **13.4m** for the `+-++-+` clusters
- **27% of total behavioral variance** comes from ±0.05 perturbation alone

This is direct evidence for the cliff-riddled landscape discovered in earlier experiments:
a 5% nudge in weight space can flip a champion into a dead gait or reverse its direction.

### 3. Fictional characters ≈ random baseline in performance

| Condition | n | Unique WV | Mean |DX| | Dead % | Champions % |
|---|---|---|---|---|---|
| places | 100 | 4 | 1.34m | 0.0% | 0.0% |
| bible | 100 | 9 | 2.35m | 0.0% | 1.0% |
| theorems | 95 | 15 | 3.25m | 8.4% | 0.0% |
| verbs | 100 | 18 | 3.18m | 5.0% | 1.0% |
| **baseline** | **100** | **100** | **7.65m** | **8.0%** | **7.0%** |
| **characters** | **2000** | **2000** | **7.66m** | **15.4%** | **13.5%** |

Characters (with perturbation) match the random baseline almost exactly in mean |DX|
(7.66m vs 7.65m). The perturbation effectively randomizes within archetype clusters,
producing baseline-like coverage. But the 15.4% dead rate is higher than baseline's 8.0%,
because the LLM's `+--++-` archetype (22% of characters) is intrinsically prone to
producing dead gaits.

### 4. The LLM *does* differentiate heroes from villains

Tested hero/villain pairs from the same story:

| Hero | Villain | Same sign? | Weight dist |
|---|---|---|---|
| Harry Potter | Lord Voldemort | **DIFFERENT** | 2.29 |
| Luke Skywalker | Darth Vader | **DIFFERENT** | 2.38 |
| Frodo Baggins | Gollum | **DIFFERENT** | 1.71 |
| Walter White | Hank Schrader | **DIFFERENT** | 2.09 |
| Dolores Abernathy | Man in Black | **DIFFERENT** | 1.87 |
| Jon Snow | Cersei Lannister | SAME | 0.42 |

5 of 6 testable hero/villain pairs received **different sign patterns** — the LLM
assigns them to different archetypal movement categories. The weight distances (1.7–2.4)
are large, confirming the LLM sees moral/narrative polarity even in 6-dimensional
weight space.

### 5. Stories predict sign clustering — weakly

- Within-story character pairs share the same sign pattern **39.4%** of the time
- Expected by chance (given marginal frequencies): **33.1%**
- Stories do bias toward particular archetypes, but the effect is modest (+6.3pp)

The strongest story-level clustering: Twin Peaks (93% `+-++-+`), This Is Us (90% `+-++-+`),
Mad Men (73% `+-++-+`), Downton Abbey (73% `+-++-+`). These are character-driven dramas
where the LLM apparently sees the ensemble as temperamentally similar.

The most diverse stories: Harry Potter (7 sign patterns across 30 characters), Game of
Thrones (4 patterns), Breaking Bad (4 patterns). These are stories with morally diverse casts.

### 6. w03 is frozen, w23/w24 are the "personality" axes

Weight-by-weight analysis:
- **w03**: 96.3% positive, mean +0.66, virtually no variance. The LLM *always* excites
  the Torso→BackLeg_Motor connection positively. This is a structural prior, not a
  character-specific choice.
- **w04**: 96.3% negative, mean -0.44. Similarly frozen (Torso→FrontLeg_Motor = inhibitory).
- **w13, w14**: Bimodal but skewed. These are the "hero vs villain" switches.
- **w23, w24**: Near-symmetric, balanced +/-. These carry the most character-specific
  information and are where the 3 archetypes actually differ.

The LLM has learned a rigid prior about how the torso sensor should drive the motors
(w03 > 0, w04 < 0), and only varies the leg-sensor weights to express character identity.

### 7. 735 LLM-distinct vectors, not just 3

While the sign structure collapses to 3 patterns, the actual weight magnitudes vary more:
735 distinct vectors (at 1-decimal-place resolution). The LLM *does* modulate intensity
within each archetype — it just doesn't cross sign boundaries often.

Combined with ±0.05 perturbation, this produces 2000/2000 unique vectors and 1,409 unique
DX values, demonstrating that the pipeline achieves full individuation when perturbation
is applied.

---

## Methodological Lessons

1. **Example anchoring is lethal**: Including `{"w03": 0.5, ...}` in the prompt collapsed
   2000 characters to 15 weight vectors. Removing example values was necessary but not
   sufficient.

2. **Temperature is the primary diversity knob**: temp 0.8 → 15 unique; temp 1.5 → 274
   unique (before perturbation). Prompt engineering mattered less than temperature.

3. **Perturbation completes the job**: The LLM picks the archetype; perturbation
   individuates within it. On a cliff-riddled landscape, this produces genuinely distinct
   behaviors. 274 → 2000 unique vectors, 184 → 1409 unique DX values.

4. **The 3-run progression** (15 → 274 → 2000 unique vectors) is itself a finding about
   how LLMs interact with structured generation tasks: small changes in prompt/temperature
   produce order-of-magnitude differences in output diversity.
