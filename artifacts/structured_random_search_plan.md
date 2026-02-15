# Structured Random Search Experiment

## Hypothesis

The persona effectiveness theory predicts that structured weight generation (via LLM-mediated conceptual seeds) should outperform uniform random search. But does the *type* of structure matter? Different conceptual domains encode different kinds of structural regularity — verbs encode action and transformation, theorems encode formal relationships, scripture encodes narrative and metaphor, place names encode geographic and cultural associations. If persona effectiveness depends on the structural properties of the seed rather than its content, different seed domains should produce measurably different gait quality distributions.

## Experimental Design

Four structured random search scripts, each generating 100 gaits, plus a uniform random baseline (100 gaits from random_search_500.py's method). Total: 500 gaits across 5 conditions.

### Condition 1: Random Verbs (`structured_random_verbs.py`)

1. Select a random verb from any language (use a multilingual verb list or LLM generation)
2. Prompt the LLM: "Given the verb '[verb]' ([language]), generate 6 synapse weights (w03, w04, w13, w14, w23, w24) in [-1, 1] for a 3-link walking robot. Translate the action/motion quality of this verb into weight magnitudes, signs, and symmetry patterns. Return only the 6 numbers."
3. Run the simulation, compute Beer-framework analytics
4. Record: verb, language, weights, all scalar metrics

Why verbs: Verbs encode action, motion, transformation — the closest semantic domain to locomotion. If any word class maps naturally to gait dynamics, it should be verbs. Verbs in different languages may encode different aspectual structures (perfective/imperfective, telic/atelic) that could map to different temporal dynamics.

### Condition 2: Random Theorems (`structured_random_theorems.py`)

1. Select a random mathematical theorem (from a curated list or LLM generation)
2. Prompt the LLM: "Given the theorem '[theorem name]', generate 6 synapse weights (w03, w04, w13, w14, w23, w24) in [-1, 1] for a 3-link walking robot. Translate the structural principle of this theorem — its symmetries, invariants, fixed points, transformations — into weight patterns. Return only the 6 numbers."
3. Run simulation, compute analytics
4. Record: theorem name, weights, all scalar metrics

Why theorems: Theorems encode formal structural relationships — symmetry, conservation, fixed points, mappings. These are the closest semantic domain to the actual mathematics of dynamical systems. The Noether gait (symmetry → conservation) and Fibonacci gaits (ratio → weight scaling) already demonstrate this channel works.

### Condition 3: Random Bible Verses (`structured_random_bible.py`)

1. Select a random Bible verse (uniform over the ~31,000 verses)
2. Prompt the LLM: "Given the verse '[book chapter:verse] — [text]', generate 6 synapse weights (w03, w04, w13, w14, w23, w24) in [-1, 1] for a 3-link walking robot. Translate the imagery, action, and emotional quality of this verse into weight patterns. Return only the 6 numbers."
3. Run simulation, compute analytics
4. Record: verse reference, text, weights, all scalar metrics

Why Bible verses: Scripture is narratively dense, metaphorically rich, and structurally varied — ranging from action ("and the walls came tumbling down") to stillness ("be still and know") to repetition ("holy, holy, holy") to enumeration ("twelve tribes," "seven seals"). It's a maximally diverse text corpus compressed into short units. It also provides a test of whether narrative/metaphorical structure transfers differently than formal/mathematical structure.

### Condition 4: Random Place Names (`structured_random_places.py`)

1. Select a random place name (cities, geographic features, from a global list)
2. Prompt the LLM: "Given the place '[place name]', generate 6 synapse weights (w03, w04, w13, w14, w23, w24) in [-1, 1] for a 3-link walking robot. Translate the character of this place — its terrain, climate, culture, energy — into weight patterns. Return only the 6 numbers."
3. Run simulation, compute analytics
4. Record: place name, weights, all scalar metrics

Why place names: Places encode embodied, multisensory associations — terrain implies movement style, climate implies energy, culture implies rhythm. "Reykjavik" and "Mumbai" and "Mariana Trench" should produce very different weight vectors. This tests whether the LLM's geographic/cultural associations produce locomotion-relevant structure.

### Condition 5: Uniform Random (baseline)

100 trials with weights drawn uniformly from [-1, 1]^6, no LLM mediation. Same method as random_search_500.py. This is the null hypothesis: structure doesn't matter, any sampling method is equivalent.

## Implementation

### Shared infrastructure

All four structured scripts share:
- The `run_trial_inmemory()` function from random_search_500.py (headless sim + Beer analytics)
- The `write_brain()` function for brain.nndf generation
- Same simulation parameters (SIM_STEPS=4000, DT=1/240, MAX_FORCE=150)
- Same output format: JSON with per-trial seed, weights, and scalar metrics

### LLM weight generation

Each script calls the LLM (via API or local model) with a structured prompt that:
1. Names the seed (verb, theorem, verse, place)
2. Asks for exactly 6 float values in [-1, 1]
3. Specifies the weight names (w03, w04, w13, w14, w23, w24)
4. Requests the structural translation (not random numbers)
5. Parses the 6 floats from the response

Fallback: if the LLM returns unparseable output, retry once. If still unparseable, skip that trial and log the failure.

### Seed selection

- **Verbs**: Curate a list of ~500 verbs across 10+ languages (including motion verbs, state verbs, transformation verbs). Or let the LLM generate a random verb on each trial.
- **Theorems**: Curate a list of ~200 theorems spanning algebra, geometry, analysis, topology, number theory, combinatorics.
- **Bible verses**: Uniform random selection from the ~31,000 verses (KJV or similar public domain text). Can use a simple index.
- **Places**: Curate a list of ~500 place names spanning continents, terrain types, scales (cities, mountains, rivers, deserts, islands).

## Metrics and Comparison

### Primary metrics (per gait)
- |DX| (displacement magnitude)
- Total distance (sqrt(DX^2 + DY^2))
- Mean speed
- Efficiency (distance per work)
- Phase lock score
- Contact entropy
- Attractor type classification

### Comparison statistics (per condition, 100 gaits each)
- Dead fraction (|DX| < 1m)
- Median |DX|
- Max |DX| (best discovery)
- Mean efficiency
- Phase lock distribution (bimodality test)
- Unique attractor types discovered
- Behavioral diversity (spread in Beer-framework metric space)

### Key questions
1. **Does structured search beat random?** Do all four structured conditions have lower dead fractions and higher median |DX| than uniform random?
2. **Does the domain matter?** Do verbs outperform theorems, or vice versa? Does the closest semantic domain to locomotion (verbs) win?
3. **Does structural formality help?** Do theorems (formal structure) outperform Bible verses (narrative structure) or places (associative structure)?
4. **What kind of structure transfers best?** Which condition produces the most diverse behavioral repertoire (spread across gaitspace)?
5. **Are there interaction effects?** Do certain seed types preferentially produce certain attractor types, motifs, or behavioral tags?

## Output

### Per-script outputs
- `artifacts/structured_random_verbs.json` — 100 trials with verb, language, weights, metrics
- `artifacts/structured_random_theorems.json` — 100 trials
- `artifacts/structured_random_bible.json` — 100 trials
- `artifacts/structured_random_places.json` — 100 trials
- `artifacts/structured_random_baseline.json` — 100 trials (uniform random)

### Comparison outputs
- `artifacts/structured_random_comparison.json` — aggregated statistics per condition
- `artifacts/plots/sr_fig01_dx_by_condition.png` — box/violin plot of |DX| across 5 conditions
- `artifacts/plots/sr_fig02_dead_fraction.png` — bar chart of dead fractions
- `artifacts/plots/sr_fig03_phase_lock_by_condition.png` — phase lock distributions overlaid
- `artifacts/plots/sr_fig04_speed_efficiency.png` — speed vs efficiency scatter, color-coded by condition
- `artifacts/plots/sr_fig05_best_of_n.png` — best-of-N curves for each condition
- `artifacts/plots/sr_fig06_diversity.png` — PCA of Beer-framework metrics, points colored by condition

### Comparison script
`structured_random_compare.py` — loads all 5 JSON files, computes comparison statistics, generates all comparison plots, runs statistical tests (Mann-Whitney U for pairwise condition comparisons).

## Expected runtime

- LLM calls: ~400 total (100 per structured condition), ~1-2 seconds each = ~10 minutes
- Simulations: 500 total, ~0.1s each = ~50 seconds
- Total: ~15 minutes
