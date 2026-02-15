# Cliff Analysis: Local Topology of the Fitness Landscape

**Date**: 2026-02-12
**Script**: `random_search_cliffs.py`
**Data**: `artifacts/random_search_cliffs.json`
**Figures**: `artifacts/plots/cliff_fig01_*` through `cliff_fig05_*`

## Experiment

50 random base points in 6D weight space. For each, 10 random perturbations at each of 3 radii (0.05, 0.1, 0.2) along random unit vectors in 6D. Same perturbation directions used across radii for comparability. Total: 1,550 simulations in 144.6s. Each simulation runs the full 4000-step PyBullet episode and computes Beer-framework analytics in memory.

## Key Findings

### 1. The landscape is cliff-riddled

At the smallest perturbation radius (r=0.05) — nudging each weight by at most 5% of its full [-1, 1] range:

| Metric | r=0.05 | r=0.1 | r=0.2 |
|---|---|---|---|
| Mean |delta_DX| | 5.16m | 5.94m | 6.92m |
| Median |delta_DX| | 2.88m | 3.77m | 4.42m |
| P90 |delta_DX| | 12.63m | 14.45m | 16.56m |
| Max |delta_DX| | 43.42m | 51.99m | 43.07m |

A 0.05 perturbation in weight space produces a median 2.88m shift in displacement and a mean 5.16m shift. One in seven perturbations (15.4%) causes a >10m shift. The maximum observed was 43.4m — the robot went from one behavioral regime to a completely different one from a tiny weight nudge.

### 2. Almost every point has a cliff nearby

| Threshold | r=0.05 | r=0.1 | r=0.2 |
|---|---|---|---|
| Any neighbor with |delta_DX| > 5m | 80% (40/50) | 88% (44/50) | 92% (46/50) |
| Any neighbor with |delta_DX| > 10m | 42% (21/50) | 56% (28/50) | 70% (35/50) |
| Any neighbor with |delta_DX| > 20m | 12% (6/50) | 20% (10/50) | 28% (14/50) |

At r=0.05, 80% of random points have at least one direction where a small step causes a >5m behavioral shift. At r=0.1, more than half have a >10m cliff nearby. Cliffs are not rare features — they are the norm.

### 3. The gradient is non-smooth (steeper at small scales)

| Radius | Mean |grad| (|delta_DX|/r) | Median |grad| |
|---|---|---|---|
| 0.05 | 103.1 | 57.6 |
| 0.10 | 59.4 | 37.7 |
| 0.20 | 34.6 | 22.1 |

If the landscape were smooth, the gradient magnitude would be roughly constant across scales. Instead, it is 3x steeper at r=0.05 than at r=0.2. This indicates fractal-like roughness: the closer you look, the more structure you find. The landscape is not a smooth bowl with occasional cliffs — it is rough at every scale.

### 4. Implications

- **For the hill climber**: Greedy small-step optimization is dangerous. A step that improves fitness in one direction may be adjacent to a cliff in another. The hill climber will need to be robust to this — perhaps by testing multiple directions, using small step sizes, or accepting occasional large regressions.

- **For evolution**: The landscape's roughness at small scales means that mutation operators in evolutionary search are constantly at risk of crossing behavioral boundaries. This may actually help exploration (large jumps to new regimes) while hindering exploitation (fine-tuning within a regime).

- **For the bifurcation finding**: The gaits 43/44 divergence (same topology, opposite behavior) is not a special case. 42% of random points have a >10m cliff within r=0.05. Bifurcations are the default topology of this landscape.

- **For controller design**: The non-smoothness suggests that gradient-based optimization (backpropagation through the simulator) would struggle. The landscape may be formally non-differentiable almost everywhere.

## Figures

| Figure | File | Description |
|---|---|---|
| 1 | `cliff_fig01_delta_dx_by_radius.png` | |delta_DX| distributions overlaid for 3 radii |
| 2 | `cliff_fig02_cliff_probability.png` | Cliff probability vs radius at 3 thresholds |
| 3 | `cliff_fig03_landscape_profiles.png` | 1D transects through weight space for 8 base points |
| 4 | `cliff_fig04_cliff_vs_base_dx.png` | Worst cliff vs base |DX| and base phase lock |
| 5 | `cliff_fig05_worst_cliff_histogram.png` | Distribution of worst-cliff-per-base at each radius |


# Notable Gaits from 500 Random Search Trials

**Date**: 2026-02-12
**Script**: `random_search_500.py`
**Data**: `artifacts/random_search_500.json`
**Figures**: `artifacts/plots/rs_fig01_*` through `rs_fig07_*`

## The Stars

### Trial 3 — "The Accidental Masterpiece"

| Metric | Value | Zoo Comparison |
|---|---|---|
| DX | +25.875m | 52% of CPG Champion (50.1m) |
| Efficiency | 0.01340 | **1.7x CPG Champion** (0.00773), 7.8x Curie (0.00172) |
| Speed | 1.319 | 40% of CPG Champion (3.33) |
| Phase Lock | 0.989 | Comparable to CPG Champion (0.998) |
| Work Proxy | 1956.4 | 30% of CPG Champion (6482) |

Weights: `w03=-0.597  w04=-0.424  w13=+0.112  w14=-0.005  w23=+0.297  w24=+0.214`

The most efficient gait found in the entire random search — 1.7x more efficient than the CPG Champion. Achieves this with a standard 6-synapse network, no hidden layer. Nearly perfect phase lock (0.989) and very low energy expenditure (work=1956, vs CPG Champion's 6482). Travels half the distance of the CPG Champion but at far lower cost per meter.

This weight vector is a strong candidate for seeding a hill climber targeting efficient locomotion.

### Trial 41 — "The Backward Rocket"

| Metric | Value |
|---|---|
| DX | -39.375m |
| Efficiency | 0.00971 |
| Speed | 3.200 |
| Phase Lock | 0.980 |

Faster and more efficient than the forward champion (trial 24). The backward direction is not inherently worse — this gait found a high-performance backward walking regime.

### Trial 24 — "The Distance King"

| Metric | Value |
|---|---|
| DX | +41.010m |
| Efficiency | 0.00369 |
| Speed | 2.556 |
| Phase Lock | 0.348 |

81.8% of the CPG Champion's displacement with a standard 6-synapse network. Low phase lock suggests a non-periodic, possibly chaotic trajectory that nonetheless covers distance efficiently.

Weights: `w03=-0.198  w04=-0.233  w13=-0.883  w14=+0.232  w23=-0.863  w24=-0.355`

## The Oddities

### Trial 272 — "The Synchronized Statue"

- Phase lock = **1.000** (perfect), DX = -1.491m
- The legs are in perfect temporal coordination... producing almost no locomotion
- Demonstrates that phase locking and displacement are independent: you can be perfectly coordinated and go nowhere

### Trial 168 — "The Random Spinner"

- Yaw = -16.688 rad (2.7 full rotations), DX = +0.462m
- Found a spinning behavior by pure chance
- Analogous to the zoo's Spinner Champion (gait 44) but discovered independently through random search

### Trial 261 — "The Entropy King"

- Entropy = 2.316 (highest of all 500), roll dominance = 0.472 (lowest of all 500)
- The most complex ground-interaction pattern and the least body-dominated rotation
- Close to the theoretical maximum entropy of log2(8) = 3.0 bits

### Trial 426 — "The Minimalist"

- Entropy = 0.233 (lowest of all 500)
- Uses essentially one contact state almost all the time — a nearly static, very simple periodic behavior

### Trial 187 — "The Chaotic Walker"

- Phase lock = 0.013 (lowest of all 500)
- Essentially zero phase coherence between joints — pure temporal chaos

## Pareto Frontier (|DX| vs Efficiency)

Only 4 out of 500 trials are Pareto-optimal:

| Trial | DX | Efficiency | Speed | Phase Lock |
|---|---|---|---|---|
| 24 | +41.010 | 0.00369 | 2.556 | 0.348 |
| 18 | +33.036 | 0.00534 | 2.109 | 0.875 |
| 41 | -39.375 | 0.00971 | 3.200 | 0.980 |
| 3 | +25.875 | 0.01340 | 1.319 | 0.989 |

The frontier is extremely thin — 99.2% of random gaits are dominated. This mirrors the zoo's Pareto frontier structure (Fuller → CPG Champion), suggesting the thinness of the frontier is a property of the body, not of the search method.

## Zoo Comparison

| Benchmark | Count | Fraction |
|---|---|---|
| Random gaits beating Curie (|DX| > 23.93m) | 27 | 5.4% |
| Random gaits beating CPG Champion (|DX| > 50.11m) | 0 | 0% |

Curie's displacement (23.9m) is reachable by ~1 in 20 random draws. The CPG Champion (50.1m) appears unreachable by 6-synapse random search — its hidden-layer oscillator architecture accesses a region of behavioral space that the standard topology cannot.

## Research Leads

1. **Seed a hill climber from Trial 3**: Its exceptional efficiency (1.7x CPG Champion) combined with strong phase lock makes it an ideal starting point for optimization targeting the speed-efficiency frontier.

2. **Investigate the 6-synapse ceiling**: The best random 6-synapse gait reached 41m. The CPG Champion (50.1m) uses a hidden layer. Is this gap a fundamental topological limit? Could a hill climber close it?

3. **Trial 272 as a control experiment**: A gait with perfect phase lock but no locomotion challenges the assumption that coordination implies performance. What makes some locked gaits walk and others stand still?
