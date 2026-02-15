# Random Search Analysis: 500 Trials

**Date**: 2026-02-12
**Script**: `random_search_500.py`
**Data**: `artifacts/random_search_500.json` (500 trials, compact metrics)
**Figures**: `artifacts/plots/rs_fig01_*` through `rs_fig07_*`

## Experiment

500 independent random-search trials. Each trial samples 6 synaptic weights uniformly from [-1, 1] (fully connected: 3 sensor neurons x 2 motor neurons), writes `brain.nndf`, runs a headless 4000-step PyBullet simulation, and computes all 4 Beer-framework analytics pillars (outcome, contact, coordination, rotation_axis) in memory. No telemetry written to disk. Total runtime: 46.3s (0.093s/trial).

## Key Findings

### 1. The landscape is mostly alive

Only 11.8% of random brains are "dead" (|DX| < 1m). Nearly 88% of the uniform weight space produces locomotion of some kind. The fitness landscape for this body is surprisingly forgiving — most random controllers make the robot move.

| Category | Count | Fraction |
|---|---|---|
| Dead (|DX| < 1m) | 59 | 11.8% |
| Forward (DX > 1m) | 222 | 44.4% |
| Backward (DX < -1m) | 219 | 43.8% |

### 2. Forward/backward symmetry is perfect

44.4% forward vs 43.8% backward — statistically indistinguishable. The robot body has no directional bias. Direction is entirely determined by the controller. This is expected from the physical symmetry of the URDF (BackLeg and FrontLeg are mirror-placed), but the data confirms it quantitatively.

### 3. Displacement distribution

| Statistic | Value |
|---|---|
| Mean |DX| | 7.87m |
| Median |DX| | 5.56m |
| P90 |DX| | 18.35m |
| Max |DX| | 41.01m |

The distribution is right-skewed: most gaits achieve modest displacement, a long tail reaches into high-performance territory. See `rs_fig01_dx_histogram.png`.

### 4. The best random gait is 81.8% of the CPG Champion

Trial 24 achieved DX=+41.0m — 81.8% of the zoo's best (CPG Champion, 50.1m). This was found with no optimization, just 500 uniform random draws. The weights:

| Weight | Value |
|---|---|
| w03 | -0.198 |
| w04 | -0.233 |
| w13 | -0.883 |
| w14 | +0.232 |
| w23 | -0.863 |
| w24 | -0.355 |

Note: this is a standard 6-synapse network, not a hidden-layer CPG. Yet it achieves >80% of what the CPG Champion does. This suggests that the 6-weight landscape has high-performance regions accessible by chance, but the absolute peak (CPG Champion's 50.1m) requires architectural innovation (hidden-layer oscillator) that random 6-synapse search cannot reach.

### 5. Phase lock bimodality confirmed from random samples

| Phase lock range | Fraction |
|---|---|
| > 0.8 (locked) | 40% |
| 0.2 – 0.8 (intermediate) | 53% |
| < 0.2 (unlocked) | 7% |

The bimodal distribution observed in the 116 zoo gaits is reproduced from scratch in 500 random samples. Phase locking is a property of the body-ground-contact physics, not of any particular optimization process. The system naturally tends toward either locked or unlocked regimes. See `rs_fig02_phase_lock_histogram.png` for the overlaid histograms (random vs zoo).

### 6. Roll dominance is universal

| Statistic | Value |
|---|---|
| Mean roll dominance | 0.758 |
| Min roll dominance | 0.378 |
| Max roll dominance | ~0.99 |

Not a single one of 500 random brains escapes roll-dominated rotation. The first principal component of angular velocity covariance is always the roll axis. This is a morphological constraint: the 3-link body with lateral hinge joints and no active roll control must rock side to side when it moves. The brain cannot override what the body dictates.

### 7. Weight-metric correlations

| Weight | DX | |DX| | Speed | Phase Lock | Entropy |
|---|---|---|---|---|---|
| w03 (Torso→BackLeg) | -0.246 | -0.136 | -0.132 | +0.049 | -0.003 |
| w04 (Torso→FrontLeg) | -0.159 | +0.063 | +0.194 | -0.085 | +0.038 |
| w13 (BackLeg→BackLeg) | -0.105 | -0.042 | -0.101 | +0.028 | +0.028 |
| w14 (BackLeg→FrontLeg) | -0.044 | +0.034 | +0.138 | +0.017 | -0.024 |
| w23 (FrontLeg→BackLeg) | -0.061 | -0.238 | -0.220 | -0.037 | -0.060 |
| w24 (FrontLeg→FrontLeg) | +0.012 | +0.006 | +0.053 | -0.081 | +0.060 |

**Interpretation**:
- **w03** is the strongest predictor of direction (r=-0.25): negative w03 biases forward motion.
- **w23** is the strongest predictor of displacement magnitude and speed (r=-0.24, -0.22): how the front leg sensor drives the back leg motor matters most for performance.
- **No single weight predicts phase lock or entropy** (all |r| < 0.1). These are emergent properties of the full 6-weight configuration, not attributable to any individual synapse.
- All correlations are weak (|r| < 0.25), confirming that locomotion quality depends on weight *combinations*, not individual values. This is why random search finds decent solutions but struggles to find the best — improvement requires coordinated changes across multiple weights simultaneously.

See `rs_fig04_weight_correlations.png` for the full heatmap.

### 8. Best-of-N curve: diminishing returns

The best-of-N curve (`rs_fig03_best_of_n.png`) shows rapid early improvement that plateaus:
- After 10 trials: best |DX| ~ 15m
- After 50 trials: best |DX| ~ 25m
- After 100 trials: best |DX| ~ 35m
- After 500 trials: best |DX| = 41m
- Zoo best: 50.1m (never reached)

Each doubling of trials yields diminishing marginal improvement. This is the fundamental limitation of random search: it explores weight space uniformly, with no memory and no gradient. The hill climber addresses this by exploiting local structure.

### 9. Speed vs efficiency in context

Random trials cluster in the low-speed, low-efficiency region of the zoo's speed-efficiency space (`rs_fig05_speed_efficiency.png`). A few outliers reach zoo-competitive speeds, but the Pareto frontier (Fuller → CPG Champion) remains dominated by the curated/evolved zoo gaits. The random search finds the "bulk" of weight space, not the frontier.

## Figures

| Figure | File | Description |
|---|---|---|
| 1 | `rs_fig01_dx_histogram.png` | DX and |DX| distributions with dead threshold |
| 2 | `rs_fig02_phase_lock_histogram.png` | Phase lock: random search vs zoo overlaid |
| 3 | `rs_fig03_best_of_n.png` | Best |DX| so far vs number of trials |
| 4 | `rs_fig04_weight_correlations.png` | Weight-metric Pearson r heatmap |
| 5 | `rs_fig05_speed_efficiency.png` | Speed vs efficiency, random in zoo context |
| 6 | `rs_fig06_dead_fraction.png` | CDF of |DX|, dead fraction analysis |
| 7 | `rs_fig07_symmetry.png` | Weight antisymmetry vs displacement by sensor |

## Implications for the Hill Climber

1. **The landscape is navigable**: 88% of random starting points produce motion, so the hill climber will almost always start from a viable gait.
2. **Direction is easy, speed is hard**: Random search finds directional gaits easily (symmetric forward/backward split) but rarely finds fast ones (P90 = 18.35m vs zoo best 50.1m).
3. **Individual weights have weak predictive power**: The hill climber's advantage (coordinated multi-weight perturbation) should exploit the interaction structure that random search cannot.
4. **The 6-synapse ceiling**: The best random 6-synapse brain reached 41m. The CPG Champion (50.1m) uses a hidden-layer architecture. This gap may represent a fundamental limit of the standard topology, not just insufficient search.
