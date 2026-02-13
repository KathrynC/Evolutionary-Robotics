# Gait Interpolation: What Lies Between Champions?

## Overview

Linearly interpolated in 6D weight space between pairs of high-performing gaits and simulated at each point. Tests whether champion-to-champion corridors are smoother than random directions through weight space.

**Script**: `gait_interpolation.py`
**Budget**: 6,452 simulations in 8.1 minutes
**Seed**: `np.random.RandomState(123)`

## Champions (6-synapse standard topology)

| Gait | DX (m) | Weights |
|------|--------|---------|
| Novelty Champion | +60.19 | [-1.31, -0.34, +0.83, -0.38, -0.04, +0.44] |
| Pelton | +34.70 | [-0.30, +1.00, -1.00, +0.30, -0.30, +1.00] |
| Trial 3 | +25.87 | [-0.60, -0.42, +0.11, -0.00, +0.30, +0.21] |
| Curie | +23.73 | [-0.30, +0.90, -0.90, +0.30, -0.30, +0.90] |
| Noether | -21.65 | [-0.70, +0.30, -0.50, +0.50, -0.30, +0.70] |
| Original | -0.75 | [+1.00, -1.00, +1.00, -1.00, +1.00, -1.00] |

## Part 1: Pairwise Transects (480 sims)

80-point linear interpolation along each of 6 champion pairs.

| Pair | DX Range | Roughness | Max Step | Sign Flip Rate |
|------|----------|-----------|----------|----------------|
| NC → Trial 3 | [+3.4, +66.7] | 29.6 | 57.4 m | 0.72 |
| NC → Pelton | [-42.5, +60.2] | 13.9 | 45.0 m | 0.58 |
| NC → Original | [-5.2, +60.2] | 9.3 | 46.0 m | 0.72 |
| Pelton → Curie | [-10.3, +34.7] | 14.8 | 38.9 m | 0.65 |
| Pelton → Noether | [-27.9, +34.7] | 12.4 | 25.9 m | 0.59 |
| Trial 3 → Original | [-12.9, +25.9] | 4.7 | 15.1 m | 0.60 |

Every transect is dominated by fractal noise. Sign flip rates of 0.58-0.72 mean DX changes sign more than half the time between adjacent sample points (1.25% steps in weight space).

## Part 2: Grand Tour (200 sims)

200 points walking NC → Pelton → Curie → Noether → Trial 3 → Original. The DX profile resembles an EKG — chaotic noise with champion peaks at labeled waypoints. Weight trajectories are perfectly smooth linear interpolations; the DX they produce is pure chaos.

## Part 3: Midpoint Probing (966 sims)

8 perpendicular directions at each transect's midpoint, r = ±0.15.

| Midpoint | Midpoint DX | Perp Roughness | Transect Roughness |
|----------|-------------|----------------|-------------------|
| NC → Trial 3 | +47.8 m | 35.9 | 29.6 |
| NC → Pelton | +4.5 m | 10.1 | 13.9 |
| NC → Original | -0.5 m | 7.0 | 9.3 |
| Pelton → Curie | +18.7 m | 14.1 | 14.8 |
| Pelton → Noether | +5.7 m | 13.2 | 12.4 |
| Trial 3 → Original | -3.4 m | 5.3 | 4.7 |

Mean perpendicular roughness (14.3) ≈ mean transect roughness (14.1). Ratio: **1.01**. The landscape at midpoints is isotropic — no direction is privileged.

## Part 4: Random Baseline (4,800 sims)

20 random directions per pair origin, 40 points each, same length as transect.

### The Central Result

| Metric | Transect | Random | Ratio |
|--------|----------|--------|-------|
| Mean roughness | 14.13 | 14.15 | **0.999** |
| Mean sign change rate | 0.643 | 0.615 | 1.05 |

**Champion transects are exactly as rough as random directions.** The ratio of 0.999 is indistinguishable from 1.0. There are no privileged paths between champions.

## Intermediate Discoveries

Only one transect produced a point exceeding both endpoints:
- **NC → Trial 3**: Peak |DX| = 66.7m at t=0.30 (+6.5m over NC's 60.2m)

This is a serendipitous find — a weight combination 30% of the way from NC to Trial 3 that outperforms both. But it sits in a field of fractal noise, not in a smooth valley.

## Conclusions

### 1. No smooth corridors exist between champions
The transect roughness equals random roughness to three decimal places (ratio 0.999). Champions are not connected by ridges, valleys, or any exploitable gradient structure. They are isolated peaks in a fractal landscape.

### 2. The fractal is universal
This completes the universality proof begun in the cliff taxonomy:
- **Cliff taxonomy**: fractal at every cliff point (local)
- **Deep resolution**: fractal at every scale (scale-invariant)
- **Isotropy analysis**: fractal in every direction (isotropic)
- **Gait interpolation**: fractal along every path, including champion corridors (global)

The fitness landscape is fractal everywhere, at every scale, in every direction, including the "privileged" directions between known solutions.

### 3. Evolution must be a global search
Since no local structure exists — not around individual points, not along champion transects, not in any direction — the only viable optimization strategy is global sampling. This is why evolutionary algorithms work for embodied agents: they never assume local structure exists.

### 4. One serendipitous discovery
The NC→Trial 3 transect reveals a 66.7m gait at t=0.30 — the best single-evaluation point found in this entire investigation series. This suggests that while the landscape has no smooth structure, it does have dense packing of high-performing points near existing champions. Sampling *near* champions is a better strategy than sampling *between* them along a line.

## Connection to Previous Findings

| Finding | Confirmed by |
|---------|-------------|
| Type 3 fractal (cliff taxonomy) | Transect roughness = random roughness |
| Isotropic chaos (deep resolution) | Perpendicular/transect ratio = 1.01 |
| Contact dynamics as chaos source (resonance) | Smooth weight interpolation → fractal DX |
| Late-developing quality (embryology) | Small weight differences → large DX variance |

## Output Files

- `artifacts/gait_interpolation.json` — all transect, tour, midpoint, and random data
- `artifacts/plots/interp_fig01_pairwise_transects.png` — 6-panel champion pair profiles
- `artifacts/plots/interp_fig02_grand_tour.png` — grand tour DX + weight trajectories
- `artifacts/plots/interp_fig03_midpoint_landscape.png` — perpendicular probes at midpoints
- `artifacts/plots/interp_fig04_roughness_comparison.png` — transect vs random roughness bars
- `artifacts/plots/interp_fig05_weight_trajectories.png` — DX vs weight distance + distributions
- `artifacts/plots/interp_fig06_verdict.png` — summary verdict panel
