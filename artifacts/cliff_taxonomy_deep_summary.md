# Deep Resolution: Step Zone Type 3 Chaos Analysis

## Overview

Follow-up to `cliff_taxonomy.py`. Probed the 10 most chaotic Step-type cliff profiles at ultra-fine scales to determine whether a smoothness floor exists and characterize the Wolfram complexity class of the fitness landscape.

**Script**: `cliff_taxonomy_deep.py`
**Budget**: 2,510 sims in 3.1 minutes
**Seed**: `np.random.seed(123)`

## Targets

10 Step profiles selected by chaos score (sign_change_rate * spectral_ratio / (1 + autocorr)):

| # | Orig Rank | Idx | Chaos Score | Cliffiness | Step Location |
|---|-----------|-----|-------------|------------|---------------|
| 1 | 7 | 60 | 4.192 | 35.6 | +0.1241 |
| 2 | 39 | 114 | 2.945 | 20.6 | +0.0552 |
| 3 | 46 | 354 | 2.853 | 19.1 | +0.0552 |
| 4 | 11 | 202 | 2.578 | 31.5 | -0.1379 |
| 5 | 44 | 193 | 2.087 | 19.3 | +0.1793 |
| 6 | 23 | 122 | 1.872 | 25.2 | +0.0552 |
| 7 | 15 | 127 | 1.418 | 28.6 | +0.0966 |
| 8 | 18 | 78 | 1.360 | 28.1 | -0.0414 |
| 9 | 47 | 253 | 1.158 | 19.0 | -0.1241 |
| 10 | 10 | 434 | 0.964 | 34.2 | -0.0414 |

## Phase 1: Logarithmic Zoom Cascade (1,200 sims)

20-point DX profiles at 6 geometrically-spaced scales centered on each target's primary step location.

**Scales**: r = +/-{0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003}

### Key Finding: No Smoothness Floor

Mean log-log slope of DX range vs scale: **0.011 +/- 0.093**

- Slope = 0 means DX range is scale-invariant (fractal)
- Slope = 1 means DX range decays proportionally (smooth/differentiable)
- 6/10 targets have negative slopes (landscape gets rougher at smaller scales)

| Target | Slope | DX@r=0.01 | DX@r=0.001 | DX@r=0.0001 | DX@r=3e-05 |
|--------|-------|-----------|------------|-------------|------------|
| #1 | -0.048 | 46.5 | 42.2 | 61.9 | 53.5 |
| #2 | +0.023 | 37.6 | 39.9 | 41.7 | 29.6 |
| #3 | +0.008 | 34.7 | 21.4 | 24.9 | 27.6 |
| #4 | +0.256 | 63.7 | 50.7 | 10.0 | 24.4 |
| #5 | +0.054 | 20.1 | 13.9 | 14.2 | 11.6 |
| #6 | -0.047 | 21.7 | 29.2 | 29.9 | 29.1 |
| #7 | +0.044 | 65.5 | 56.3 | 48.2 | 51.7 |
| #8 | -0.054 | 36.7 | 34.6 | 47.7 | 55.2 |
| #9 | -0.088 | 19.0 | 17.0 | 29.4 | 27.7 |
| #10 | -0.037 | 23.9 | 18.5 | 25.7 | 31.4 |

### Derivative Scaling (diverges as scale shrinks)

| Scale | Mean |dDX/dr| |
|-------|----------------|
| r=0.01 | 9,140 |
| r=0.003 | 29,838 |
| r=0.001 | 94,027 |
| r=0.0003 | 300,214 |
| r=0.0001 | 892,470 |
| r=0.00003 | 3,020,839 |

The gradient diverges as ~1/r, confirming non-differentiability.

## Phase 2: Directional Fan (810 sims)

8 directions evenly spaced in the perpendicular plane + gradient direction, 9-point profiles at r=+/-0.005.

### Key Finding: Isotropic Chaos

- Mean isotropy (std/mean of perpendicular DX ranges): **0.262**
- Mean gradient/perpendicular ratio: **0.978**

The gradient direction is no rougher than perpendicular directions. Chaos is equally strong in every direction.

| Target | Perp Mean (m) | Perp Std (m) | Grad (m) | Grad/Perp |
|--------|---------------|-------------|----------|-----------|
| #1 | 39.15 | 10.91 | 28.00 | 0.72 |
| #2 | 28.15 | 10.60 | 18.34 | 0.65 |
| #3 | 19.00 | 6.07 | 27.37 | 1.44 |
| #4 | 38.13 | 9.11 | 38.59 | 1.01 |
| #5 | 10.41 | 2.50 | 13.35 | 1.28 |
| #6 | 23.97 | 6.27 | 26.68 | 1.11 |
| #7 | 39.26 | 4.32 | 32.26 | 0.82 |
| #8 | 29.96 | 7.70 | 38.78 | 1.29 |
| #9 | 30.07 | 11.33 | 19.56 | 0.65 |
| #10 | 27.48 | 4.29 | 22.02 | 0.80 |

## Phase 3: 2D Micro-Grids (500 sims)

10x10 grids in gradient + perpendicular plane at r=+/-0.005 for top 5 targets.

### Key Finding: No Spatial Structure

The DX heatmaps show random patchwork with no ridges, valleys, or gradients. Local cliffiness (|grad DX|) reaches 50,000 m/unit — a 0.001 change in weight shifts DX by 50 meters.

| Target | DX Range | Mean Cliffiness | Max Cliffiness |
|--------|----------|----------------|----------------|
| #1 | [-43.7, +18.3] | 12,922 | 38,944 |
| #2 | [-2.6, +39.4] | 9,614 | 36,378 |
| #3 | [-29.4, +17.6] | 7,153 | 30,815 |
| #4 | [-51.8, +14.6] | 11,712 | 51,792 |
| #5 | [-23.9, -3.4] | 3,734 | 11,907 |

## Verdicts

| Question | Result |
|----------|--------|
| **Smoothness floor?** | NO — slope 0.011, structure persists to r = 3e-05 |
| **Isotropy?** | YES — comparable chaos in all directions |
| **Wolfram class** | **Type 3 (Chaotic)** |
| **Gradient descent viable?** | NO — at any scale. Derivatives diverge. |
| **Optimization implication** | Global search required (evolutionary, random) |

## Wolfram Classification Context

Steps were identified as the richest Type 3 zone across three independent metrics:
- Flattest fractal slope (0.064 vs 0.203 for Canyons) in the taxonomy analysis
- Highest spectral ratio (2.584) — flattest power spectrum, closest to white noise
- Most representation in top chaos quartile

The deep resolution confirms: Steps sit firmly in **Wolfram Class III** — aperiodic, scale-invariant, isotropic chaos. The fitness landscape in these zones is genuinely non-differentiable, resembling a continuous-but-nowhere-differentiable function (Weierstrass-type).

This is *not* Type 4 (edge of chaos) — there is no directional structure or exploitable pattern. And it's not numerical noise — simulations are fully deterministic (verified to float64 precision).

## Output Files

- `artifacts/cliff_taxonomy_deep.json` — all raw data, profiles, verdicts
- `artifacts/plots/deep_fig01_zoom_cascade.png` — self-similarity across 6 zoom levels
- `artifacts/plots/deep_fig02_fractal_dimension.png` — log-log DX range vs scale (the key result)
- `artifacts/plots/deep_fig03_directional_fan.png` — 8 perpendicular directions + gradient
- `artifacts/plots/deep_fig04_isotropy.png` — angular structure analysis
- `artifacts/plots/deep_fig05_micro_grids.png` — 2D DX and cliffiness texture
- `artifacts/plots/deep_fig06_smoothness_verdict.png` — summary verdict panel
