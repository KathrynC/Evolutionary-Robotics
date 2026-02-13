# Super-Gaits Analysis: Key Findings

*Generated 2026-02-12 from causal surgery + interpolation analysis*

## Causal Surgery Findings

**Novelty Champion's architecture has a clear hierarchy:**
- **w13 (BackLeg->BackMotor)** and **w03 (Torso->BackMotor)** are load-bearing -- zeroing either collapses DX from 60m to ~0m. These form the rear propulsion loop.
- **w14 (BackLeg->FrontMotor)** is nearly redundant -- zeroing it changes DX by only +1%. The gait runs fine without this cross-leg connection.
- **w23 (FrontLeg->BackMotor)** is the surprise -- it's the tiniest weight (-0.037) but *halving* it boosts DX from 60.2m to 68.4m. Evolution over-tuned this synapse.

**CPG Champion has no redundancy** -- every synapse is load-bearing. But the hidden-layer oscillator degrades gracefully under partial ablation rather than catastrophically.

## The Two Super-Gaits

Two independent routes to beating the Novelty Champion (DX=60.19m):

### 1. Interpolation Super-Gait (t=0.52 on Nov<->T3)

| Metric | Novelty Champion | Interp Super |
|--------|-----------------|--------------|
| DX | 60.19m | 68.21m (+13%) |
| Work | 8,731 | 5,708 (-35%) |
| Efficiency | 0.0084 | 0.0124 (+48%) |
| Phase lock | 0.796 | 0.889 |
| Heading consistency | 0.946 | 0.954 |
| Fragility | 1.00x | 0.70x |

Found by interpolating between NC and Trial 3. A *gentler controller* with smaller, more balanced weights that achieves 13% more distance on 35% less energy. Higher phase lock means the legs are better coordinated.

### 2. w23-Half Variant

| Metric | Novelty Champion | w23-Half |
|--------|-----------------|----------|
| DX | 60.19m | 68.37m (+13.6%) |
| Work | 8,731 | 8,770 (same) |
| Efficiency | 0.0084 | 0.0082 |
| Phase lock | 0.796 | 0.808 |
| Heading consistency | 0.946 | 0.968 |
| Fragility | 1.00x | 0.40x |

Found in ablation data -- just halving one tiny synapse (w23: -0.037 -> -0.018). A "free upgrade" that evolution missed. Same energy budget, 13% more displacement, and 2.5x more robust.

## The Cliff at t=0.52 -> t=0.54

The most dramatic finding: a total weight perturbation of just **0.049 across all 6 synapses** causes a **69.3m swing in DX** (from +68.2 to -1.1). The collapse signature:

- Phase lock: 0.889 -> 0.459 (legs lose coordination)
- Heading consistency: 0.954 -> 0.464 (robot spirals)
- Back leg frequency: 2.58 Hz -> 0.06 Hz (essentially stops oscillating)
- Torso contact appears (robot face-plants)
- It still spends *more* energy (6,686 vs 5,708) despite going nowhere -- thrashing instead of walking

## The Fitness Landscape is Cliff-Riddled

Interpolating between NC and Trial 3 (51 points) reveals the landscape is **not smooth** -- DX swings violently between ~70m and near-zero across the range. A 2% interpolation step from the NC (t=0.00 -> t=0.02) causes a 47.5m drop. Gradient descent would fail catastrophically -- this explains why evolutionary search is necessary.

## The Big Interpretive Takeaway

Peak performance lives on a **phase boundary** -- the best gait in the entire landscape (t=0.52) is one step away from total collapse. The robot is operating right at a bifurcation where coordinated locomotion transitions into directionless thrashing.

The w23-Half variant is the more practical find: same performance, 2.5x more robustness (fragility ratio 0.40x vs NC's 1.00x).

## Sensitivity Rankings

| Synapse | NC | Interp Super | w23-Half |
|---------|-----|-------------|----------|
| Torso->Back | 1,840 | 351 | 707 |
| Torso->Front | 1,103 | 2,124 | 3,947 |
| BackLeg->Back | 619 | 84 | 1,013 |
| BackLeg->Front | 4,161 | 7,944 | 1,282 |
| FrontLeg->Back | 21,465 | 9,683 | 2,058 |
| FrontLeg->Front | 2,091 | 1,796 | 3,656 |
| **Total** | **31,279** | **21,982** | **12,663** |

NC's Achilles' heel is FrontLeg->Back (w23) at |gradient|=21,465. Both super-gaits tame this. The w23-Half variant has the most evenly distributed sensitivity -- no single synapse dominates.

## Files

- `analyze_super_gaits.py` -- analysis script (28+ sims, ~6s)
- `artifacts/super_gaits_analysis.json` -- full metrics, sensitivity data, weight analysis
- `artifacts/plots/super_fig01_trajectory.png` -- XY trajectory, X vs time, Z bounce
- `artifacts/plots/super_fig02_joints.png` -- joint positions and velocities (2x4 grid)
- `artifacts/plots/super_fig03_phase.png` -- phase portraits (j0 vs j1, colored by time)
- `artifacts/plots/super_fig04_energy.png` -- instantaneous power and cumulative work
- `artifacts/plots/super_fig05_contacts.png` -- contact raster patterns
- `artifacts/plots/super_fig06_sensitivity.png` -- local sensitivity bar chart
