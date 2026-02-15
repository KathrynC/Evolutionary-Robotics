# Gaitspace Walker Designs: Robots That Navigate Weight Space

**Date**: 2026-02-12

## Motivation

The random search and cliff analysis established the topology of the 6-dimensional fitness landscape: 88% alive, cliff-riddled (42% of points have a >10m cliff within r=0.05), fractal roughness (gradient 3x steeper at small scales), bimodal phase boundaries, and a thin Pareto frontier (0.8% of random points are non-dominated). The question is: what kinds of agents are suited to navigate this terrain?

Each "walker" is an algorithm that moves through weight space, evaluating gaits by running simulations (~0.1s per evaluation), and deciding where to step next based on what it senses about the local landscape.

## The Terrain (Summary of Established Properties)

| Property | Value | Source |
|---|---|---|
| Dimensions | 6 (synaptic weights, each in [-1, 1]) | Standard 6-synapse topology |
| Evaluation cost | ~0.1s per point (4000-step headless simulation + analytics) | random_search_500.py |
| Fraction alive (|DX| > 1m) | 88.2% | 500 random trials |
| Forward/backward symmetry | 44.4% / 43.8% | 500 random trials |
| Cliff prevalence (|delta_DX| > 10m within r=0.05) | 42% of base points | Cliff analysis, 50 base × 30 perturbations |
| Gradient scaling | 3x steeper at r=0.05 than r=0.2 | Cliff analysis |
| Phase lock bimodality | 40% locked (>0.8), 7% unlocked (<0.2) | 500 random trials |
| Pareto frontier thickness | 4/500 = 0.8% non-dominated | 500 random trials |

## Five Walker Designs

### 1. The Hill Climber

**Question it answers**: What is the best gait reachable by local search from a random starting point?

**Algorithm**: Stand on a point in weight space, evaluate fitness (e.g., DX). Propose a random perturbation at fixed radius. Evaluate the new point. If fitness improves, move there; otherwise stay. Repeat.

**Predicted behavior given terrain topology**: The cliff analysis predicts the hill climber will be fragile. At any given point, 42% of random directions within r=0.05 lead to a >10m cliff. The hill climber's rejection criterion (only accept improvements) will prevent it from falling off cliffs — but this also means it gets trapped on whatever plateau it starts on. Step size is critical: too large and it jumps across regimes randomly (becoming random search); too small and it stalls on micro-plateaus.

**Terrain-specific design choices**:
- Step size should be tuned to the cliff prevalence. Given that r=0.05 produces mean |delta_DX| of 5.16m, a step size of 0.01–0.02 might be appropriate for exploitation.
- Multiple restarts from different random starting points would help, since the landscape has many distinct basins.

**Status**: Next Ludobots module (L. The Hill Climber).

### 2. The Ridge Walker

**Question it answers**: What is the shape of the Pareto frontier between speed and efficiency?

**Algorithm**: Maintain the current point's position on the Pareto frontier (or as close as possible). At each step, propose a perturbation and accept it only if the new point is Pareto-non-dominated by the old one (i.e., it doesn't get worse on both objectives simultaneously). Preferentially step toward regions that extend the known frontier.

**Predicted behavior**: Since only 0.8% of random points are Pareto-optimal, finding the frontier is hard. The ridge walker would need to start from a known Pareto point (e.g., Trial 3 or Trial 24 from the random search) and trace along the frontier surface. The cliff-riddled landscape means the frontier itself may be discontinuous — the walker might find that the frontier has gaps where no continuous path connects the efficiency end (Trial 3) to the displacement end (Trial 24).

**Terrain-specific design choices**:
- Two evaluation objectives: DX (or |DX|) and efficiency (distance/work).
- Acceptance criterion: Pareto non-domination rather than scalar fitness improvement.
- Could extend to 3+ objectives (add phase lock, entropy) for higher-dimensional frontier mapping.

### 3. The Cliff Mapper

**Question it answers**: Where are the behavioral boundaries in weight space? What is the topology of the critical surfaces that separate walking from spinning, locked from unlocked, forward from backward?

**Algorithm**: At each point, probe N random directions at small radius. Estimate the local gradient magnitude in each direction. Walk toward the steepest gradient (toward the cliff, not away from it). When a cliff edge is found (a direction where |delta_DX| exceeds a threshold), switch to edge-following mode: step perpendicular to the cliff direction, staying on the boundary while tracing its shape.

**Predicted behavior**: The cliff analysis showed that 80% of random points have a >5m cliff within r=0.05, and the gradient is fractal (steeper at smaller scales). The cliff mapper would find boundaries almost immediately from any starting point. The interesting output is not individual gaits but the geometry of the boundary network — a map of the critical surfaces in 6D weight space, projected into behavioral coordinates.

**Terrain-specific design choices**:
- "Cliff" threshold: |delta_DX| > 10m per r=0.05 step (based on empirical P85 from cliff analysis).
- Edge-following: binary search along the cliff direction to find the precise boundary, then step perpendicular.
- Output: a catalog of boundary points, each annotated with the behavioral regimes on either side.

**Research value**: This walker would directly test whether the phase lock bimodality corresponds to a connected critical surface in weight space, or whether there are multiple disconnected phase boundaries.

### 4. The Novelty Seeker

**Question it answers**: What is the full diversity of behaviors this body can produce? Are there behavioral regions that no random search or fitness-driven optimization would ever find?

**Algorithm** (after Lehman & Stanley 2011): Maintain an archive of behavioral descriptors (DX, speed, phase lock, entropy, roll dominance, yaw) for every evaluated point. At each step, propose K candidate perturbations. Evaluate each. Compute the behavioral novelty of each candidate (mean distance to k-nearest neighbors in the archive). Accept the most novel candidate. Add it to the archive.

**Predicted behavior**: The novelty seeker would initially explore the "easy" behavioral variation (forward/backward, fast/slow) and then progressively seek out rare behaviors — extreme spinners, high-entropy gaits, low-roll-dominance gaits, perfectly phase-locked non-movers (like Trial 272). It would fill in the sparse regions of the gaitspace scatterplots.

**Terrain-specific design choices**:
- Behavioral descriptor: 6D vector (DX, speed, efficiency, phase lock, entropy, roll dominance).
- Novelty metric: Euclidean distance in normalized behavioral space (each dimension scaled to [0,1] by zoo range).
- Archive size: unlimited (every evaluated point is kept); k=15 for nearest-neighbor novelty.

**Research value**: Would reveal whether the zoo's 116 gaits sample the full behavioral diversity or whether large regions of gaitspace remain undiscovered. Particularly interesting for finding gaits with low roll dominance (the rarest property in both the zoo and random search).

### 5. The Ensemble Explorer

**Question it answers**: How many distinct behavioral basins does weight space contain? What is the basin structure of the landscape?

**Algorithm**: Launch N walkers (e.g., 20) from random starting points. Each walker does local hill climbing. Periodically (every M steps), compute pairwise behavioral distances between all walkers. If two walkers are in similar behavioral regions (distance < threshold in behavioral space), teleport the worse-performing one to a random unexplored location. Continue until all walkers have converged to distinct basins.

**Predicted behavior**: Given the cliff analysis, we expect many distinct basins separated by cliffs. The ensemble would gradually sort itself into one walker per basin, revealing the basin count and the behavioral signature of each. The cliff prevalence (42% of points have nearby >10m cliffs) suggests the basin count is high — possibly dozens within the 6D hypercube.

**Terrain-specific design choices**:
- Basin identity: two walkers are in the "same basin" if their behavioral descriptors are within 20% of each other on all dimensions and there exists a continuous path between them that never crosses a >10m cliff.
- Teleportation trigger: behavioral distance < 2.0 (in normalized space) between any pair.
- N=20 walkers, M=50 steps between communication rounds.

**Research value**: Would produce a definitive count of distinct behavioral regimes accessible to the 6-synapse topology, and identify the "champion" of each basin.

## The Sensing Question

In physical locomotion, the robot senses ground contact, joint angles, and balance. In gaitspace navigation, the walker's "sensors" are the Beer-framework analytics pillars. A walker can sense:

- **Local fitness**: DX, speed, efficiency at the current point
- **Local gradient**: estimated from small perturbations (the cliff analysis is exactly this sensing)
- **Local cliffiness**: variance of fitness among neighbors (high variance = near a cliff)
- **Behavioral novelty**: distance to nearest archived behaviors
- **Phase lock regime**: whether the current point is in the locked or unlocked phase

A walker that senses its own local landscape topology could make intelligent decisions about step size (small near cliffs, large on plateaus), direction (toward novelty, along ridges, away from cliffs), and strategy (exploit when on a promising slope, explore when on a flat plateau).

## Computational Budget

| Design | Steps | Walkers | Evaluations | Time estimate |
|---|---|---|---|---|
| Hill climber | 1,000 | 1 | 1,000 | ~100s |
| Ridge walker | 500 | 1 | ~1,500 (3 candidates/step) | ~150s |
| Cliff mapper | 200 | 1 | ~2,000 (10 probes/step) | ~200s |
| Novelty seeker | 500 | 1 | ~5,000 (10 candidates/step) | ~500s |
| Ensemble explorer | 500 | 20 | ~10,000 | ~1,000s (~17 min) |

All are feasible on a single machine in under 20 minutes. The evaluation primitive (`run_trial_inmemory` from `random_search_500.py`) provides the interface: weights in, analytics out, 0.1s per call.

## The Meta-Design

The most ambitious extension: a walker whose step-size and direction strategy is itself a neural network, trained by meta-evolution to navigate cliff-riddled fitness landscapes efficiently. The gaitspace terrain becomes the training environment for a meta-learner — a robot that evolves to be good at evolving. The walker would learn general strategies for cliff-riddled landscapes: when to take small careful steps, when to make large exploratory jumps, how to detect and trace cliff edges, when to switch from exploitation to exploration.

This is a research program rather than a single experiment, but the infrastructure built here (fast evaluation, Beer analytics, cliff detection) provides all the necessary components.
