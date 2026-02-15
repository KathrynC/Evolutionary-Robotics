# Dark Matter: What the Dead Gaits Are Actually Doing

**Date**: 2026-02-12
**Script**: `analyze_dark_matter.py`
**Data**: `artifacts/dark_matter.json` (59 gaits)
**Figures**: `artifacts/plots/dark_fig01_*` through `dark_fig06_*`

## Overview

59 out of 500 random-search trials (11.8%) produced gaits with |DX| < 1m — the "dead" zone. Every prior analysis discarded them. Here we re-simulate all 59 with full telemetry and find that they are not dead at all. They are circlers, cancellers, spinners, rockers, and coordinated oscillators that go nowhere. Some travel over 30 meters and end up back where they started. The dark matter of gaitspace is dynamically rich and categorically diverse.

## The Census

| Type | Count | Fraction | Description |
|---|---|---|---|
| **Other** | 23 | 39.0% | Coordinated oscillation, no net translation |
| **Circler** | 17 | 28.8% | Curved paths that loop back near the origin |
| **Canceller** | 13 | 22.0% | Walk out, walk back — high path length, zero net |
| **Spinner** | 4 | 6.8% | Slow rotation in place, joints settle to fixed positions |
| **Rocker** | 2 | 3.4% | High phase lock, pure rocking oscillation |
| Frozen | 0 | 0.0% | — |

### Finding 0: Nothing Is Frozen

Not a single one of 59 dead gaits has motionless joints. Every set of weights in [-1, 1]^6 produces joint oscillation. The category "truly inert" does not exist in this weight space. The robot always moves its joints; the question is only whether that motion produces net displacement.

This means the 6D landscape has no fixed-point region — the entire space is oscillatory. Fitness is not "can this controller produce motion?" (always yes) but "does this motion translate into displacement?" (only 88% of the time).

## Behavioral Profiles

| Type | Speed | Path Length | Work | Phase Lock | Total Yaw | Path Straightness | Heading Consistency |
|---|---|---|---|---|---|---|---|
| Other | 0.71 m/s | 11.9m | 5,233 | **0.794** | 6.4 rad | 0.357 | 0.242 |
| Circler | 0.64 m/s | 10.6m | **8,839** | 0.588 | 2.9 rad | 0.079 | 0.104 |
| Canceller | **0.87 m/s** | **14.5m** | 7,730 | 0.552 | 6.1 rad | 0.125 | 0.149 |
| Spinner | 0.59 m/s | 9.7m | 5,512 | 0.597 | 4.2 rad | **0.023** | 0.101 |
| Rocker | 0.25 m/s | 4.1m | 2,628 | **0.917** | 3.2 rad | **0.352** | **0.327** |

**New metrics** (added to `compute_beer_analytics.py`, available to all downstream scripts):
- **Path straightness** = net_displacement / path_length. Range [0, 1]. A score of 1.0 means the robot walked a perfect straight line; 0.0 means it ended where it started regardless of how far it traveled.
- **Heading consistency** = |mean(e^{iθ})| where θ = atan2(vy, vx), computed only for timesteps where speed > 0.1 m/s. Range [0, 1]. A score of 1.0 means the robot always moved in the same direction; 0.0 means its heading was uniformly distributed around the compass. Analogous to phase_lock but for heading instead of joint coordination.

## Finding 1: Cancellers Are Walking Gaits That Change Their Minds

The 13 cancellers are the most surprising category. Mean speed: 0.87 m/s. Mean path length: 14.5 meters. These robots are genuinely *walking* — they cover ground at speed comparable to movers in the random search (the overall mean speed was 1.34 m/s for all 500 trials, including the fast ones). But they end up back near the origin.

The XY trajectories (fig 2) show tangled, multi-directional paths: the robot walks forward, turns, walks sideways, turns again, comes back. These are not simple back-and-forth oscillations. They're complex multi-phase trajectories where the heading changes multiple times during the 16.7-second simulation.

The cancellers have moderate phase lock (0.552) and the highest mean speed of any dark type. Their heading consistency is 0.149 — they move in many directions during a trial, never committing to one for long. Their path straightness is 0.125, confirming that they cover 8× more path than net distance. They represent a category that any DX-based fitness function misses entirely: competent walkers with unstable heading. A fitness function that rewarded *path length* instead of *net displacement* would rank these among the good gaits.

## Finding 2: The "Other" Category Is the Most Phase-Locked

The 23 gaits classified as "Other" (39% of dark matter) have the second-highest phase lock (0.794) — higher than many *successful* movers. Their phase portrait (fig 6) shows a tight diagonal orbit: both joints oscillate in near-perfect coordination along a correlated axis.

These gaits oscillate coherently but don't translate that oscillation into displacement. They're the dynamical equivalent of an engine revving in neutral — the motor is running smoothly, the joints are coordinated, but the power isn't reaching the wheels.

Why? Their mean speed is 0.71 m/s (not zero), and their path length is 11.9m. They ARE moving — but they're moving in a way that cancels out. The yaw (6.4 rad total) is high, meaning they rotate as they move. The combination of forward motion + rotation produces a curving trajectory that, over 16.7 seconds, happens to curve back near the start.

The heading consistency metric now quantifies this: Others score 0.242 — they have a weak directional bias but rotate enough to cancel it. Circlers score even lower at 0.104 — their heading is nearly uniformly distributed. Yet Others have significantly higher path straightness (0.357 vs 0.079), meaning that moment-to-moment the Others walk straighter segments between turns, while Circlers curve continuously.

This suggests that "Other" and "Circler" are really the same phenomenon at different curvature scales. The Others walk in straighter segments with discrete turns; the Circlers curve continuously. Both end up near the origin, but via different trajectory geometries.

## Finding 3: Rockers Are the Purest Oscillators

Only 2 rockers, but they have the highest phase lock (0.917), the lowest speed (0.25 m/s), and — surprisingly — the highest heading consistency (0.327) and path straightness (0.352) of any dark type. The joint gallery (fig 4) shows clean, regular oscillation with both joints moving in sync. The height trace shows the characteristic bouncing pattern — z oscillates rhythmically.

The high heading consistency is counterintuitive: how can a gait that barely moves have the most consistent heading? Because when the Rockers *do* translate (at 0.25 m/s, they cover 4.1m of path), that translation is along a stable axis. They don't rotate much (total yaw 3.2 rad, lowest of any type). Their rocking oscillation has a slight directional asymmetry that produces a weak but consistent drift. They're the most "honest" dead gaits — they go nowhere because they barely move, not because they move vigorously in cancelling directions.

These are the rocking chair of gaitspace: perfectly coordinated, energy-minimal oscillation that converts to pure body rocking with negligible forward progress. They use the least energy (work = 2,628, roughly half the other types) because the oscillation is gentle and efficient — but efficient at staying in place.

The Rockers might be the starting point for the most energy-efficient gaits. If a small perturbation could add a directional bias to their pure rocking oscillation, the result would be a gait that combines the Rocker's coordination with forward translation — similar to Trial 3's "lazy walker" strategy but discovered from the opposite direction (starting from pure oscillation rather than starting from locomotion).

## Finding 4: Spinners Settle Into Fixed Points — Then Rotate

The 4 spinners show an interesting temporal pattern in the joint gallery (fig 4): the joints oscillate during the first 2-3 seconds, then settle to nearly fixed positions. But the robot continues to slowly rotate, accumulating yaw. The height trace shows the robot maintaining normal standing height.

These are the closest thing to "motionless" in the landscape, but they're not truly fixed-point controllers. The initial oscillation excites a slow rotational mode that persists even after the joints stop moving. The rotation comes from residual angular momentum, not from active joint torques.

## Finding 5: Circlers Spend the Most Energy Going Nowhere

The 17 circlers have the highest work proxy (8,839) — spending 70% more energy than the Others while covering similar path length. Their phase portraits show wide orbits — the joints swing through a larger range of coordinated motion. But this vigorous activity produces curved paths that loop back on themselves.

Circlers are the dark matter analogue of the Novelty Champion's diagonal trajectory: they move fast and energetically along a curved heading. The difference is that the Novelty Champion's curvature is gentle enough that 16.7 seconds of walking covers 73 meters net. The Circlers' curvature is tight enough that 16.7 seconds brings them back to the origin.

This implies there's a **curvature threshold**: gaits with heading rate below some critical value produce large net displacement; gaits above it produce circling. The threshold probably depends on speed — faster gaits can tolerate more curvature because they cover more ground per rotation.

## The Dark Matter Spectrum

Arranging the types by dynamical complexity:

```
Frozen (0%)  →  Spinner (7%)  →  Rocker (3%)  →  Other (39%)  →  Circler (29%)  →  Canceller (22%)
  [none]      [settling]      [oscillating]    [coordinated]    [locomoting]      [walking]
                                                                   in circles     multi-phase
```

The spectrum runs from simple to complex. No gaits are frozen. A few settle to fixed points (spinners). A few oscillate in place (rockers). Most oscillate coherently (others). Many translate in curves (circlers). And some walk complex multi-directional paths (cancellers).

**The dead zone is not a failure mode — it's a dynamical zoo.** These 59 gaits span the full range from near-quiescent to fully active locomotion. The only thing they share is that their net displacement is low. Measured by any other criterion — energy, coordination, contact complexity, trajectory diversity — they are as varied as the "successful" movers.

## Implications

### The DX fitness function is blind to most of the dynamics

59 gaits discarded as "dead" include robots walking 14.5 meters (cancellers), oscillating with 0.92 phase lock (rockers), and burning 8,800 units of work (circlers). A fitness landscape measured only by DX treats all of these identically: zero. The landscape we mapped in the 500-trial random search and the walker competition is a projection of the full dynamical reality onto a single axis.

### Curvature is the hidden variable — and now we measure it

The distinction between a "mover" and a "circler" is heading stability, not speed or coordination. Two gaits with identical speed, phase lock, and energy expenditure can produce 50m displacement or 0m displacement depending on whether the heading drifts.

We now measure this directly with two metrics added to `compute_beer_analytics.py`:

| Metric | What it captures | Dark matter range | Mover range (for comparison) |
|---|---|---|---|
| **path_straightness** | net_displacement / path_length | 0.02 – 0.36 | 0.85 – 0.98 |
| **heading_consistency** | |mean(e^{iθ})| | 0.10 – 0.33 | 0.80 – 0.99 |

The gap is enormous. Successful movers score 0.85+ on both metrics; the entire dark matter population scores below 0.36 on both. Heading consistency cleanly separates the dead zone from the living — more cleanly than speed, phase lock, or energy, all of which overlap between movers and dark matter types.

Within the dark matter, heading consistency separates the types into three tiers:
1. **Spinners + Circlers** (0.10): heading uniformly distributed — pure rotation or continuous curvature
2. **Cancellers** (0.15): weak directional preference, overridden by frequent heading changes
3. **Rockers + Others** (0.24–0.33): real directional bias, just not enough speed to escape the dead zone

### The Rockers are seeds for efficient gaits

The 2 rockers (phase lock 0.917, work 2,628) are oscillating efficiently in place. If a directed perturbation could break the symmetry that keeps them stationary — adding a slight directional bias — the result would be an extremely efficient walker. This is the inverse of the hill climber's approach: instead of starting from a walker and optimizing fitness, start from a rocker and optimizing asymmetry.

### The Cancellers suggest a multi-objective landscape

13 gaits that walk 14.5 meters and come back. Under DX fitness, they're worthless. Under path-length fitness, they're in the 60th percentile. Under a "heading stability" fitness that penalizes curvature, they'd score poorly. The dead zone is where different fitness functions disagree most strongly — it's the most "contested" region of the landscape, where what you measure determines what you find.

## Figures

| Figure | File | Description |
|---|---|---|
| 1 | `dark_fig01_overview.png` | 6-panel scatter matrix of key descriptors (incl. path_straightness vs heading_consistency), colored by type |
| 2 | `dark_fig02_xy_trajectories.png` | XY trajectory gallery, 5 panels by type (up to 8 per type) |
| 3 | `dark_fig03_clusters.png` | PCA of behavioral space: heuristic types vs k-means clusters |
| 4 | `dark_fig04_joint_gallery.png` | Joint angles + height/speed for one representative per type |
| 5 | `dark_fig05_contact_patterns.png` | Contact patterns over time, one representative per type |
| 6 | `dark_fig06_phase_portraits.png` | Phase portraits (j0 vs j1) colored by time, one per type |
