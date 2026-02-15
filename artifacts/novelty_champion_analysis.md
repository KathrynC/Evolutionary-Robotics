# The Novelty Champion: A Power Runner Found by Not Looking

**Date**: 2026-02-12
**Script**: `analyze_novelty_champion.py`
**Figures**: `artifacts/plots/champ_fig01_*` through `champ_fig07_*`

## Overview

The Walker Competition's Novelty Seeker found a gait with DX = +60.2m — 20% beyond the CPG Champion (50.1m), the previous record-holder. This gait uses a standard 6-synapse network, no hidden layer, no CPG architecture. It was found by an algorithm that optimizes for behavioral novelty, not displacement. The algorithm wasn't trying to go far. It went farthest.

This analysis compares the Novelty Champion against the CPG Champion and Trial 3 ("The Accidental Masterpiece") using full-resolution telemetry and Beer-framework analytics.

## Head-to-Head Comparison

| Metric | Novelty Champion | CPG Champion | Trial 3 | NC/CPG |
|---|---|---|---|---|
| DX | +60.19m | +50.11m | +25.88m | 1.20x |
| DY | -41.83m | -0.14m | +4.22m | — |
| **Net distance** | **73.30m** | **50.11m** | **26.22m** | **1.46x** |
| Mean speed | 4.62 m/s | 3.33 m/s | 1.76 m/s | 1.39x |
| Speed CV | 0.27 | 0.42 | 0.39 | 0.63x |
| Work proxy | 8,731 | 6,482 | 1,956 | 1.35x |
| Efficiency (dist/work) | 0.0084 | 0.0077 | 0.0134 | 1.09x |
| Phase lock | **0.796** | 0.642 | 0.600 | 1.24x |
| Delta phi | -3.08 rad | -2.73 rad | -0.08 rad | — |
| Joint 0 freq | 2.22 Hz | 2.46 Hz | 0.24 Hz | 0.90x |
| Joint 1 freq | 1.32 Hz | 2.22 Hz | 0.42 Hz | 0.59x |
| BackLeg work | 3,387 | 3,506 | 1,309 | 0.97x |
| FrontLeg work | 5,343 | 2,976 | 648 | 1.80x |
| Duty back leg | 0.205 | 0.191 | 0.217 | 1.07x |
| Duty front leg | 0.037 | 0.037 | 0.069 | 0.99x |

## The Headline: Net Distance Is 73 Meters

The Novelty Champion doesn't walk in a straight line. It curves diagonally from the origin to (60, -42), covering a net distance of 73.3 meters — 46% more than the CPG Champion's 50.1m. The yaw_net is only +0.068 rad, so the robot barely rotates — it simply launches along a diagonal heading and holds it. The DY = -42m is not from turning; it's from the initial heading angle.

This changes the framing. "20% more DX" undersells it. In total ground covered, the Novelty Champion outperforms the CPG Champion by nearly half again.

## Finding 1: It's a Runner, Not a Walker

The speed CV = 0.27 is the lowest of all three gaits (CPG: 0.42, Trial 3: 0.39). The Novelty Champion maintains the most consistent velocity — no significant acceleration or deceleration phases. The X-vs-time plot (fig 1) shows a nearly perfect straight line from 0 to 60 meters.

The contact pattern (fig 3) confirms this: ground contacts are extremely brief. The duty cycle numbers (20% back leg, 3.7% front leg) are comparable to the CPG Champion, but the contact events are shorter and more frequent — brief toe-taps rather than sustained stance phases. This is the signature of a running gait: short ground contacts, fast turnover, constant speed.

## Finding 2: The Most Coordinated Gait

Phase lock = 0.796, substantially higher than both the CPG Champion (0.642) and Trial 3 (0.600). This is the most tightly coordinated gait of the three.

Delta phi = -3.08 rad ≈ -π means the legs alternate perfectly in anti-phase: when one extends, the other retracts. This is the classical alternating-leg running gait. The CPG Champion also alternates (δφ = -2.73) but less perfectly. Trial 3 shuffles with both legs nearly in-phase (δφ = -0.08).

The phase portrait (fig 5) tells the story visually. The Novelty Champion traces a wide, regular orbit in joint space — high amplitude, repeatable cycles. The CPG Champion's orbit is tighter and more erratic. Trial 3's orbit is a narrow diagonal line (correlated, not alternating).

## Finding 3: The Front Leg Does the Heavy Lifting

| Joint | Novelty Champ | CPG Champ | Trial 3 |
|---|---|---|---|
| BackLeg work | 3,387 (39%) | 3,506 (54%) | 1,309 (67%) |
| FrontLeg work | 5,343 (61%) | 2,976 (46%) | 648 (33%) |

The Novelty Champion is front-leg dominant: the front leg does 61% of the total work, spending 1.8x more energy than the CPG Champion's front leg. This is the opposite of both comparisons — the CPG Champion is roughly balanced, and Trial 3 is back-leg dominant.

The cumulative work plot (fig 4) shows this clearly: the front leg's work curve rises steeply and consistently, while the back leg's curve is gentler. The front leg is the primary driver of locomotion.

## Finding 4: Polyrhythmic Oscillation

The FFT spectra (fig 7) reveal something unexpected: the two joints oscillate at **different frequencies**.

| Joint | Novelty Champ | CPG Champ | Trial 3 |
|---|---|---|---|
| BackLeg (j0) | 2.22 Hz | 2.46 Hz | 0.24 Hz |
| FrontLeg (j1) | 1.32 Hz | 2.22 Hz | 0.42 Hz |

The CPG Champion's joints are nearly frequency-matched (2.46/2.22 ≈ 1.1:1) — a standard 1:1 locked oscillation. The Novelty Champion's frequency ratio is 2.22/1.32 ≈ 5:3, suggesting a **polyrhythmic gait** where the back leg completes 5 cycles for every 3 of the front leg.

Despite this frequency mismatch, the phase lock is 0.796 — high. This is possible because phase locking can exist between signals at rational frequency ratios (5:3 entrainment). The Hilbert-transform phase lock score captures this: the instantaneous phase relationship remains stable even though the oscillation frequencies differ.

This polyrhythmic coordination is not something a simple CPG (Central Pattern Generator) would produce. CPGs typically enforce 1:1 frequency locking through reciprocal inhibition. The Novelty Champion's 5:3 rhythm emerges from the reactive sensor-motor loop interacting with the body's contact dynamics — a more complex coordination than the CPG Champion's engineered oscillator.

## Finding 5: The Weight That Breaks the Box

```
Sensor 0 (Torso):    w03=-1.308  w04=-0.343  sum=-1.651  diff=-0.965
Sensor 1 (BackLeg):  w13=+0.833  w14=-0.376  sum=+0.457  diff=+1.209
Sensor 2 (FrontLeg): w23=-0.037  w24=+0.438  sum=+0.401  diff=-0.474

Total drive to BackLeg motor (m3): -0.512
Total drive to FrontLeg motor (m4): -0.281
```

**w03 = -1.308** is outside the canonical [-1, 1] range. Every prior experiment — the 116 zoo gaits, the 500 random trials, the cliff analysis — constrained weights to [-1, 1]. The Novelty Seeker's perturbation process (adding a random unit vector × radius without clamping) accidentally discovered that the landscape extends usefully beyond the box.

This single weight makes the torso→back-leg connection extremely strong and inhibitory: when the torso sensor fires, the back leg gets a powerful retraction signal. Combined with w13 = +0.833 (back leg sensor strongly excites back leg motor), this creates a sharp contrast:
- **Torso contact → strong back leg retraction** (protective: "torso down, pull in")
- **Back leg contact → strong back leg extension** (propulsive: "foot down, push off")

The front leg is almost disconnected from the torso sensor (w04 = -0.343, moderate) and from the back leg sensor (w23 ≈ 0, effectively zero). It runs on its own feedback loop: w24 = +0.438 (front leg contact excites front leg motor). The front leg is semi-autonomous.

This creates an asymmetric architecture: the back leg is tightly coupled to the torso, the front leg is semi-independent, and the two interact mainly through body mechanics (ground reaction forces and body inertia) rather than through neural connections.

## Finding 6: Speed vs Efficiency Tradeoff

| | Distance | Energy | Efficiency |
|---|---|---|---|
| Novelty Champ | 73.3m | 8,731 | 0.0084 |
| CPG Champ | 50.1m | 6,482 | 0.0077 |
| Trial 3 | 26.2m | 1,956 | 0.0134 |

The three champions form a spectrum:
- **Trial 3**: Low speed, low energy, highest efficiency. The lazy walker.
- **CPG Champion**: Medium speed, medium energy. The engineered oscillator.
- **Novelty Champion**: High speed, high energy, moderate efficiency. The power runner.

The Novelty Champion spends 35% more energy than the CPG Champion and gets 46% more distance. Its marginal efficiency (the extra distance per extra energy spent) is (73.3 - 50.1) / (8731 - 6482) = 23.2m / 2249 work = 0.0103 — actually *more* efficient than the CPG Champion's 0.0077. The extra energy is being spent more productively than the CPG Champion's baseline energy.

## The Three Control Strategies

Each champion represents a fundamentally different approach to locomotion:

**Trial 3 — The Passive Exploiter** (0.24 Hz, DX=26m)
Barely participates. Weak weights, low frequency, lets the body's natural rocking dynamics carry it forward. Maximum efficiency through minimum intervention.

**CPG Champion — The Forced Oscillator** (2.46 Hz, DX=50m)
Hidden-layer half-center oscillator that discovered the body's resonant frequency and drives it hard. 1:1 frequency-locked gait. Maximum displacement through resonant forcing.

**Novelty Champion — The Polyrhythmic Runner** (2.22/1.32 Hz, DX=60m)
5:3 polyrhythmic coordination with semi-autonomous front leg. Drives even harder than the CPG Champion, but with an asymmetric, frequency-mismatched pattern that extracts more distance per cycle. Maximum speed through polyrhythmic coordination.

The progression is clear: from passive to forced to polyrhythmic. Each step up the performance ladder requires more complex coordination — not in the network architecture (the Novelty Champion has simpler topology than the CPG Champion) but in the dynamic pattern that emerges from sensor-motor-body interaction.

## Research Implications

### 1. The [-1, 1] constraint was a hidden ceiling

The champion's w03 = -1.308 proves that the canonical weight range was artificially limiting performance. A systematic exploration of the extended range [-2, 2] or even [-3, 3] is warranted. How much better can we do if we remove the box?

### 2. 6-synapse > hidden-layer for this body

The Novelty Champion (6 synapses, 60.2m DX) surpasses the CPG Champion (hidden-layer architecture, 50.1m DX). The barrier to performance was search, not architecture. For this simple 3-link body, the 6D weight space contains points that outperform any hidden-layer design found so far. The CPG architecture's advantage (internal oscillator) is actually a constraint — it enforces 1:1 frequency locking that prevents the polyrhythmic coordination that the Novelty Champion achieves.

### 3. Polyrhythmic gaits are a new category

The 5:3 frequency ratio between joints is not represented in the 116-gait zoo. The zoo's gaits are overwhelmingly 1:1 locked (high phase lock) or incoherent (low phase lock). The Novelty Champion occupies a new region: high phase lock at a non-1:1 frequency ratio. This suggests the zoo's 7 morphological categories may be missing a "polyrhythmic" category.

### 4. The front-leg hypothesis

The Novelty Champion's front-leg-dominant energy profile (61% of total work) is unique. Trial 3 is back-leg dominant, the CPG Champion is balanced. Why does front-leg dominance produce faster locomotion? Possibly because the front leg acts as a *pulling* rather than *pushing* mechanism — extending forward and pulling the body after it, rather than the back leg pushing the body forward. The semi-autonomous front leg (w23 ≈ 0) may be key: freed from back-leg interference, it runs its own contact-extension cycle at 1.32 Hz, providing a steady pull.

### 5. Novelty search as a general tool for this landscape

The Walker Competition showed that on the cliff-riddled 6D landscape, novelty search outperforms hill climbing, ridge walking, Pareto optimization, and ensemble methods. This is not a novelty-search-specific result — it's a landscape-specific result. The landscape has many deep, narrow basins separated by cliffs. Any algorithm that explores broadly (large step size, non-fitness-driven acceptance) will outperform algorithms that exploit locally. Novelty search happens to be the purest exploration algorithm, which is why it won.

## Figures

| Figure | File | Description |
|---|---|---|
| 1 | `champ_fig01_trajectory.png` | XY trajectory, X vs time, vertical bounce — three champions overlaid |
| 2 | `champ_fig02_joints.png` | Joint positions and velocities — three-panel comparison |
| 3 | `champ_fig03_contacts.png` | Contact patterns over time — three-panel comparison |
| 4 | `champ_fig04_energy.png` | Instantaneous power and cumulative work by joint |
| 5 | `champ_fig05_phase.png` | Phase portraits (j0 vs j1) colored by time, with phase lock scores |
| 6 | `champ_fig06_rotation.png` | Roll/pitch/yaw angular velocity — 3×3 grid |
| 7 | `champ_fig07_fft.png` | FFT spectra of joint angles with peak annotations |

## What's Next

The most actionable follow-up: **extend the weight range**. The Novelty Champion found performance outside [-1, 1] by accident. Running the Novelty Seeker with weights initialized in [-2, 2] and unclamped perturbation would systematically explore this extended landscape. If w03 = -1.31 is good, is w03 = -2.0 better?

Second: **the hybrid walker**. The competition showed the Novelty Seeker hits its best gait early (~150 evals) then wastes the remaining 850. A two-phase strategy — novelty search for global exploration (200 evals), then hill climbing from the best point found (800 evals at small radius) — would combine the Novelty Seeker's basin-finding ability with the Hill Climber's fine-tuning.

Third: **video**. This gait needs to be seen. The polyrhythmic 5:3 coordination, the diagonal trajectory, the front-leg pull — these are physical phenomena best understood visually.
