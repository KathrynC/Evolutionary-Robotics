# Behavioral Embryology: How Do Gaits Develop?

## Overview

Tracked the emergence of locomotion behavior across 9 cumulative time windows (50 to 4000 steps) for 19 gaits: 2 named champions (Novelty Champion, Trial 3), 7 zoo gaits loaded from existing telemetry, and 10 random-weight controls.

**Script**: `behavioral_embryology.py`
**Budget**: 12 simulations + 7 loaded from telemetry. Ran in 1.7 seconds.

## Time Windows

| Window | Steps | Time (s) |
|--------|-------|----------|
| 1 | 50 | 0.21 |
| 2 | 100 | 0.42 |
| 3 | 200 | 0.83 |
| 4 | 500 | 2.08 |
| 5 | 1000 | 4.17 |
| 6 | 1500 | 6.25 |
| 7 | 2000 | 8.33 |
| 8 | 3000 | 12.50 |
| 9 | 4000 | 16.67 |

## Finding 1: No Stumbling Phase

Contact entropy reaches 80% of its final value within 0.2 seconds for every gait tested — evolved and random alike. Contact duty cycles stabilize within the first second. The body's contact rhythm crystallizes almost immediately upon simulation start. There is no extended transient or "stumbling" period.

**Implication**: The body-ground system has inherent preferred contact modes that activate instantly regardless of the neural controller. The body finds *a* rhythm before the neural network has had time to influence it.

## Finding 2: Gait Quality is a Late-Developing Property

At 500 steps (~2.1 seconds):
- Evolved gaits mean |DX|: **2.27 m**
- Random gaits mean |DX|: **2.02 m**

At 4000 steps (~16.7 seconds):
- Evolved gaits mean |DX|: **32.41 m**
- Random gaits mean |DX|: **4.32 m**

Evolved and random gaits are **indistinguishable** in early development. The 8x performance gap only manifests over the full simulation. Small differences in neural output compound through contact dynamics over thousands of timesteps.

**Implication**: This is consistent with the fractal fitness landscape finding — the body is a chaos amplifier. Tiny differences in control signals grow exponentially through the contact cascade, producing massive displacement differences only at long timescales.

## Finding 3: The Novelty Champion Develops Slowly

| Metric | NC Onset (80%) | Mean Evolved Onset |
|--------|----------------|-------------------|
| DX | 16.7s (never reaches 80% early) | 16.7s |
| Mean speed | 4.2s | 2.5s |
| Heading consistency | 0.8s | 0.4s |
| Phase lock | 0.8s | 0.3s |

The NC takes longer than average to achieve heading consistency and phase lock. Its DX accumulation rate accelerates through the middle of the simulation (peaks at ~6-8 seconds). It's not a fast starter — it's a slow burner that builds momentum gradually.

## Finding 4: Phase Lock Does Not Predict Performance

| Group | Mean Phase Lock (final) | Mean |DX| (final) |
|-------|------------------------|-------------------|
| Evolved | 0.648 | 32.41 m |
| Random | 0.764 | 4.32 m |

Random gaits have **higher** average phase lock than evolved gaits. Rigid synchronization between joints does not produce good locomotion. The evolved gaits succeed through flexible, adaptive coordination — not through metronome-like precision.

## Onset Times (80% of Final Value)

| Gait | DX | Speed | Phase Lock | Entropy |
|------|-----|-------|------------|---------|
| Novelty Champion | 16.7s | 4.2s | 0.8s | 0.2s |
| Trial 3 | 12.5s | 0.8s | 0.2s | 0.2s |
| Hidden CPG | 16.7s | 8.3s | 0.2s | 0.2s |
| Curie Amplified | 16.7s | 2.1s | 0.2s | 0.2s |
| Pelton | 16.7s | 6.2s | 0.2s | 0.2s |
| Original | 2.1s | 0.4s | 0.2s | 0.2s |
| Noether CPG | 16.7s | 0.4s | 0.2s | 0.2s |
| Curie | 16.7s | 0.4s | 0.2s | 0.2s |
| Mordvintsev | 16.7s | 0.8s | 0.2s | 0.2s |

Pattern: Contact metrics (entropy) stabilize instantly. Coordination (phase lock) stabilizes within 1 second. Speed stabilizes in 1-8 seconds. DX almost never reaches 80% before the full simulation — displacement accumulates throughout.

## Final DX Rankings

| Rank | Gait | DX (m) |
|------|------|--------|
| 1 | Novelty Champion | +60.19 |
| 2 | Hidden CPG | +50.11 |
| 3 | Noether CPG | -43.23 |
| 4 | Curie Amplified | +37.14 |
| 5 | Pelton | +34.70 |
| 6 | Trial 3 | +25.88 |
| 7 | Curie | +23.73 |
| 8 | Random_04 | +15.62 |
| 9 | Mordvintsev | +9.47 |
| 10 | Random_03 | +8.09 |

## Connection to Previous Findings

1. **Fractal landscape** (cliff taxonomy): The late emergence of gait quality is the temporal signature of the fractal fitness landscape. Small weight differences → small early behavioral differences → large late displacement differences. The chaos amplifier needs time to work.

2. **Resonance mapping**: Contact entropy stabilizing instantly while DX takes the full simulation is consistent with the resonance finding — the body has inherent frequency preferences (fast) but performance depends on how the controller exploits them over time (slow).

3. **Phase lock paradox**: Random gaits being *more* phase-locked than evolved gaits explains why open-loop sine waves (perfectly phase-locked by construction) underperform NN control. The NN's advantage isn't coordination precision — it's adaptive timing in response to contact feedback.

## Output Files

- `artifacts/behavioral_embryology.json` — windowed analytics for all 19 gaits
- `artifacts/plots/emb_fig01_displacement_curves.png` — x(t) trajectories + DX at windows
- `artifacts/plots/emb_fig02_speed_emergence.png` — speed, heading, efficiency over time
- `artifacts/plots/emb_fig03_contact_entropy.png` — contact entropy + duty cycles
- `artifacts/plots/emb_fig04_phase_lock.png` — phase lock + frequency emergence
- `artifacts/plots/emb_fig05_developmental_fingerprints.png` — radar plots at 3 time scales
- `artifacts/plots/emb_fig06_onset_summary.png` — onset bars + evolved vs random box plots
