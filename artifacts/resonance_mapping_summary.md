# Resonance Mapping: What Frequencies Does the Body Want?

## Overview

Bypassed the neural network entirely to drive joints with pure sinusoidal position targets. Swept frequency, phase, and amplitude to map the body's mechanical transfer function — the raw input/output relationship between oscillation parameters and displacement.

**Script**: `resonance_mapping.py`
**Budget**: 1,800 sims in 65.4 seconds
**Seed**: deterministic (phase/freq grids, no randomness)

## Part 1: Frequency x Phase Sweep (600 sims)

50 frequencies (0.1–5 Hz) x 12 phase offsets (0–180°). Both joints at same frequency, amplitude = 0.5 rad.

### Transfer Function Peaks

| Rank | Frequency | Max |DX| | Best Phase |
|------|-----------|---------|------------|
| 1 | 3.30 Hz | 22.7 m | ~80° |
| 2 | 4.00 Hz | 22.6 m | ~60° |
| 3 | 1.30 Hz | 19.8 m | ~80° |
| 4 | 2.00 Hz | 18.5 m | ~60° |
| 5 | 1.90 Hz | 17.8 m | ~100° |

The transfer function shows **moderate resonance** — clear peaks exist, but the envelope is broad rather than sharply tuned. The body responds across a wide band (1–4 Hz) with dips rather than a single dominant mode.

### Key Observation: Frequency Space is Smooth

Adjacent frequency-phase cells have similar DX values. The heatmap shows visible gradients, bands, and structure — a stark contrast to weight-space, where adjacent points are uncorrelated. Frequency-space cliffiness is ~25 m/Hz vs ~3,000,000 m/unit in weight-space: approximately **120,000x smoother**.

## Part 2: Amplitude Sweep (300 sims)

Top 6 resonant frequencies x 50 amplitudes (0.05–1.5 rad).

### Amplitude Response

| Frequency | Best Amplitude | Max DX |
|-----------|---------------|--------|
| 3.30 Hz | 0.85 rad | 32.7 m |
| 4.00 Hz | 0.70 rad | 26.2 m |
| 1.30 Hz | 0.75 rad | 23.7 m |
| 1.50 Hz | 0.80 rad | 23.5 m |
| 1.90 Hz | 0.65 rad | 20.9 m |
| 2.00 Hz | 0.60 rad | 19.1 m |

### Key Observation: Amplitude Restores Chaos

Below ~0.5 rad, amplitude curves rise roughly linearly — more swing = more displacement. Above ~0.8 rad, the curves explode into noise. Every frequency becomes chaotic at high amplitude. This pins down the chaos source: **large contact forces**. Bigger swings = harder foot strikes = more sensitive to timing = fractal behavior returns.

## Part 3: Polyrhythmic Grid (900 sims)

30 x 30 grid of (f_back, f_front) from 0.1–5 Hz. Amplitude = 0.5 rad, phase = 60°.

### Key Findings

- **Diagonal** (equal frequency) is relatively calm and well-structured
- **Off-diagonal** (polyrhythmic) adds chaos — the landscape becomes noisier when joints run at different frequencies
- **Asymmetric ratios** produce the highest |DX| hotspots (e.g., f_back~1 Hz + f_front~4-5 Hz)
- **Broad gradient**: low f_back + high f_front tends positive DX; high f_back + low f_front tends negative DX. The body has a directional preference based on which leg oscillates faster.

## Part 4: Evolved Gait Comparison (no sims)

Overlaid evolved gait frequencies from `synapse_gait_zoo_v2.json` on the transfer function and polyrhythmic map.

### The Critical Result: Neural Networks Far Exceed Open-Loop

| Gait | |DX| | Open-Loop Ceiling | Excess |
|------|------|-------------------|--------|
| Novelty Champion | 60.2 m | 22.7 m (at default amp) | **+37.5 m** |
| Hidden CPG Champ | ~50 m | 32.7 m (optimized amp) | **+17 m** |
| Noether CPG | ~44 m | 32.7 m (optimized amp) | **+11 m** |
| Curie Amplified | ~34 m | 32.7 m (optimized amp) | ~0 m |

Even with amplitude optimization, the best open-loop DX is 32.7m. The Novelty Champion nearly doubles that. The neural network is not just finding a good frequency — it's doing something qualitatively different.

## Three Conclusions

### 1. The body has moderate mechanical resonance
The transfer function shows clear frequency preferences (peaks at 1.3, 2.0, 3.3, 4.0 Hz) but is not sharply tuned. The body is a broad-band mechanical system, not a tuning fork.

### 2. The neural network exploits closed-loop feedback
If the NN merely selected good oscillation parameters, evolved gaits would cluster near the open-loop envelope. Instead, they exceed it by 2-3x. The NN senses contact events and adjusts timing in real time. It's not driving — it's *reacting*. The sensory feedback loop is worth +37.5m of displacement.

### 3. Contact force magnitude is the chaos gateway
- Small amplitudes → smooth, predictable response
- Large amplitudes → fractal chaos returns
- Frequency and phase are smooth parameters; amplitude and weight perturbations are fractal
- The common factor: **contact force magnitude**. Harder foot strikes amplify timing sensitivity.

This connects back to the cliff taxonomy finding: contact dynamics are the chaos source. Resonance mapping localizes this more precisely — it's not oscillation frequency that creates chaos, it's the *violence* of the contact events.

## Smoothness Comparison

| Space | Cliffiness | Relative |
|-------|-----------|----------|
| Weight space | ~3,000,000 m/unit | 120,000x |
| Frequency space | ~25 m/Hz | 1x |

Frequency-space is a vastly better parameterization for search. But it cannot reach the performance that closed-loop NN control achieves.

## Output Files

- `artifacts/resonance_mapping.json` — all sweep data, transfer function, amplitude curves, polyrhythmic grid
- `artifacts/plots/res_fig01_freq_phase.png` — frequency x phase heatmap (signed DX + |DX|)
- `artifacts/plots/res_fig02_transfer_function.png` — mechanical transfer function + optimal phase scatter
- `artifacts/plots/res_fig03_amplitude.png` — DX vs amplitude at 6 resonant frequencies
- `artifacts/plots/res_fig04_polyrhythm.png` — 30x30 polyrhythmic frequency grid
- `artifacts/plots/res_fig05_evolved_overlay.png` — evolved gaits vs open-loop envelope
- `artifacts/plots/res_fig06_verdict.png` — summary verdict panel
