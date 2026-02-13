# Hilbert-Space Gaits: Multi-Frequency CPG Design

## Motivation

The 23 multi_frequency gaits in the complex attractor subtype have multiple competing spectral peaks, but none were designed that way. Can we deliberately construct gaits where multiple distinct frequencies are built in as controllable design parameters — where frequency content is a basis decomposition rather than an emergent accident?

## Approach

### 1. Spectral census of existing zoo

Extract full spectra from all 116 gaits (not just dominant frequency — top 4 peaks per joint). Cluster gaits by spectral profile to see if there are natural frequency "chords." Ask: which frequency pairs co-occur? Do certain weight motifs reliably produce certain spectral signatures?

### 2. Hidden-layer architectures as oscillator banks

A single half-center oscillator (gaits 43/44) produces one dominant frequency. Stack multiple half-center pairs — each tuned to a different frequency via their reciprocal inhibition weights — so motor neurons receive a superposition. This is how biological CPGs work: pools of oscillators at different intrinsic frequencies, coupled to produce complex rhythms.

Design topologies with 2, 3, and 4 oscillator pairs. Each pair is a basis function; the motor weights are the Fourier coefficients.

### 3. Resonance-informed target frequencies

Use resonance_mapping.py data (the body's mechanical transfer function) to identify which frequencies the body responds to preferentially. A Hilbert-space gait should excite multiple resonant modes simultaneously. The resonance data tells us *which* frequencies to build in — e.g., a gait hitting the body's first and third harmonics simultaneously would be qualitatively different from one hitting first and second.

### 4. Spectral fitness function for search

Instead of optimizing for displacement, define fitness in the frequency domain: "maximize spectral power at target frequencies f1 and f2 in the joint velocity signal, minimize power elsewhere." Evolutionary search finds weight configurations that produce clean multi-frequency gaits rather than spectrally messy accidental ones.

### 5. Superposition linearity test

The deep question: does weight-space superposition map to behavior-space superposition in this nonlinear system? For hidden-layer architectures it should, approximately — the hidden oscillators are somewhat independent before they hit the motor neurons. For standard 6-synapse or crosswired topologies, the nonlinearity of the body probably shatters clean spectral control. The hidden layer buys something closer to actual orthogonal decomposition.

## Experimental plan

1. Extract full spectra from all 116 gaits (top 4 peaks per joint, relative amplitudes)
2. Cluster gaits by spectral profile — find natural frequency "chords"
3. Design 3-4 hidden-layer topologies with 2, 3, 4 oscillator pairs
4. Use resonance mapping peaks as target frequencies
5. Search with spectral fitness to tune oscillator weights
6. Characterize resulting gaits with Beer-framework analytics — do multi-frequency CPGs open a new region of gaitspace?
