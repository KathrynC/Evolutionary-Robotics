# Sensor Design Specification: Expanding the Robot's Nervous System

**Date:** 2026-02-15
**Source:** ChatGPT session (Script Execution Result, 2:38 AM) + Claude Code discussion

## Context

The Motion Gait Dictionary (58 concepts, 365 entries) revealed that the robot's primary failure mode is **sensory death**: landing in a configuration with no ground contact, receiving all-zero sensor input, and freezing permanently. The current robot has only 3 touch sensors (one per link, bottom-facing, exteroceptive only).

A parallel ChatGPT session ran a DeepSeek-generated PCA visualization script on the dictionary's weight vectors, producing a scatter plot of motor primitives (explained variance: 0.282 + 0.201 = 0.483 on mock data). The conversation then pivoted to sensor design.

## The Design Request (Kathryn Cramer)

> "I think I'd like more neurons: I want neurons on the sides of each cube, a neuron on the top, so it can tell whether the top is in contact with each other. And also, I want some form of neuron so that the legs can tell when they are touching each other. One of the most interesting gaits I've seen exploited the fact that this arrangement allows the legs to overlap."

## Proposed Sensor Architecture

### Current (3 sensors, 2 motors)

```
[FrontLeg]---[Torso]---[BackLeg]
  bottom      bottom     bottom     ← 3 touch sensors (ground contact only)
```

### Proposed: Multi-Face Contact Sensors

Each of the 3 cube-shaped links gets sensors on multiple faces:

| Sensor location | Per link | Total | What it detects |
|---|---|---|---|
| **Bottom** (existing) | 1 | 3 | Ground contact |
| **Top** | 1 | 3 | Inverted configuration, mutual link contact from above |
| **Sides** (left/right) | 2 | 6 | Lateral contact, environmental obstacles |
| **Inter-limb faces** | 1 | 2* | When legs are touching each other |

*Inter-limb sensors are on the faces where adjacent links can make contact during joint rotation.

**Minimum proposed total: ~14 sensors** (up from 3), still with 2 motors.

### Why These Specific Sensors

1. **Top sensors**: The robot currently can't tell if it's upside down. Many "freeze" gaits die because the robot flips but has no signal to indicate the problem. A top sensor fires when the top face contacts the ground = the robot knows it's inverted.

2. **Side sensors**: Enable detection of lateral contact events. Currently invisible to the robot. Important for gaits that involve rolling, twisting, or lateral sliding.

3. **Inter-limb contact sensors**: The most interesting gaits exploit the fact that the hinge joints allow legs to overlap and collide with each other. Currently the robot has no way to sense this. Inter-limb sensors would let the NN respond to self-contact — a form of proprioception through collision.

### Relationship to Sensory Death

The key property of multi-face sensors: **in any physical configuration, at least some sensors will be active**. If the robot is upright, bottom sensors fire. If inverted, top sensors fire. If on its side, side sensors fire. If legs overlap, inter-limb sensors fire. There is no orientation that produces total sensory blackout.

This is the structural fix for sensory death: not proprioception in the traditional sense (joint angle sensors), but **omnidirectional exteroception** — contact sensing in every direction so that some subset of sensors is always active regardless of body configuration.

### Design Questions (Open)

- Should top/side sensors be the same modality as bottom sensors (binary contact), or something richer (force magnitude, contact normal)?
- Should the number of motors also increase, or should we first test whether richer sensing alone improves communication fidelity?
- How does the neural network topology scale? 14 sensors → 2 motors = 28 synapses in standard topology, vs. current 6. Does the LLM need a different prompting strategy for 28 weights?
- Should inter-limb sensors be symmetric (both links sense the contact) or asymmetric?

### What This Enables

With 14+ sensors and multi-face coverage:
- The robot can distinguish upright from inverted from sideways
- The NN can respond to self-collision (leg overlap), opening new gait possibilities
- Sensory death is eliminated — no configuration produces all-zero input
- The semantic-to-behavior communication channel gains bandwidth: 58 motion concepts may finally become distinguishable

## Also From This Session

A DeepSeek-generated script (`deepseek_python_20260214_105739.py`) performs PCA on the dictionary's 6D weight vectors, producing a scatter plot of motor primitives. ChatGPT ran it with mock data (40 gaits across 4 concepts: trot, hop, slither, spin). The script should be run on the real `motion_gait_dictionary_v2.json` data.

## Source

- ChatGPT session: `videos/Script Execution Result.pdf` (12 pages, 2026-02-15 2:38 AM)
- DeepSeek script referenced: `deepseek_python_20260214_105739.py`
- Related: [reframing_context.md](reframing_context.md), [gamified_progressive_search_review.md](gamified_progressive_search_review.md), [motion_gait_dictionary_compilation_notes.md](motion_gait_dictionary_compilation_notes.md)
