# Motion Gait Dictionary: Compilation Video & Key Insights

**Date:** 2026-02-15

## The Compilation Video

All 58 motion concepts from the Motion Gait Dictionary v2 were rendered as individual simulation videos and compiled into a single presentation-ready video.

### Production Pipeline

1. **Per-concept videos** (`record_concept_videos.py`): For each of 58 concepts, a video was rendered containing:
   - Title card (3s): concept name, description, entry count
   - N simulation clips: each showing the 3-link PyBullet robot executing one dictionary entry, with lower-third captions displaying the motion word, language, and LLM model
   - Credits card (4s): LLMs used, concept stats

2. **Compilation assembly** (`compile_concept_video.py`): All 58 concept videos assembled alphabetically with:
   - Opening title card (5s): "Motion Gait Dictionary" / "Semantic Motion Concepts for a Neural Walking Robot" / Kathryn Cramer / University of Vermont / date / "58 concepts - 365 entries - 5 LLMs - 5 languages"
   - Section divider cards (2s each) between concepts
   - Closing credits (10s): all 5 LLMs (gpt-4.1-mini, qwen3-coder:30b, deepseek-r1:8b, llama3.1:latest, gpt-oss:20b), prompt design consultants, key references (Beer 1996, Sims 1994, Bongard/Ludobots)

### Final Output

| Property | Value |
|---|---|
| File | `videos/motion_gait_dictionary_compilation.mp4` |
| Duration | 211.7 minutes (~3.5 hours) |
| Size | 3,708 MB (~3.7 GB) |
| Resolution | 1280x720 @ 30 fps |
| Codec | H.264 (libx264), CRF 23 |
| Segments | 118 (opening + 58 section cards + 58 concept videos + closing credits) |
| Concepts | 58 (alphabetical: amble through zigzag) |
| Total entries | 365 across 5 languages (en, de, fi, fr, zh) and 5 LLMs |

### Per-Concept Videos

58 individual videos in `videos/concepts/`, ranging from 6 MB (headstand, 1 entry) to 161 MB (patrol, 11 entries). Total disk usage for individual concept videos: ~3.6 GB.

## Key Insight: Sensory Death and the Case for Proprioception

The most important observation from watching hundreds of simulated gaits:

### The Problem: Sensory Death

Many gaits "die" mid-simulation. The robot lands in a configuration where no body link is in contact with the ground. With only 3 touch sensors (one per link, all exteroceptive), the neural network receives all-zero input. With zero input, motor outputs flatline. The robot freezes permanently.

This is **sensory death**: loss of environmental contact produces total sensory blackout. The robot has no way to feel its own body position, so once it loses ground contact, it has no information to generate recovery movements.

Evidence: "freeze" has 22 dictionary entries -- the most of any concept. A large fraction of LLM-generated weight configurations produce gaits that eventually freeze because the robot lands wrong and loses all sensory feedback.

### The Gaits That Survive

The most interesting surviving gaits are the **hoppers** -- gaits where the "torso" link is repurposed as a limb and one of the leg links becomes a de facto "head." These work precisely because the repurposed torso-as-limb keeps cycling through ground contact, keeping touch sensors alive and the neural network fed. They've accidentally solved the sensory death problem by finding a body geometry that maintains contact feedback throughout the gait cycle.

The evolutionary optimizer doesn't know which link is "supposed" to be the torso. It finds what works -- and what works is whatever configuration keeps the sensors alive.

### The Fix: Proprioceptive Sensors

The robot needs sensors that can't go dark. Proprioceptive sensors -- joint angles, angular velocity, accelerometers -- always produce non-zero signal regardless of ground contact. The robot would always feel its own body position, even mid-air.

Biological organisms have both exteroceptive (touch, vision) and interoceptive (proprioception, vestibular) sensing. You can close your eyes and still know where your limbs are. Proprioception is the sense that never goes dark.

**Evolutionary robotics is largely ignoring this failure mode.** The standard approach optimizes weights on a fixed sensor topology. But if the sensor topology creates systematic blackout conditions, no amount of weight optimization can fix it.

### The Bigger Reframe: Communication, Not Optimization

The Motion Gait Dictionary establishes a form of **call and response** between humans and robots:

1. Human provides a semantic concept ("crawl", "hop", "retreat")
2. LLM translates it into neural network weights
3. Robot executes the weights as behaviour
4. Human evaluates whether behaviour matches intent

This is a **communication channel**. The fidelity of this channel is bottlenecked not by the weights (365 examples show LLMs can generate diverse weights) and not by the vocabulary (58 concepts across 5 languages) -- it's bottlenecked by the **nervous system's expressive bandwidth**.

Three touch sensors and two motors is too impoverished a body to faithfully express the difference between "saunter" and "amble" or "stumble" and "stagger."

**The next evolutionary pressure should be on the nervous system topology itself** -- not to make the robot "better" in a fitness sense, but to increase the fidelity of the semantic-to-behaviour channel. We're not optimizing for locomotion. We're optimizing for communication.

### Implications

- The innovation substrate should shift from weights to nervous system architecture
- Proprioceptive sensors would eliminate the sensory death failure mode
- More joints/links would increase the robot's expressive bandwidth
- The goal is not a faster robot but a more articulate one -- one that can faithfully distinguish between 58+ motion concepts
- This reframes evolutionary robotics from optimization to communication theory
