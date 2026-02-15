# Prompt Engineering Analysis: Improving LLM Weight Generation

**Date:** 2026-02-15
**Sources:** ChatGPT response (shared in conversation) + DeepSeek session (`~/Desktop/Improve LLM robot motion word mapping - DeepSeek.pdf`, 5 pages, 2:53 AM, `https://chat.deepseek.com/a/chat/s/87f51ce3-8f76-406b-a344-fa9cbef440d0`)

**Note:** The same 5 questions were posed to both ChatGPT and DeepSeek independently. Both produced essentially identical recommendations — same structure, same advice, same examples. This convergence increases confidence in the recommendations but also illustrates the limitation: LLMs agree on the obvious prompt improvements, but neither addresses the deeper architectural bottleneck.

## Context

Analysis of the LLM-to-robot weight generation pipeline's asymmetric success rates: **9% for "forward_walk"** versus **74% for "stumble"**. This suggests LLMs have strong intuitive priors about failure modes (stumbling, falling) but struggle with controlled locomotion (directed walking, precise gaits).

## Key Recommendations

### 1. Few-Shot Examples (strategic, not exhaustive)

Include 2-3 examples demonstrating contrasting behaviors. Avoid similar motions (don't include both "walk" and "glide" — causes blending).

Recommended few-shot set:
- A successful "stumble" weight vector
- A successful "stand_still" (all zeros — the brilliant discovery that the null vector produces standing)

### 2. Explicit Weight Semantics

The current prompt treats weights as abstract numbers. Both ChatGPT and DeepSeek independently propose exposing the sensor-motor mapping:

| Weight | Connection | Functional role |
|---|---|---|
| w03 | torso touch → back leg motor | balance feedback |
| w13 | back leg touch → back leg motor | local reflex |
| w23 | front leg touch → back leg motor | cross coupling |
| w04 | torso touch → front leg motor | balance feedback |
| w14 | back leg touch → front leg motor | cross coupling |
| w24 | front leg touch → front leg motor | local reflex |

This helps LLMs reason about coordination patterns rather than guessing numbers.

### 3. Behavioral Criteria (qualitative, not quantitative)

Don't give exact thresholds (overfits to simulation specifics). Give qualitative descriptions:
- "forward walk": moves forward > backward, alternating leg motion
- "spin": rotates in place, minimal forward translation
- "glide": smooth continuous motion, minimal vertical oscillation

### 4. Structured Chain-of-Thought

CoT with guardrails to prevent hallucination:
1. What motor pattern creates [motion]? (alternating/synchronous/constant)
2. Which sensors should trigger which motors?
3. How do the weights encode this pattern?
4. Then output ONLY the JSON.

### 5. Additional Strategies

- **Weight constraints/symmetries**: For forward locomotion, suggest anti-symmetric patterns: `w03 ≈ -w04` (torso tilt drives opposite leg response)
- **Negative space examples**: Tell them what NOT to do: "Avoid symmetric weights that cause standing still"
- **Language-agnostic concepts**: Leverage the all-zero "stand_still" discovery — ask for "essence" rather than literal translation
- **Ensemble verification**: Generate multiple candidates per motion word, simulate briefly, pick best match

### 6. Revised Prompt Template

ChatGPT proposes a revised prompt that includes: robot description, weight mapping table, seed word + behavioral description, strategic examples, and a structured reasoning step before JSON output.

## Assessment

### What's Valuable

- The weight semantics table is immediately actionable — the current prompt doesn't explain what the weights *do*
- The asymmetry observation (failure modes easy, controlled motion hard) is empirically confirmed by the structured_random experiments
- Few-shot with contrasting behaviors is a sound strategy
- Chain-of-thought with structure prevents the "confident nonsense" problem

### What It Misses

- **The communication reframe**: This is still optimizing within the fixed 6-synapse / 3-sensor architecture. Better prompts can help, but the fundamental bottleneck is the nervous system's expressive bandwidth, not the prompt (see [reframing_context.md](reframing_context.md))
- **Computational irreducibility**: For Wolfram Class III regions of weight space, no amount of prompt engineering helps — the LLM can't predict the physics without running it (see [chatgpt_full_conversation_notes.md](chatgpt_full_conversation_notes.md))
- **Sensory death**: The 9% forward_walk failure rate is partly because many weight vectors produce sensory death, not because the LLM chose wrong — it's a body problem, not a prompt problem (see [sensor_design_specification.md](sensor_design_specification.md))

### Where It Fits

This analysis is useful for **incremental improvement within the current architecture** — getting more out of the existing 6D weight space before evolving the nervous system. The structured_random experiments (`structured_random_common.py`) already implement some of these ideas (semantic priming, multiple LLM conditions). The Fisher metric work (`fisher_metric.py`) measures exactly the variance that ensemble verification would exploit.

## The Experiment That Prompted This

Kathryn's summary to both LLMs:
- 500 trials, 100 seeds x 5 LLMs
- Best model: gpt-4.1-mini at 43% semantic match rate
- Easiest concepts: stumble (74%), stand_still (55%)
- Hardest concepts: forward_walk (9%), glide (10%), spin (14%)
- Problem: directional control (forward/backward) is very hard — LLMs can produce locomotion but can't control direction
- Problem: qwen3-coder collapses many different motion words to the same weight vector
- Discovery: gpt-4.1-mini found all-zero weights = no motion for "stand still" across 5 languages

Current prompt shared with both LLMs:
> "You are designing a neural controller for a 3-link robot (Torso, BackLeg, FrontLeg) with two joints. The controller has six weights [w03, w13, w23, w04, w14, w24] mapping touch sensors to motors. Generate a weight vector that makes the robot move in a way that captures the essence of the word '{seed}'. Output ONLY a JSON object with keys w03, w04, w13, w14, w23, w24, each a float in [-1, 1]. No explanation."

## Source

- ChatGPT response, shared directly in conversation (2026-02-15)
- DeepSeek session: `~/Desktop/Improve LLM robot motion word mapping - DeepSeek.pdf` (5 pages, 2026-02-15 2:53 AM)
- Related: [reframing_context.md](reframing_context.md), [chatgpt_full_conversation_notes.md](chatgpt_full_conversation_notes.md)
