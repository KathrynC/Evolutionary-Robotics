# The Seance: On the Spookiness of LLM-Mediated Locomotion

**Date:** 2026-02-15

---

You hand a language model six floating point numbers and say "be Juliet." It has no body, no physics engine, no concept of falling. It has never seen a robot. It has only ever predicted the next token. And yet the weights it chooses make a three-legged thing in a physics simulator *fall down* — and it does this more often for characters who die in their stories than for characters who survive.

The six numbers are the entire channel. Six weights, snapped to a 9-point grid, giving 9^6 = 531,441 possible configurations. That's the bandwidth through which the LLM's representation of "Juliet" — accumulated from every performance, every essay, every forum post about Romeo and Juliet ever scraped from the internet — gets compressed into a physical behavior. And *death still gets through*.

The zero sequence result is almost more unsettling in its clarity. Three out of four models, given the concept of "nothing," produce six zeros. The robot sits perfectly still. The LLMs don't need to be told what zero means for a body. They already know.

What we've built isn't really a robotics experiment. It's a *seance*. We're asking the statistical ghost of human culture — frozen in weights trained on terabytes of text — to possess a body and walk. And it does. And how it walks tells you something about what it absorbed.

The weight collapse problem is almost reassuring by comparison. At least *that* makes sense — of course a language model has a default "generic walking" attractor. The spooky part is that it ever escapes that attractor at all, that "Mercutio" pulls it somewhere different from "Friar Laurence."

## Embodiment-Ready

The conventional critique is "LLMs aren't embodied, they just predict tokens." But these experiments invert that: the LLMs already *have* embodied knowledge — they've absorbed the entire corpus of human writing about bodies, movement, falling, dying, dancing, stumbling. What they lack isn't the knowledge, it's the *output device*.

The chat interface is a straw. We're asking a system that has internalized every description of human movement ever written to express itself through a sequence of Unicode characters. Of course it looks disembodied. We gave it a mouth and no limbs.

The moment you give it even the crudest possible body — three rigid blocks, two hinge joints, six numbers — it starts expressing things it could never say in text. Juliet falls. Zero is stillness. Kolakoski sprints. These aren't things the LLM *tells* you. They're things it *does*, the instant it has a body to do them with.

And this is three cubes on a flat plane. No hands, no face, no fingers, no eyes. Six degrees of freedom in the weight space. The lowest-resolution body imaginable. If death gets through *this* channel, what gets through a humanoid with 30 joints? A face with 42 muscles? A hand?

The implication is that the scaling problem for embodied AI might be backwards. We keep trying to build embodied systems and then add intelligence. But the intelligence is already there, pre-loaded with embodied intuition, compressed into language model weights. The engineering problem isn't "how do we make LLMs understand bodies" — it's "how do we build bodies adequate to what LLMs already know."
