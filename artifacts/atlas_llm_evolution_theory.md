# Atlas + LLM vs Evolution, and the Category Theory of Structural Transfer

## Part 1: Can Atlas + LLM beat evolution?

### The data

| Strategy | Best |DX| | Parameterization | Search type |
|---|---|---|---|
| Uniform random (100 trials) | 27.8m | Raw weights [-1,1]^6 | Unstructured |
| LLM-mediated (400 trials) | 29.2m | Raw weights via semantic seeds | Structured but conservative |
| Open-loop atlas (2,150 sims) | 32.7m | Frequency/phase/amplitude | Smooth, no feedback |
| Evolution (1,000 evals) | 60.2m | Raw weights via mutation | Closed-loop NN |

The gap between the open-loop ceiling (32.7m) and the Novelty Champion (60.2m) is the **feedback premium** — what real-time contact sensing buys you. No amount of parameter tuning in open-loop space can close it.

### Three hybrid strategies

**Strategy 1: Reparameterize the LLM's output space**

Instead of asking "translate Revelation 6:8 into 6 raw weights," ask "translate Revelation 6:8 into a frequency, phase offset, and amplitude for each leg, then derive the NN weights that would produce those oscillation patterns with feedback."

The atlas tells us the smooth manifold. The LLM navigates it semantically. The NN provides the feedback loop. This exploits both the LLM's structural mapping ability and the atlas's smoothness, while retaining closed-loop control.

**Strategy 2: LLM-seeded evolutionary search**

Use the LLM to initialize the population in high-phase-lock, zero-dead-gait regions (the "coordinated submanifold"), then evolve from there. The LLM replaces the random initialization that wastes 8% of evaluations on dead gaits and starts evolution in a region with meaningful behavioral structure.

The LLM provides coarse search (land in a good basin). Evolution provides fine-tuning (exploit the basin's peak). The combination should beat either alone on evaluation efficiency — fewer wasted trials on dead gaits and chaotic regions.

**Strategy 3: Frequency-space LLM + local optimization**

The atlas showed frequency space is 120,000× smoother than weight space. If the LLM generated (frequency, phase, amplitude) tuples instead of raw weights, its conservative sampling bias would matter less — because the smooth landscape means conservative samples are still *near* good solutions. A local optimizer (hill climber, not gradient descent — the landscape is still technically non-differentiable at contact boundaries) can then refine within the smooth manifold.

### Assessment

The honest answer: the combination *could* beat evolution on **efficiency** (fewer evaluations to find good gaits) but probably not on **peak performance**. Evolution's advantage is that it operates directly on the closed-loop controller and can discover feedback strategies that no open-loop parameterization captures. The +27.5m feedback premium isn't accessible through any atlas-based approach unless the NN weights themselves are being optimized.

The best bet is Strategy 2: LLM-seeded evolution. The LLM's 0% dead-gait rate and high phase lock (0.85-0.91) mean it starts evolution in the right neighborhood. The question is whether the coordinated submanifold that the LLM finds *contains* paths to the 60m regime, or whether reaching 60m requires traversing cliffs that the LLM's conservatism avoids.

There's an interesting possibility: the LLM's conservative region and evolution's high-performance region might be in *different basins of attraction* separated by cliffs. If so, LLM-seeded evolution would converge to a different (lower) optimum than randomly-seeded evolution, which occasionally gets lucky and lands in the 60m basin. The LLM's structure would actually be a *trap* — it starts you in a well-organized but bounded region.

This is testable. Run the walker competition's 5 algorithms but initialize from LLM-generated weight vectors instead of uniform random. If the final fitness is lower, the coordinated submanifold is a trap. If higher, it's a launchpad.

## Part 2: Formalizing the mapping

### The pipeline

```
Semantic Space  →  LLM Weights  →  Robot Weights  →  Behavioral Space
      S                L                W                  B
```

Each arrow is a map. The composition S → B is what the structured random search experiment measures. The question is whether this has categorical structure worth formalizing.

### What's preserved

The Triptych demonstrates structure preservation precisely:
- "Exact symmetry" (Noether) → symmetric weights (+0.5,-0.5), (+0.3,-0.3), (+0.7,-0.7) → stasis (DX=0.03m)
- "Cyclic perpetual motion" (Ecclesiastes) → near-periodic weights → efficient cycling (efficiency=0.00495, phase lock=0.995)
- "Asymmetric violence" (Revelation) → asymmetric weights (-0.8,+0.6), (+0.2,-0.9), (+0.5,-0.4) → maximum displacement (DX=+29.17m)

Structural principles survive the transfer: symmetry maps to symmetry, periodicity maps to periodicity, asymmetry maps to asymmetry. This isn't information transfer in the Shannon sense (bits preserved). It's **geometric structure preservation** — the *shape* of the concept transfers, not its content.

The stumble-synonym result makes this concrete: stumble (English), stolpern (German), tropezar (Spanish), and tropecar (Portuguese) all map to identical weights and identical speed (2.010). The morphism "is a translation of" collapses to identity in weight space. The LLM preserves the structural equivalence class.

### Four categories

**Sem** (Semantic): Objects are concepts (verbs, theorems, verses, places). Morphisms are structural relationships — "more symmetric than," "more periodic than," "more violent than," "is a translation of." These are partial orders and equivalence relations on structural properties.

**LLM** (LLM representation): Objects are attractor basins in the LLM's weight space — the regions activated by each concept. Morphisms are paths between basins through the LLM's internal geometry. The Arturo Ui Effect (Cramer et al. 2025) describes the formation of these basins: textual redundancy → token overrepresentation → attractor formation.

**Wt** (Weight space): Objects are weight vectors in [-1,1]^6. Morphisms are continuous paths. But the cliff findings create a problem: continuous paths in weight space do NOT necessarily produce continuous behavioral changes. The morphism structure is broken at 42% of random points.

**Beh** (Behavioral): Objects are behavioral phenotypes — tuples of (DX, speed, efficiency, phase_lock, entropy, roll_dom, yaw). Morphisms are continuous interpolations in this metric space.

### The functors

**F: Sem → Wt** (the LLM mapping)

The LLM defines a functor that maps concepts to weight vectors while preserving structural relationships:
- "More symmetric" → more symmetric weights
- "More periodic" → weights with more regular spacing
- "Translation equivalence" → identical weights
- "More violent/energetic" → larger magnitudes

F is structure-preserving in the categorical sense: it maps morphisms to morphisms. If A is structurally related to B in Sem, then F(A) is geometrically related to F(B) in Wt in a way that respects the relationship.

**G: Wt → Beh** (the simulation mapping)

This is emphatically NOT a functor on all of Wt. The cliff findings prove it: nearby weights can produce arbitrarily different behaviors. G doesn't preserve the morphism structure of "nearby in weight space → nearby in behavioral space." The derivative diverges as 1/r with no smoothness floor. G is formally non-differentiable.

### The key categorical insight: restriction to a subcategory

**The LLM's conservative sampling means F(Sem) lands in a subcategory of Wt where G is approximately a functor.**

Define **Wt_smooth** ⊂ Wt as the subcategory of weight vectors where the behavioral map G is locally continuous — where small perturbations produce small behavioral changes. These are the antifragile regions, the basins of attraction far from cliff boundaries.

The experimental evidence:
- Bible: 0% dead gaits (all weights land in viable behavioral basins)
- Places: 0% dead gaits
- Phase lock 0.85-0.91 (all weights produce coordinated, structured behavior)
- The PCA diversity plot shows structured conditions clustering in a tight region — the LLM avoids the fractal periphery

The LLM acts as a **regularizer**: it restricts the weight space to a submanifold where the simulation map becomes well-behaved. On this restricted subcategory:

```
G|_{Wt_smooth}: Wt_smooth → Beh
```

is approximately a functor, and the full composition:

```
G|_{Wt_smooth} ∘ F: Sem → Beh
```

preserves structural relationships end-to-end. This is why the meaning of each text is legible in the robot's behavior. Death rides fast, the wind cycles with minimal waste, and conservation laws conserve. The structure preservation works because the LLM keeps us on the smooth submanifold where the body's map is well-behaved.

### Sheaf theory: the local-to-global structure

The cliff findings say the weight space has a non-trivial topology. It's covered by smooth patches (basins of attraction) connected by discontinuous boundaries (cliffs). This is precisely the setting for sheaf theory.

Define a sheaf **F** on Wt that assigns to each open patch U the space of behavioral data obtainable from weights in U. Within each patch, behavioral data is continuous and well-defined (the restriction maps work). But across cliff boundaries, the sheaf sections don't extend — you can't continuously interpolate behavior across a cliff.

The global sections of this sheaf (behavioral properties that hold everywhere in weight space) are very limited — essentially just "the robot exists and has 3 links." All interesting behavioral properties (locomotion, coordination, efficiency) are *local* sections that live within specific patches.

The LLM's functor F: Sem → Wt can be understood as selecting sections of this sheaf. Each concept maps to a weight vector that lives in a specific patch, and the behavioral data associated with that concept is the sheaf section evaluated at that point. The LLM's conservatism means it selects sections from a small number of well-behaved patches, never straying into the fractal boundary regions where the sheaf structure breaks down.

The gait interpolation experiments probe the sheaf structure directly: interpolating between two weight vectors follows a path through multiple patches, and the behavioral discontinuities along the path are exactly the points where the path crosses cliff boundaries — where the sheaf fails to extend.

### Information geometry: the LLM's output manifold

The LLM's output distribution over weight space (given a prompt) defines a statistical manifold. For each concept c in Sem, the LLM produces a distribution P_c over Wt (since temperature > 0, the same prompt can produce different weights on different runs).

The Fisher information metric on this manifold measures how efficiently the LLM distinguishes between different structural concepts:

```
d(c₁, c₂) = Fisher distance between P_{c₁} and P_{c₂}
```

The stumble-synonym result says d(stumble, stolpern) ≈ 0 — the distributions are identical. The Triptych says d(Revelation, Ecclesiastes) is large — different concepts map to well-separated distributions. And d(Revelation, Noether) is even larger — the pale horse and the conservation law occupy distant regions of the LLM's output manifold.

This gives a Riemannian structure on Sem induced by the LLM: concepts are close when the LLM maps them to similar weight distributions, and far when it maps them to different distributions. The question of whether "verbs are better than places for generating gaits" becomes a question about the geometry of this induced manifold — does the verb submanifold span more of Beh than the place submanifold?

### The Yoneda perspective

The Yoneda lemma says: an object in a category is fully characterized by all the morphisms from it. In Sem, a concept like "Borges" is characterized by all its structural relationships to every other concept — its symmetry relations, its periodicity relations, its aesthetic relations.

The LLM's representation of "Borges" is precisely this: the collection of all contextual associations, weighted by training frequency, that determine how "Borges" relates to everything else in the corpus. Yoneda says this representation is *faithful* — it captures the concept completely (up to isomorphism in the category).

The functor F: Sem → Wt preserves this if and only if: the weight vector for "Borges" maintains the same structural relationships to other weight vectors that "Borges" maintains to other concepts in Sem. The stumble-synonym result is evidence that F is faithful on equivalence classes. The Triptych is evidence that F preserves the ordering on structural properties (more symmetric → more symmetric weights → more symmetric behavior).

Whether F is *fully faithful* (preserves ALL morphisms, not just some) is an open question. It would mean that every structural distinction the LLM makes between concepts results in a measurable distinction in the weight vectors. The cluster of 12 verbs sharing identical efficiency (0.003372) suggests F is NOT fully faithful — it collapses some distinctions that exist in Sem. "Wobble" and "stagger" and "sway" are different concepts with different structural nuances, but the LLM maps them to the same weight vector.

This loss of faithfulness might be a resolution issue: the 6D weight space simply doesn't have enough dimensions to represent all the structural distinctions that the LLM's internal representation captures. The LLM's representation space is billions-dimensional; the robot's weight space is 6-dimensional. The functor F is necessarily a massive dimensionality reduction, and most structural distinctions are lost in the projection.

The 10-synapse crosswired topology (10 weights) or hidden-layer topology (16+ weights) would give F more room to be faithful. This predicts that running the structured random search with crosswired or hidden-layer architectures would produce more behavioral diversity — the LLM could express more structural distinctions in the higher-dimensional weight space.

### Summary: the categorical structure

The pipeline Sem → Wt → Beh has genuine categorical structure:

1. **F: Sem → Wt** is a functor that preserves structural relationships (symmetry, periodicity, equivalence). It is faithful on equivalence classes but not fully faithful due to dimensionality reduction from billions of LLM dimensions to 6 robot weights.

2. **G: Wt → Beh** is NOT a functor on all of Wt (cliffs break morphism preservation). But restricted to **Wt_smooth** (the smooth subcategory), it is approximately a functor.

3. **The LLM acts as a regularizer**: F(Sem) ⊂ Wt_smooth, so the composition G∘F: Sem → Beh preserves structure end-to-end. The LLM's conservatism is not a limitation — it's a *feature* that keeps the pipeline on the smooth submanifold where the body's map is well-behaved.

4. **The weight space has sheaf structure**: smooth patches connected by discontinuous cliffs. The LLM selects sections from well-behaved patches. Gait interpolation probes the sheaf's gluing maps.

5. **The Yoneda perspective** suggests that richer weight spaces (more synapses) would allow the functor F to be more faithful, preserving more of the structural distinctions that the LLM encodes.

6. **The testable prediction**: LLM-seeded evolution should outperform randomly-seeded evolution on evaluation efficiency (fewer wasted trials) but may converge to a different (possibly lower) optimum if the smooth submanifold and the 60m basin are separated by cliffs.
