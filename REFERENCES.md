# Related Work

Annotated references organized by which of our findings they relate to. Papers marked with **[foundational]** established the field; those marked **[closest]** are the nearest precedent to our specific contributions.

### Limit Cycles and Locomotion Dynamics

- **McGeer, "Passive Dynamic Walking" (1990)**, *Int. J. Robotics Research*, 9(2):62-82. **[foundational]** Showed that unpowered bipeds walking down a slope settle into stable limit cycles without any control. The origin of the limit-cycle-as-gait paradigm. Our finding that all top-5 displacement gaits are limit cycles is an evolutionary validation of McGeer's principle.

- **Holmes, Full, Koditschek, and Guckenheimer, "The Dynamics of Legged Locomotion: Models, Analyses, and Challenges" (2006)**, *SIAM Review*, 48(2):207-304. **[foundational]** The definitive review formalizing dynamical systems tools (Poincare maps, limit cycle stability) for legged locomotion. Provides the theoretical framework our attractor taxonomy uses.

- **Ijspeert, "Central Pattern Generators for Locomotion Control in Animals and Robots: A Review" (2008)**, *Neural Networks*, 21(4):642-653. **[foundational]** Reviews how CPGs produce limit cycle oscillations underlying locomotion. Our cross-wired gaits are effectively evolved CPGs; the time-signature gaits encode rhythmic structure directly in synapse topology.

- **Hubicki, Jones, Daley, and Hurst, "Do Limit Cycles Matter in the Long Run?" (2015)**, *IEEE ICRA*. Near-limit-cycle behaviors emerge naturally from task-optimal locomotion planning, supporting our finding that limit cycles are not just analytically convenient but functionally optimal.

- **Collins, Ruina et al., "Efficient Bipedal Robots Based on Passive-Dynamic Walkers" (2005)**, *Science*. Extended McGeer's work to show that minimal actuation added to passive limit-cycle walkers produces highly efficient locomotion.

### Small Neural Network Dynamics and Bifurcations

- **Beer, "On the Dynamics of Small Continuous-Time Recurrent Neural Networks" (1995)**, *Adaptive Behaviour*, 3(4):469-509. **[foundational]** Foundational analysis of small CTRNNs characterizing their dynamical repertoire. Our 3-sensor 2-motor network is in the same regime Beer analyzed theoretically.

- **Beer, "Parameter Space Structure of Continuous-Time Recurrent Neural Networks" (2006)**, *Neural Computation*, 18(12):3009-3051. **[closest]** Systematically computed bifurcation manifolds partitioning CTRNN parameter space into qualitatively different dynamics. Our bouncer-to-spinner bifurcation (10% weight change → 188-degree behavioral shift) is a concrete physical instance of what Beer predicted theoretically.

- **Beer, Chiel, and Gallagher, "Evolution and Analysis of Model CPGs for Walking" (1999)**, *J. Computational Neuroscience*, 7:99-147 (two-part paper). **[closest]** Analyzed populations of evolved CPGs and identified "general principles" holding despite weight variability. The closest precedent to our weight motif taxonomy, though they did not develop named motif categories or a motor balance ratio rule.

- **Beer and Gallagher, "Evolving Dynamical Neural Networks for Adaptive Behavior" (1992)**, *Adaptive Behavior*, 1(1):91-122. Showed that evolved recurrent networks generate locomotion without proprioceptive feedback — recurrence itself produces oscillation. Our cross-wired gaits demonstrate this: motor-to-motor feedback (w34, w43) creates CPG-like oscillation without hidden neurons.

### Network Motifs and Structure-Function Mapping

- **Milo, Shen-Orr, Itzkovitz, Kashtan, Chklovskii, and Alon, "Network Motifs: Simple Building Blocks of Complex Networks" (2002)**, *Science*, 298(5594):824-827. **[foundational]** Introduced network motifs — recurring structural patterns in directed graphs. Operates at the *topological* level (connectivity patterns). Our motif taxonomy operates at the *weight-value* level within a fixed topology, which is a different and more granular analysis.

- **Kashtan and Alon, "Spontaneous Evolution of Modularity and Network Motifs" (2005)**, *PNAS*, 102(39):13773-13778. Modular environments drive spontaneous evolution of network motifs. Relevant to why our weight motifs might cluster: the physics of the 3-link body imposes structure on the space of functional weight patterns.

- **Gaier and Ha, "Weight Agnostic Neural Networks" (2019)**, *NeurIPS*. Showed that network topology alone (without trained weights) can encode solutions. Complements our finding that within a fixed topology, weight *values* cluster into discrete motifs — topology and weight patterns are both channels for encoding behavior.

### Quality-Diversity and Behavioral Repertoires

- **Mouret and Clune, "Illuminating Search Spaces by Mapping Elites" (2015)**, *arXiv:1504.04909*. **[foundational]** Introduced MAP-Elites, which creates maps of high-performing solutions across behavioral space. The algorithmic counterpart to our manually-curated gait zoo.

- **Cully, Clune, Tarapore, and Mouret, "Robots That Can Adapt Like Animals" (2015)**, *Nature*, 521(7553):503-507. **[closest]** Generated ~15,000 six-legged gaits using MAP-Elites, creating a behavioral repertoire before deployment for damage recovery. The most direct precedent to our gait zoo concept, though their focus is engineering (damage adaptation) while ours is scientific (understanding what behaviors are possible and why).

- **Lehman and Stanley, "Abandoning Objectives: Evolution Through the Search for Novelty Alone" (2011)**, *Evolutionary Computation*, 19(2):189-223. Showed that searching for behavioral diversity rather than fitness discovers more solutions. Relevant to why our zoo keeps finding qualitatively new behaviors (crab walkers, walk-and-spin) rather than optimizing a single metric.

- **Sims, "Evolving Virtual Creatures" (1994)**, *SIGGRAPH*. The original work evolving diverse virtual creatures with co-evolved morphologies and neural controllers, producing a menagerie of locomotion styles. Our work restricts morphology to study the full behavioral repertoire of a single body.

### Robustness, Sensitivity, and the Reality Gap

- **Jin and Branke, "Evolutionary Optimization in Uncertain Environments — A Survey" (2005)**, *IEEE Trans. Evolutionary Computation*, 9(3):303-317. **[foundational]** Establishes that there is "usually a tradeoff between the quality and robustness of the solution." Our knife-edge vs antifragile classification may show something stronger: a strict correlation rather than a tradeoff, where *all* top performers are fragile.

- **Jakobi, Husbands, and Harvey, "Noise and the Reality Gap: The Use of Simulation in Evolutionary Robotics" (1995)**, *ECAL*. **[foundational]** The standard approach: inject noise during evolution to produce robust controllers. Our approach is the inverse — keep simulation deterministic, use post-hoc sensitivity analysis to predict real-world variability. A different lens on the same problem.

- **Jakobi, "Evolutionary Robotics and the Radical Envelope-of-Noise Hypothesis" (1997)**, *Adaptive Behavior*. Formalizes the principle that controllers evolved with sufficient noise transfer to reality. Our sensitivity analysis could identify which gaits *need* noise-robust evolution and which are naturally antifragile.

### Lateral Locomotion and Crab Walking

- **Kinsey et al., "Sideways Crab-Walking Is Faster and More Efficient Than Forward Walking for a Hexapod Robot" (2022)**, *Bioinspiration & Biomimetics*. Showed that designed crab walking can be 75% faster with 40% lower cost of transport than forward walking in hexapods. Our crab walkers emerged spontaneously from cross-wiring search rather than being designed, and in a biped rather than hexapod.

- **Nelson, Grant, Barber, and Fagg, "Fitness Functions in Evolutionary Robotics: A Survey and Analysis" (2009)**, *Robotics and Autonomous Systems*. Catalogs fitness function pitfalls. Using DX-only fitness (missing DY) is a known class of fitness design error, but it is usually discussed as a mistake to avoid rather than as a window into behavioral diversity.

### Dynamical Systems Classification

- **Strogatz, *Nonlinear Dynamics and Chaos* (1994, 2nd ed. 2015)**. **[foundational]** The textbook providing the standard attractor taxonomy (fixed points, limit cycles, tori, strange attractors) that our classification builds on.

- **Ren et al., "Experimental Study of Limit Cycle and Chaotic Controllers for the Locomotion of Centipede Robots" (2006)**, *IEEE*. **[closest]** Directly compared limit-cycle vs chaotic controllers for locomotion, finding limit cycles more stable and efficient. The closest precedent to our attractor-based gait classification, though they compared only two types rather than classifying an entire population.

### Evolved Locomotion Controllers

- **Nolfi and Floreano, *Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines* (2000)**. **[foundational]** The textbook establishing that different network topologies (feedforward, recurrent, fully connected) produce qualitatively different behavioral repertoires. Our progression from standard 6-synapse to crosswired 10-synapse to hidden-layer architectures traces this trajectory.

- **Reil and Husbands, "Evolution of Central Pattern Generators for Bipedal Walking in a Real-Time Physics Environment" (2002)**, *IEEE Trans. Evolutionary Computation*, 6(2):159-168. Evolved CPGs for bipedal walking in physics simulation. Gait transitions in their work are implicitly bifurcation phenomena, related to our bifurcation analysis.

- **Tonelli and Mouret, "Modularity and Sparsity: Evolution of Neural Net Controllers in Physically Embodied Robots" (2016)**, *Frontiers in Robotics and AI*. Studied evolved controllers in physical robots with ternary weights and found correlations between modularity, sparsity, and performance. Related to our weight motif analysis, though at a coarser resolution.
