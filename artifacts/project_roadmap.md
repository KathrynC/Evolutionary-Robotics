# Synapse Gait Zoo: Project Roadmap

*Updated 2026-02-13*

## Completed

### Phase 1: Landscape Characterization
- 500 random trials characterizing the 6D fitness landscape
- Cliff analysis (88% alive, cliff-riddled, fractal roughness, bimodal phase boundaries, thin Pareto frontier)
- 5 gaitspace walker algorithm designs (`artifacts/gaitspace_walker_designs.md`)

### Phase 2: Walker Competition
- 5 walkers (Hill Climber, Ridge Walker, Cliff Mapper, Novelty Seeker, Ensemble Explorer), 1,000 evals each
- **Result: Novelty Seeker wins at DX=60.2m**
- Script: `walker_competition.py`, 6 figures (`comp_fig01-06`)

### Phase 3: Novelty Champion Deep-Dive
- Deep analysis of the 60.2m gait
- Key findings: net distance 73.3m (diagonal), 5:3 polyrhythmic frequency ratio, front-leg dominant (61% of work), w03=-1.308 outside [-1,1]
- Script: `analyze_novelty_champion.py`, 7 figures (`champ_fig01-07`)

### Phase 4: Meta-Optimization Analysis
- "Can we beat evolution by stacking meta-levels?"
- Conclusion: scaling outward > stacking upward
- Written up as `artifacts/meta_optimization_analysis.md`

### Phase 5: Dark Matter Survey
- Re-analyzed 59 "dead" gaits (|DX| < 1m), clustered by full behavioral trajectory
- 5 categories: Cancellers (22%), Other (39%), Circlers (29%), Rockers (3%), Spinners (7%)
- Key insight: curvature/heading is the hidden variable separating movers from non-movers
- Script: `analyze_dark_matter.py`, 6 figures (`dark_fig01-06`)

### Phase 6: Heading Metrics Added to Pipeline
- Added `path_straightness` and `heading_consistency` to `compute_beer_analytics.py`
- Regenerated `synapse_gait_zoo_v2.json` for all 116 gaits

### Phase 7: Causal Surgery + Gait Interpolation
- 63 surgery sims (zero/half/negate each synapse for 3 champions) + 153 interpolation sims (51 points x 3 pairs)
- Key findings: w03/w13 are load-bearing for NC; halving w23 improves DX by 13.6%; Nov<->T3 landscape is cliff-riddled
- Script: `causal_surgery_interpolation.py`, 8 figures (`surg_fig01-08`)

### Phase 8: Analyze the Two Super-Gaits
- Deep comparison of 4 gaits: Interp Super (t=0.52), w23-Half Variant, Novelty Champion, Cliff Collapse (t=0.54)
- Key findings: Interp Super gets 13% more DX on 35% less energy; w23-Half is 2.5x more robust; cliff at t=0.52->0.54 is a 69.3m collapse from 0.049 total weight change
- Script: `analyze_super_gaits.py`, 6 figures (`super_fig01-06`)

### Phase 9: Resonance Mapping
- Bypassed NN entirely; drove joints with sinusoidal sweeps across frequency/phase/amplitude (~2,150 sims)
- Mapped the body's mechanical transfer function
- Script: `resonance_mapping.py`, 6 figures (`res_fig01-06`)

### Phase 10: Mid-Simulation Weight Switching (Causal Surgery v2)
- Brain transplants between champion pairs at 4 switch times, timing sweeps, synapse ablation, rescue experiments (~206 sims)
- Key findings: gaits partially converge to new controller (27-124% convergence by step 2000); w13/w14 are most critical mid-sim synapses; random brains transplanted onto champion trajectories recover only 8-21%
- Script: `causal_surgery.py`, 6 figures (`cs_fig01-06`)

### Phase 11: Behavioral Embryology
- Tracked gait emergence during the first 500+ steps for multiple gaits
- Script: `behavioral_embryology.py`

### Phase 12: Fine-Grained Cliff Mapping
- Adaptive probing of top 50 cliffiest points with gradient/perpendicular profiles and multi-scale classification
- Deep resolution of Type 3 chaos zones with fractal dimension estimation
- Scripts: `cliff_taxonomy.py`, `cliff_taxonomy_deep.py`

### Phase 13: Validation Experiments
- Three experiments testing the novelty champion's legitimacy:
- **Timestep halving: FAIL.** DT=1/480 → DX drops from 60.2m to 24.8m (-59%). DT=1/960 → 3.3m (-95%). The 60m gait is a simulation artifact — its performance is entirely dependent on the DT=1/240 timestep.
- **Signal path tracing for w03=-1.31:** The out-of-range weight is critical. Clamping w03 to -1.0 drops DX from 60.2m to 14.6m (-76%). Pre-tanh activations reach [-2.18, +2.18] with 20% of steps in deep saturation. The extra 0.10 of tanh drive from -1.31 vs -1.0 translates to 0.16 rad of additional joint angle.
- **Random walk vs Novelty Seeker:** Random walk at r=0.2 reaches 46.1m (77% of novelty). But pure random sampling from [-1,1] found 55.7m (93% of novelty). The novelty mechanism's main contribution may be allowing weights to drift outside [-1,1] (where the DT-dependent artifact lives), not the novelty selection itself.
- **Follow-up: timestep-halved the entire top 4.** CPG Champion (50.1→11.4m, −77%), Curie (23.7→4.3m, −82%), Pelton (34.7→20.8m, −40%). At DT=1/960, all four collapse to ≤3m. The entire zoo's locomotion is DT-dependent — controllers have co-evolved with the 1/240 integration error. Absolute displacement numbers are not transferable to higher-fidelity simulation, but comparative findings (phase locking, contact entropy independence, bifurcation structure) remain valid within the DT=1/240 regime.
- Script: `validation_experiments.py`, artifact: `artifacts/validation_experiments.json`

---

## Open Ideas (not yet done)

### Validation experiments (from Phase 3 follow-up)
~~1. **Timestep halving test** -- done (Phase 13, FAIL)~~
~~2. **Signal path tracing for w03=-1.31** -- done (Phase 13)~~
~~3. **Random walk at r=0.2 vs Novelty Seeker** -- done (Phase 13)~~
4. **Replication with 5 different seeds** -- how much variance in the competition? (~35min)
~~4b. **Timestep-halve the CPG champion** -- done (Phase 13 follow-up, FAIL: all 4 top gaits are DT-dependent)~~

### Brainstormed experiments (from Phase 3 brainstorm)
~~5. **Resonance Mapping** -- done (Phase 9)~~
~~6. **Mid-Simulation Weight Switching** -- done (Phase 10)~~
~~7. **Behavioral Embryology** -- done (Phase 11)~~
8. **Surrogate Model** -- train a model to predict DX from weights. Replace 0.09s sims with microsecond inference.
9. **Transfer Entropy** -- compute information flow through each connection using transfer entropy on telemetry.

### Search extensions (from Phase 6 menu)
10. **Extend weight range** -- run Novelty Seeker with [-2,2] weights (w03=-1.31 suggests the sweet spot may be outside [-1,1])
11. **Hybrid walker** -- Novelty search (200 evals) then hill climb (800 evals)
12. **Body co-optimization** -- add 3 body params (link lengths/masses), search 9D

### Post-surgery ideas (from Phase 7 follow-up)
~~13. **Fine-grained cliff mapping** -- done (Phase 12)~~
14. **Cross-architecture behavioral interpolation** -- match metrics rather than weights to compare CPG Champion
