# Archetypometrics Experiment: Session Notes

## What This Is

2000 fictional character names from UVM's Archetypometrics project (Dodds et al. 2025) run through the LLM → synapse weight → PyBullet simulation pipeline.

**Data source**: https://doi.org/10.5281/zenodo.16953724
- "Archetypometrics: The Essence of Character" — Peter Sheridan Dodds, Julia Witte Zimmerman, Calla G. Beauregard, Ashley M. A. Fehr, Mikaela Fudolig, Timothy R. Tangherlini, Christopher M. Danforth
- University of Vermont / Computational Story Lab
- 2000 characters across 341 stories, scored on 464 semantic-differential traits
- Character list extracted from `data/plain/current/2000/data_characters.tsv` in the Zenodo archive

The character TSV is saved locally at:
`artifacts/archetypometrics_characters.tsv`

## Scripts

| Script | Purpose |
|---|---|
| `structured_random_archetypometrics.py` | Runs 2000 characters through Ollama → PyBullet pipeline |
| `analyze_archetypometrics.py` | Full analysis: clustering, genre breakdown, cross-condition comparison, 8 figures |

## How to Re-Run

```bash
# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate er

# Ensure Ollama is running with qwen3-coder:30b
# ollama run qwen3-coder:30b

# Run the experiment (~31 minutes, 2000 trials at ~1s each)
python3 structured_random_archetypometrics.py

# Run analysis (generates 8 figures + JSON)
python3 analyze_archetypometrics.py
```

## Results Summary (First Run — February 2025)

### Top-Level Numbers
- **2000 trials**, 0 failures, 9 dead (0.45%)
- **15 unique weight vectors** (faithfulness = 0.007)
- **98.4% map to the same 4 archetypes found in the celebrity experiment**
- Median |DX| = 1.55m, Max |DX| = 29.17m
- Runtime: 1887 seconds (~31.5 minutes)

### The 4 Core Archetypes (from celebrity experiment, confirmed here)

| Archetype | Weights (w03,w04,w13,w14,w23,w24) | DX | Count | % |
|---|---|---|---|---|
| Assertive | (+0.8, -0.6, +0.2, -0.9, +0.5, -0.4) | +1.55m | 780 | 39.0% |
| Default | (+0.6, -0.4, +0.2, -0.8, +0.5, -0.3) | +1.18m | 709 | 35.4% |
| Contrarian | (+0.8, -0.6, -0.2, +0.9, +0.5, -0.4) | -1.19m | 248 | 12.4% |
| Transgressor | (+0.6, -0.4, -0.2, +0.8, +0.3, -0.5) | -5.64m | 231 | 11.6% |

### 11 New Weight Vectors (fiction-only, not found in celebrities)

| DX | Count | Weights | Characters |
|---|---|---|---|
| **+29.17m** | 3 | (-0.8, +0.6, +0.2, -0.9, +0.5, -0.4) | Lila Crane (Psycho), David Mills (Se7en), Rust Cohle (True Detective) |
| **-25.12m** | 3 | (-0.6, +0.8, +0.9, -0.4, -0.2, +0.5) | Moaning Myrtle (HP), Violet Baudelaire (Unfortunate Events), Stella Gibson (The Fall) |
| **+20.97m** | 3 | (-0.8, +0.6, +0.9, -0.4, -0.3, +0.7) | Nurse Ratched, Tracy Mills (Se7en), Glen Lantz (Nightmare on Elm St) |
| **+7.79m** | 10 | (-0.8, +0.6, +0.9, -0.4, -0.5, +0.7) | Dr. House, Severus Snape, Mr. Robot, Dominique DiPierro, William Somerset, Donald Thompson, Marge Thompson, Scar (FMA), O'Brien (1984), Macbeth |
| **+7.21m** | 2 | (+0.6, -0.8, +0.2, -0.9, +0.5, -0.4) | Ellis Carver (The Wire), Norman Wilson (The Wire) |
| **+0.63m** | 5 | (-0.8, +0.6, +0.2, -0.9, -0.4, +0.7) | James Doakes (Dexter), Elliot Alderson (Mr. Robot), Winston Smith (1984), Paul Spector (The Fall), Jim Burns (The Fall) |
| **+0.73m** | 2 | (+0.6, -0.8, -0.2, +0.9, +0.5, -0.4) | Marla Singer (Fight Club), Gollum (LOTR) |
| **-5.47m** | 1 | (-0.8, +0.6, -0.2, +0.9, -0.5, +0.3) | Tony Johnson (After Life) |
| **-0.61m** | 1 | unique | Lisa Johnson (After Life) |
| **+0.30m** | 1 | unique | Faustus Blackwood (Sabrina) |

### Notable Findings

1. **The +29.17m vector is the Revelation gait** — the project's all-time displacement champion, originally found by the Bible verse "Revelation". Fiction found it independently through Lila Crane, David Mills, and Rust Cohle. All three are characters defined by encountering extreme violence/darkness.

2. **The +7.79m "dark brilliance" cluster**: Dr. House, Severus Snape, Macbeth, O'Brien (the torturer from 1984), Scar from FMA. These are morally complex, intellectually formidable characters who operate through manipulation or dark authority.

3. **The +0.63m "surveillance state" cluster**: Winston Smith, Elliot Alderson, James Doakes. Characters defined by watching, being watched, or existing under systems of control.

4. **Downton Abbey is the most collapsed story**: all 15 characters → 1 unique weight vector. Other highly collapsed: This Is Us, Money Heist, Modern Family, Mamma Mia (all 1 unique from 10 chars).

5. **Breaking Bad, Hannibal, The Witcher, Futurama** are the least collapsed (4 unique from 10 chars each, faithfulness = 0.400).

6. **Crime drama** has the highest genre faithfulness (0.062) — morally complex characters break the archetype mold more often.

7. **Twin Peaks** median |DX| = 5.64 — the highest of any story with 10+ characters (most characters classified as Transgressor).

### Cross-Condition Comparison

| Condition | N | Dead% | Med|DX| | Max|DX| | MeanPL | Faithful | U-test z |
|---|---|---|---|---|---|---|---|
| archetypometrics | 2000 | 0.4% | 1.55 | 29.17 | 0.912 | 0.007 | (self) |
| celebrities | 132 | 0.0% | 1.18 | 5.64 | 0.908 | 0.030 | +3.501 *** |
| politics | 79 | 0.0% | 1.18 | 5.64 | 0.905 | 0.051 | +4.111 *** |
| verbs | 100 | 5.0% | 1.55 | 25.12 | 0.850 | 0.180 | -1.449 |
| theorems | 95 | 8.4% | 2.79 | 9.55 | 0.904 | 0.158 | -7.317 *** |
| bible | 100 | 0.0% | 1.55 | 29.17 | 0.908 | 0.090 | +0.001 |
| places | 100 | 0.0% | 1.18 | 5.64 | 0.884 | 0.040 | +8.339 *** |
| baseline | 100 | 8.0% | 6.64 | 27.79 | 0.613 | 1.000 | -10.655 *** |

### Genre Breakdown

| Genre | N | Dead% | Med|DX| | Max|DX| | Faithful |
|---|---|---|---|---|---|
| other | 933 | 0.6% | 1.55 | 29.17 | 0.014 |
| drama | 203 | 0.0% | 1.18 | 5.64 | 0.020 |
| comedy | 168 | 0.0% | 1.19 | 5.64 | 0.024 |
| scifi | 163 | 0.0% | 1.19 | 5.64 | 0.025 |
| fantasy | 152 | 0.7% | 1.55 | 25.12 | 0.046 |
| crime_drama | 113 | 1.8% | 1.55 | 7.21 | 0.062 |
| horror | 89 | 0.0% | 1.19 | 29.17 | 0.056 |
| anime | 55 | 0.0% | 1.55 | 7.79 | 0.091 |
| disney_pixar | 52 | 0.0% | 1.55 | 5.64 | 0.077 |
| superhero | 45 | 0.0% | 1.55 | 1.55 | 0.044 |
| classic_lit | 27 | 0.0% | 1.18 | 5.64 | 0.148 |

## Output Files

| File | Description |
|---|---|
| `artifacts/structured_random_archetypometrics.json` | Raw trial data (2000 entries with weights, dx, speed, etc.) |
| `artifacts/structured_random_archetypometrics_analysis.json` | Full analysis results |
| `artifacts/archetypometrics_characters.tsv` | Source character list from Zenodo |
| `artifacts/plots/arc_fig01_dx_by_genre.png` | DX distribution by genre (box plot) |
| `artifacts/plots/arc_fig02_weight_pca.png` | Weight space PCA colored by genre |
| `artifacts/plots/arc_fig03_cluster_distribution.png` | Cluster size histogram + cumulative coverage |
| `artifacts/plots/arc_fig04_faithfulness_by_story.png` | Faithfulness by story (horizontal bar) |
| `artifacts/plots/arc_fig05_archetype_overlap.png` | Pie chart: fiction vs 4 celebrity archetypes |
| `artifacts/plots/arc_fig06_cross_condition_pca.png` | Behavioral PCA: fiction vs all conditions |
| `artifacts/plots/arc_fig07_dimensionality.png` | Effective dimensionality comparison |
| `artifacts/plots/arc_fig08_story_diversity.png` | Story size vs weight diversity |

## Context: The Broader Experiment

This is one condition in a structured random search experiment testing whether LLMs serve as structured samplers of neural network weight space. The pipeline:

1. Give Ollama (qwen3-coder:30b, temperature=0.8) a seed (character name + story)
2. LLM returns 6 synapse weights for a 3-link walking robot
3. Run headless PyBullet simulation (4000 steps @ 240 Hz)
4. Compute Beer-framework analytics (displacement, speed, phase lock, entropy, etc.)

**Other conditions already run:**
- `structured_random_verbs.py` — 100 English/German/Spanish/Portuguese verbs
- `structured_random_theorems.py` — 95 mathematical theorems
- `structured_random_bible.py` — 100 Bible verses
- `structured_random_places.py` — 100 place names
- `structured_random_baseline.py` — 100 random seeds (control)
- `structured_random_celebrities.py` — 132 celebrity/public figure names
- `structured_random_politics.py` — 79 political figures
- `structured_random_archetypometrics.py` — 2000 fictional characters (this one)

Total across all conditions: ~2700 LLM-mediated trials plus ~25,000 supporting simulations.

## Key Insight for Re-Run

The concern about "Trump hairball contamination" is worth noting: the celebrity experiment (132 names) was designed around tokenization lexicons from the "Revenge of the Androids" paper, which centered on Trump-orbit names. This may have biased the prompt phrasing or name selection. The archetypometrics experiment is cleaner — 2000 characters chosen by an independent research team (Dodds et al.) with no political agenda.

For a clean re-run:
1. Start a fresh Claude Code window
2. The scripts are self-contained — just run `structured_random_archetypometrics.py`
3. The character list comes from the Zenodo dataset, not from any prior session context
4. Ollama state is independent between runs (temperature=0.8 introduces stochasticity)
5. Consider running celebrities separately from the Trump-orbit framing — the analysis script (`analyze_archetypometrics.py`) already does cross-condition comparison without assuming the celebrity groupings

## Prompt Used (from structured_random_archetypometrics.py)

```
Generate 6 synapse weights for a 3-link walking robot inspired by
the fictional character: {name} from {story}. The weights are w03, w04, w13, w14,
w23, w24, each in [-1, 1]. Translate the character's personality,
energy, movement style, and archetypal role into weight
magnitudes, signs, and symmetry patterns.
Return ONLY a JSON object like
{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}
with no other text.
```
