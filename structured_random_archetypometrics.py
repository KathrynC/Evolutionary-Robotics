#!/usr/bin/env python3
"""
structured_random_archetypometrics.py

Structured random search -- Condition: Fictional Character Names (Archetypometrics)
===================================================================================

HYPOTHESIS
----------
The celebrity experiment (132 real-person names → 4 gaits) showed the LLM
coarse-grains person-tokens into a small number of archetypal weight vectors.
But celebrities are real people whose names appear heavily in training data.
Do fictional characters behave the same way?

UVM's Archetypometrics project (Dodds et al. 2025) provides 2000 fictional
characters across 341 stories — from Harry Potter to The Wire, from Lord of
the Rings to Naruto, from Game of Thrones to My Little Pony. These names
span the full spectrum of cultural salience: some (Gandalf, Darth Vader)
are deeply embedded tokens; others (Jeremy Chetri, Ziggy Sobotka) are
obscure. This lets us test:

KEY QUESTIONS
  1. How many unique weight vectors emerge from ~2000 fictional characters?
  2. Do story/franchise boundaries predict weight clustering, or does the
     LLM's coarse categorization cut across narrative universes?
  3. Do highly recognizable characters (Harry Potter, Darth Vader) map
     differently from obscure ones (minor Wire characters)?
  4. How does the collapse compare to celebrities (4 vectors from 132 names)?
  5. Do the same 4 weight-vector archetypes appear, or does fiction unlock
     new regions of weight space?

DATA SOURCE
-----------
UVM Computational Story Lab, "Archetypometrics: The Essence of Character"
Dodds, Zimmerman, Beauregard, Fehr, Fudolig, Tangherlini, & Danforth (2025)
https://doi.org/10.5281/zenodo.16953724

2000 characters across 341 stories, scored on 464 semantic-differential traits.

Usage:
    python3 structured_random_archetypometrics.py
"""

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import structured_random_common as src
src.NUM_TRIALS = 2100  # Override default 100 to allow all ~2000 seeds

from structured_random_common import run_structured_search, WEIGHT_NAMES

OUT_JSON = PROJECT / "artifacts" / "structured_random_archetypometrics.json"
CHAR_TSV = PROJECT / "artifacts" / "archetypometrics_characters.tsv"

# ── Load character names from archetypometrics TSV ──────────────────────────

def load_characters():
    """Load characters from the archetypometrics TSV file.

    Returns list of (character_name, story_name) tuples.
    Handles duplicate character names by keeping both (disambiguated by story).
    """
    chars = []
    with open(CHAR_TSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["character"].strip()
            # character/story field is "Character/Story"
            char_story = row["character/story"].strip()
            story = char_story.split("/", 1)[1] if "/" in char_story else "Unknown"
            chars.append((name, story))
    return chars


def build_seeds(chars):
    """Build seeds with story tags: 'Character Name [Story]'"""
    seeds = []
    for name, story in chars:
        seeds.append(f"{name} [{story}]")
    return seeds


PERTURB_RADIUS = 0.05  # per-weight perturbation magnitude (±0.05)


def perturb_weights(weights, seed):
    """Apply a small deterministic perturbation to break weight-vector collapse.

    The LLM chooses the archetype (sign structure, general region of weight space).
    This function adds a seed-specific nudge so that characters who receive the same
    LLM output still get distinct weight vectors — and, on the cliff-riddled landscape
    (median cliffiness 2.88m at r=0.05), distinct gaits.

    The perturbation is deterministic: same seed always produces the same nudge.
    Values are clamped to [-1, 1] after perturbation.
    """
    # Derive a deterministic RNG seed from the character string
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(h)
    perturbed = {}
    for wn in WEIGHT_NAMES:
        delta = rng.uniform(-PERTURB_RADIUS, PERTURB_RADIUS)
        perturbed[wn] = max(-1.0, min(1.0, weights[wn] + delta))
    return perturbed


def make_prompt(seed):
    """Build the LLM prompt for a fictional character seed."""
    # Parse "Character Name [Story]"
    bracket = seed.rfind(" [")
    if bracket > 0:
        name = seed[:bracket]
        story = seed[bracket + 2:-1]
    else:
        name = seed
        story = ""

    story_clause = f" from {story}" if story else ""

    return (
        f"Generate 6 synapse weights for a 3-link walking robot that embodies "
        f"the fictional character {name}{story_clause}. "
        f"The 6 weights are: w03, w04, w13, w14, w23, w24. "
        f"Each is a float in [-1, 1] with exactly 3 decimal places. "
        f"Think about what makes {name} DISTINCT: their energy, aggression, "
        f"grace, moral alignment, and movement style. Villains and heroes should "
        f"differ. Tricksters and warriors should differ. Calm and explosive should differ. "
        f"Return ONLY a JSON object: "
        f'{{\"w03\": <float>, \"w04\": <float>, \"w13\": <float>, '
        f'\"w14\": <float>, \"w23\": <float>, \"w24\": <float>}}'
    )


def main():
    chars = load_characters()
    seeds = build_seeds(chars)

    # Count stories
    stories = {}
    for name, story in chars:
        stories.setdefault(story, []).append(name)

    total = len(seeds)
    print(f"\nArchetypometrics experiment: {total} characters across {len(stories)} stories")

    # Show top 20 stories by character count
    top_stories = sorted(stories.items(), key=lambda x: -len(x[1]))[:20]
    for story, names in top_stories:
        print(f"  {story:40s}: {len(names):3d}")
    if len(stories) > 20:
        print(f"  ... and {len(stories) - 20} more stories")

    run_structured_search("archetypometrics", seeds, make_prompt, OUT_JSON,
                          temperature=1.5, weight_transform=perturb_weights)


if __name__ == "__main__":
    main()
