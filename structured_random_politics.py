#!/usr/bin/env python3
"""
structured_random_politics.py

Structured random search — Condition: Political Figures
========================================================

HYPOTHESIS
----------
Political figure names are among the most semantically loaded tokens in any
LLM's vocabulary. Unlike verbs (which encode action) or theorems (which encode
structure), political names encode dense associative networks: personality,
ideology, relationships, media narratives, emotional valence. The question is
whether the LLM's token-level representation of these names produces meaningful
structure when projected into the 6D weight space of a walking robot.

This experiment directly parallels the AI Seances project (Cramer 2022), which
found that GPT-3 could maintain consistent "personas" for public figures —
suggesting that LLM representations of political names carry structured
behavioral signatures, not just surface-level associations.

SEED DESIGN
-----------
79 political figure names organized into 4 groups:
  - Family (7):      Trump family members
  - Admin (22):      Administration officials and allies
  - Adjacent (19):   Adjacent figures (media, tech, foreign)
  - Opposition (31): Opposition figures, media, investigators

The grouping is metadata for analysis — the LLM sees only the name.
Seeds are single-token or few-token names (last names, first names, or
handles) as they would appear in training data. This maximizes the chance
that the LLM's representation is richly structured.

PROMPT STRATEGY
--------------
The prompt asks the LLM to translate "the public persona, energy, and
characteristic style" of the political figure into weight patterns. This
gives the LLM latitude to use whatever associations it has — personality,
movement style, public image, emotional tone — without constraining which
aspect maps to which weight.

KEY QUESTIONS
  1. Do family members cluster in weight space? (Functor faithfulness test)
  2. Do admin vs opposition form distinct behavioral phenotypes?
  3. Which names produce the highest-displacement gaits?
  4. How does weight diversity compare to other structured conditions?
  5. Are there collapse clusters (multiple names → identical weights)?

Usage:
    python3 structured_random_politics.py
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import run_structured_search

OUT_JSON = PROJECT / "artifacts" / "structured_random_politics.json"

# ── Seed list: Political figures organized by group ──────────────────────────
# Group membership is encoded in the seed string as a suffix tag so the
# analysis script can recover it without a separate lookup table.

FAMILY = [
    "realDonaldTrump",
    "Ivanka",
    "Jared",
    "Kushner",
    "Melania",
    "Tiffany",
    "Lara",
]

ADMIN = [
    "Bannon",
    "Barr",
    "Barrett",
    "Bolton",
    "Conway",
    "Flynn",
    "Hicks",
    "Haley",
    "Huckabee",
    "Kavanaugh",
    "Kennedy",
    "McCarthy",
    "McConnell",
    "Meadows",
    "Pence",
    "Pompeo",
    "Pruitt",
    "Sanders",
    "Scalia",
    "Sessions",
    "Spicer",
    "Tillerson",
]

ADJACENT = [
    "Assange",
    "Burnett",
    "Byrne",
    "Cobb",
    "Cruz",
    "Epstein",
    "Elon",
    "Farage",
    "Gates",
    "Giuliani",
    "Manafort",
    "Marco",
    "Maxwell",
    "Mercer",
    "Musk",
    "Putin",
    "Rubio",
    "Shapiro",
    "Snowden",
]

OPPOSITION = [
    "Woodward",
    "Bernstein",
    "Baldwin",
    "Baker",
    "Bernie",
    "Biden",
    "Booker",
    "Brandon",
    "Carroll",
    "Cassidy",
    "Comey",
    "Castro",
    "Cheney",
    "Clinton",
    "Clintons",
    "Daniels",
    "Feinstein",
    "Garland",
    "Hillary",
    "Hutchinson",
    "McCain",
    "Mueller",
    "Pelosi",
    "Podesta",
    "Romney",
    "Rosenstein",
    "Schiff",
    "Schumer",
    "Steele",
    "Warren",
    "Yates",
]

# Build seeds with group tags for later analysis
SEEDS = []
for name in FAMILY:
    SEEDS.append(f"{name} [family]")
for name in ADMIN:
    SEEDS.append(f"{name} [admin]")
for name in ADJACENT:
    SEEDS.append(f"{name} [adjacent]")
for name in OPPOSITION:
    SEEDS.append(f"{name} [opposition]")


def make_prompt(seed):
    """Build the LLM prompt for a political figure seed.

    The prompt asks the LLM to translate the figure's public persona into
    6 synapse weights for a walking robot. The group tag (e.g. [family])
    is included in the seed string but the LLM sees it as part of the
    context — it may or may not influence the weights.
    """
    # Extract just the name for the prompt (keep group tag for context)
    name = seed.split(" [")[0]
    return (
        f"Generate 6 synapse weights for a 3-link walking robot inspired by "
        f"the political figure: {name}. The weights are w03, w04, w13, w14, "
        f"w23, w24, each in [-1, 1]. Translate the public persona, energy, "
        f"and characteristic style of this figure into weight magnitudes, "
        f"signs, and symmetry patterns. "
        f'Return ONLY a JSON object like '
        f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
        f"with no other text."
    )


def main():
    print(f"\nPolitical figures experiment: {len(SEEDS)} seeds")
    print(f"  Family:     {len(FAMILY)}")
    print(f"  Admin:      {len(ADMIN)}")
    print(f"  Adjacent:   {len(ADJACENT)}")
    print(f"  Opposition: {len(OPPOSITION)}")
    run_structured_search("politics", SEEDS, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
