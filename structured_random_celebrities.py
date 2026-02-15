#!/usr/bin/env python3
"""
structured_random_celebrities.py

Structured random search -- Condition: Celebrity / Public Figure Names
======================================================================

HYPOTHESIS
----------
Names that appear as tokens in LLM vocabularies carry dense associative
networks that shape model behavior (Cramer et al., "Revenge of the Androids:
LLMs, The Arturo Ui Effect, Tokenization, and Narrative Collapse", 2025).
These "person-tokens" -- names like Trump, Musk, Kardashian, Beyonce, Einstein
-- function as attractor nodes in the model's embedding space.

This experiment tests whether that token-level structure produces meaningful
differentiation when projected through the 6-synapse bottleneck of a walking
robot. Unlike the earlier political figures experiment (79 names from the
Trump orbit), this uses the full range of celebrity names identified in
tokenization lexicons: politicians, reality TV, tech billionaires, musicians,
actors, athletes, authors, and historical figures.

KEY QUESTIONS
  1. How many unique weight vectors emerge from ~130 celebrity names?
  2. Do domain boundaries (politics vs entertainment vs sports) predict
     weight clustering, or does the LLM's coarse categorization cut
     across domain lines?
  3. Do names with stronger token-level presence (single-token names,
     high training-data frequency) produce more distinctive gaits?
  4. Which celebrity names produce the highest-displacement gaits?
  5. How does the Kardashian cluster compare to the Trump cluster?
  6. Do historical figures (Einstein, Shakespeare) map differently
     from contemporary celebrities?

SEED DESIGN
-----------
~130 names across 12 groups drawn from tokenization lexicon analysis:
  - Trump Family (8)
  - Trump Admin (14)
  - US Politics (16)
  - International Politics (13)
  - Controversial/Whistleblower (8)
  - Kardashian/Reality TV (12)
  - Tech Titans (9)
  - Musicians (15)
  - Actors/Entertainment (12)
  - Sports (8)
  - Cultural/Authors (7)
  - Historical (10)

Names use full-name format for unambiguity (e.g., "Kim Kardashian" not "Kim").

Usage:
    python3 structured_random_celebrities.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import structured_random_common as src
src.NUM_TRIALS = 200  # Override default 100 to allow all ~130 seeds

from structured_random_common import run_structured_search, WEIGHT_NAMES

OUT_JSON = PROJECT / "artifacts" / "structured_random_celebrities.json"

# ── Seed lists by domain ─────────────────────────────────────────────────────
# Names drawn from tokenization lexicon analysis in Cramer et al. (2025),
# "Revenge of the Androids." These are names found as tokens or near-tokens
# in LLM vocabularies (OpenAI GPTs, LLaMA, Stable Diffusion), plus names
# from Wikipedia association extractions around key token-nodes (Trump,
# Manafort, Kardashian).

TRUMP_FAMILY = [
    "Donald Trump",
    "Ivanka Trump",
    "Melania Trump",
    "Jared Kushner",
    "Donald Trump Jr",
    "Eric Trump",
    "Tiffany Trump",
    "Barron Trump",
]

TRUMP_ADMIN = [
    "Steve Bannon",
    "Michael Flynn",
    "Mike Pence",
    "Jeff Sessions",
    "Rudy Giuliani",
    "Michael Cohen",
    "Kellyanne Conway",
    "Hope Hicks",
    "Sean Spicer",
    "William Barr",
    "Mike Pompeo",
    "Rex Tillerson",
    "Kevin McCarthy",
    "Mitch McConnell",
]

US_POLITICS = [
    "Joe Biden",
    "Barack Obama",
    "Hillary Clinton",
    "Nancy Pelosi",
    "Chuck Schumer",
    "Bernie Sanders",
    "Elizabeth Warren",
    "Mitt Romney",
    "Liz Cheney",
    "Adam Schiff",
    "Robert Mueller",
    "James Comey",
    "John McCain",
    "Ted Cruz",
    "Marco Rubio",
    "AOC",
]

INTERNATIONAL = [
    "Vladimir Putin",
    "Kim Jong Un",
    "Angela Merkel",
    "Emmanuel Macron",
    "Volodymyr Zelensky",
    "Boris Johnson",
    "Nigel Farage",
    "Justin Trudeau",
    "Narendra Modi",
    "Xi Jinping",
    "Benjamin Netanyahu",
    "Viktor Yanukovych",
    "Ferdinand Marcos",
]

CONTROVERSIAL = [
    "Julian Assange",
    "Edward Snowden",
    "Jeffrey Epstein",
    "George Soros",
    "Ghislaine Maxwell",
    "Paul Manafort",
    "Roger Stone",
    "Sean Hannity",
]

KARDASHIAN = [
    "Kim Kardashian",
    "Kylie Jenner",
    "Kendall Jenner",
    "Khloe Kardashian",
    "Kourtney Kardashian",
    "Kris Jenner",
    "Rob Kardashian",
    "Caitlyn Jenner",
    "Travis Scott",
    "Scott Disick",
    "Blac Chyna",
    "OJ Simpson",
]

TECH = [
    "Elon Musk",
    "Mark Zuckerberg",
    "Jeff Bezos",
    "Bill Gates",
    "Steve Jobs",
    "Peter Thiel",
    "Sam Altman",
    "Tim Cook",
    "Jack Dorsey",
]

MUSICIANS = [
    "Rihanna",
    "Beyonce",
    "Kanye West",
    "Taylor Swift",
    "Drake",
    "Madonna",
    "Lady Gaga",
    "Eminem",
    "Jay-Z",
    "Adele",
    "Ed Sheeran",
    "Justin Bieber",
    "Ariana Grande",
    "Billie Eilish",
    "BTS",
]

ENTERTAINMENT = [
    "Oprah Winfrey",
    "Leonardo DiCaprio",
    "Johnny Depp",
    "Angelina Jolie",
    "Tom Hanks",
    "Dwayne Johnson",
    "Arnold Schwarzenegger",
    "Will Smith",
    "Tom Cruise",
    "Keanu Reeves",
    "Brad Pitt",
    "Meryl Streep",
]

SPORTS = [
    "LeBron James",
    "Cristiano Ronaldo",
    "Lionel Messi",
    "Serena Williams",
    "Tiger Woods",
    "Tom Brady",
    "Michael Jordan",
    "Usain Bolt",
]

CULTURAL = [
    "Neil Gaiman",
    "Amanda Palmer",
    "Stephen King",
    "JK Rowling",
    "Noam Chomsky",
    "Jordan Peterson",
    "Joe Rogan",
]

HISTORICAL = [
    "Albert Einstein",
    "William Shakespeare",
    "Napoleon Bonaparte",
    "Mahatma Gandhi",
    "Martin Luther King",
    "Nelson Mandela",
    "Winston Churchill",
    "Abraham Lincoln",
    "Cleopatra",
    "Charles Darwin",
]

# ── Build seeds with domain tags ─────────────────────────────────────────────
GROUPS = {
    "trump_family": TRUMP_FAMILY,
    "trump_admin": TRUMP_ADMIN,
    "us_politics": US_POLITICS,
    "international": INTERNATIONAL,
    "controversial": CONTROVERSIAL,
    "kardashian": KARDASHIAN,
    "tech": TECH,
    "musician": MUSICIANS,
    "entertainment": ENTERTAINMENT,
    "sports": SPORTS,
    "cultural": CULTURAL,
    "historical": HISTORICAL,
}

SEEDS = []
for group_name, names in GROUPS.items():
    for name in names:
        SEEDS.append(f"{name} [{group_name}]")


PERTURB_RADIUS = 0.05  # per-weight perturbation magnitude (±0.05)


def perturb_weights(weights, seed):
    """Apply a small deterministic perturbation to break weight-vector collapse.

    Same approach as archetypometrics: LLM picks the archetype, perturbation
    individuates within it. Deterministic via SHA-256 hash of seed string.
    """
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(h)
    perturbed = {}
    for wn in WEIGHT_NAMES:
        delta = rng.uniform(-PERTURB_RADIUS, PERTURB_RADIUS)
        perturbed[wn] = max(-1.0, min(1.0, weights[wn] + delta))
    return perturbed


def make_prompt(seed):
    """Build the LLM prompt for a celebrity/public figure seed."""
    name = seed.split(" [")[0]
    domain = seed.split(" [")[1].rstrip("]") if " [" in seed else ""
    return (
        f"Generate 6 synapse weights for a 3-link walking robot that embodies "
        f"the public figure {name}. "
        f"The 6 weights are: w03, w04, w13, w14, w23, w24. "
        f"Each is a float in [-1, 1] with exactly 3 decimal places. "
        f"Think about what makes {name} DISTINCT: their energy, aggression, "
        f"grace, authority, and movement style. Politicians and athletes should "
        f"differ. Entertainers and intellectuals should differ. Brash and calm should differ. "
        f"Return ONLY a JSON object: "
        f'{{"w03": <float>, "w04": <float>, "w13": <float>, '
        f'"w14": <float>, "w23": <float>, "w24": <float>}}'
    )


def main():
    total = len(SEEDS)
    print(f"\nCelebrity / Public Figure experiment: {total} seeds across {len(GROUPS)} domains")
    for gname, names in GROUPS.items():
        print(f"  {gname:20s}: {len(names):3d}")
    run_structured_search("celebrities", SEEDS, make_prompt, OUT_JSON,
                          temperature=1.5, weight_transform=perturb_weights)


if __name__ == "__main__":
    main()
