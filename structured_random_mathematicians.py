#!/usr/bin/env python3
"""
structured_random_mathematicians.py

Structured random search -- Condition: Mathematicians & Thinkers
================================================================

HYPOTHESIS
----------
Mathematician names carry dense associative structure in LLM embeddings:
"Euler" evokes prolific systematic computation, "Ramanujan" evokes
intuitive pattern-recognition, "Grothendieck" evokes radical abstraction.
This experiment tests whether those associations produce meaningful
differentiation in the 6-synapse bottleneck.

SEED DESIGN
-----------
104 mathematicians from mathematicians_curated.json, with metadata
(primary_field, style, known_for, era) injected into prompts to give
the LLM richer differentiation cues.

Seeds use format "Name [primary_field]" (e.g., "Euclid [geometry]").

Usage:
    python3 structured_random_mathematicians.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import structured_random_common as src
src.NUM_TRIALS = 110  # Override to allow all 104 mathematicians

from structured_random_common import run_structured_search, WEIGHT_NAMES

OUT_JSON = PROJECT / "artifacts" / "structured_random_mathematicians.json"
CURATED_JSON = PROJECT / "artifacts" / "mathematicians_curated.json"

# ── Load mathematician metadata ─────────────────────────────────────────────

with open(CURATED_JSON) as f:
    _curated = json.load(f)

MATHEMATICIANS = _curated["mathematicians"]
MATH_LOOKUP = {m["name"]: m for m in MATHEMATICIANS}

# Build seeds as "Name [primary_field]"
SEEDS = [f"{m['name']} [{m['primary_field']}]" for m in MATHEMATICIANS]

PERTURB_RADIUS = 0.05  # per-weight perturbation magnitude (±0.05)


def perturb_weights(weights, seed):
    """Apply a small deterministic perturbation to break weight-vector collapse.

    Same approach as celebrities: LLM picks the archetype, perturbation
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
    """Build the LLM prompt for a mathematician seed."""
    name = seed.split(" [")[0]
    meta = MATH_LOOKUP.get(name, {})
    return (
        f"Generate 6 synapse weights for a 3-link walking robot that embodies "
        f"the mathematician/thinker {name}. "
        f"The 6 weights are: w03, w04, w13, w14, w23, w24. "
        f"Each is a float in [-1, 1] with exactly 3 decimal places. "
        f"Think about what makes {name} DISTINCT: their intellectual style "
        f"({meta.get('style', '')}), energy, famous contributions "
        f"({meta.get('known_for', '')[:80]}), and temperament. "
        f"Systematic thinkers and intuitive leapers should differ. "
        f"Algebraists and geometers should differ. "
        f"Return ONLY a JSON object: "
        f'{{"w03": <float>, "w04": <float>, "w13": <float>, '
        f'"w14": <float>, "w23": <float>, "w24": <float>}}'
    )


def main():
    total = len(SEEDS)
    # Group by field for summary
    fields = {}
    for m in MATHEMATICIANS:
        f = m["primary_field"]
        fields.setdefault(f, []).append(m["name"])

    print(f"\nMathematician / Thinker experiment: {total} seeds across {len(fields)} fields")
    for fname, names in sorted(fields.items(), key=lambda x: -len(x[1])):
        print(f"  {fname:20s}: {len(names):3d}")

    run_structured_search("mathematicians", SEEDS, make_prompt, OUT_JSON,
                          temperature=1.5, weight_transform=perturb_weights)


if __name__ == "__main__":
    main()
