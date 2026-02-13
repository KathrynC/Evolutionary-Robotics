#!/usr/bin/env python3
"""
structured_random_theorems.py

Structured random search — Condition #2: Mathematical Theorems
===============================================================

HYPOTHESIS
----------
Mathematical theorems encode structural principles — symmetry, fixed points,
convergence, periodicity, invariance — that are precisely the properties that
determine dynamical system behavior. A theorem like "Brouwer Fixed Point" is
*about* fixed points; "Poincaré Recurrence" is *about* periodicity. The LLM
should map these mathematical structures into weight patterns that produce
corresponding dynamical behaviors in the robot.

This condition is the strongest test of "structural transfer" — whether the
LLM can read out a mathematical principle and encode it as a weight geometry
that the body then develops into behavior enacting that principle.

SEED DESIGN
-----------
108 theorems spanning 12 branches of mathematics:
  Analysis (16):       IVT, MVT, FTC, Bolzano-Weierstrass, Banach/Brouwer FPTs, ...
  Algebra (14):        FTA, Cayley-Hamilton, Sylow, CRT, Spectral Theorem, ...
  Geometry/Topology (14): Pythagoras, Euler formula, Gauss-Bonnet, Hairy Ball, ...
  Number Theory (11):  Fermat's Little, Quadratic Reciprocity, PNT, FLT, ...
  Combinatorics (12):  Ramsey, Hall's Marriage, Four Color, Sperner, ...
  Probability (7):     CLT, LLN, Bayes, Chebyshev, ...
  Dynamical Systems (9): Poincaré Recurrence, KAM, Lyapunov Stability, ...
  Logic (4):           Gödel, Cantor, Zorn, Tychonoff
  Differential Eqs (3): Picard-Lindelöf, Sturm-Liouville, Noether
  Information Theory (2): Shannon source/channel coding
  Linear Algebra (3):  Perron-Frobenius, SVD, Courant-Fischer

Each theorem is referenced by its conventional name only (no statement or
proof), so the LLM must rely on its internal representation of the theorem's
structural content.

PROMPT STRATEGY
--------------
The prompt asks the LLM to translate "the structural principle of this theorem
— its symmetries, invariants, fixed points, transformations, or key
relationships — into weight patterns." This targets the mathematical structure
directly, not metaphorical associations.

KEY RESULTS (from 95-trial run, 5 seeds fell outside the 100 sample)
----------------------------------------------------------------------
  Dead: 8.4% (same as baseline)
  Median |DX|: 2.79m (vs 6.64m baseline)
  Max |DX|: 9.55m (from "Pythagorean Theorem")
  Mean phase lock: 0.904 (vs 0.613 baseline — highest of all conditions)

Notable: Theorems dominate the phase lock leaderboard (18/20 top slots).
Poincaré Recurrence Theorem is the most phase-locked gait in the entire pool
(0.999) — a theorem about systems returning to near-initial states producing
near-perfect periodicity. Noether's Theorem produces the deadest gait (DX=0.03m)
with exact pairwise anti-symmetry: (+0.5,-0.5), (+0.3,-0.3), (+0.7,-0.7).

Usage:
    python3 structured_random_theorems.py
"""

import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import run_structured_search

OUT_JSON = PROJECT / "artifacts" / "structured_random_theorems.json"

# ── Seed list: mathematical theorems ─────────────────────────────────────────
# Organized by branch of mathematics. Each theorem name is its conventional
# reference — the LLM must access its internal representation of the theorem's
# content, not just pattern-match on keywords. Over-provisioned (108 theorems)
# so that shuffle + [:100] gives diverse samples across branches.

THEOREMS = [
    # Analysis — theorems about continuity, convergence, completeness, fixed points
    "Intermediate Value Theorem",
    "Mean Value Theorem",
    "Fundamental Theorem of Calculus",
    "Taylor's Theorem",
    "L'Hopital's Rule",
    "Bolzano-Weierstrass Theorem",
    "Heine-Borel Theorem",
    "Monotone Convergence Theorem",
    "Dominated Convergence Theorem",
    "Stone-Weierstrass Theorem",
    "Arzela-Ascoli Theorem",
    "Baire Category Theorem",
    "Banach Fixed Point Theorem",
    "Brouwer Fixed Point Theorem",
    "Implicit Function Theorem",
    "Inverse Function Theorem",
    # Algebra — theorems about group structure, ring structure, matrix properties
    "Fundamental Theorem of Algebra",
    "Cayley-Hamilton Theorem",
    "Sylow Theorems",
    "Lagrange's Theorem on subgroup order",
    "Chinese Remainder Theorem",
    "Fundamental Theorem of Arithmetic",
    "Jordan Normal Form Theorem",
    "Spectral Theorem for symmetric matrices",
    "Rank-Nullity Theorem",
    "Burnside's Lemma",
    "Wedderburn's Little Theorem",
    "Artin-Wedderburn Theorem",
    "Nullstellensatz",
    "Structure Theorem for finitely generated abelian groups",
    # Geometry and Topology — theorems about shape, curvature, surfaces, embeddings
    "Pythagorean Theorem",
    "Euler's Polyhedron Formula",
    "Gauss-Bonnet Theorem",
    "Hairy Ball Theorem",
    "Borsuk-Ulam Theorem",
    "Jordan Curve Theorem",
    "Poincare-Hopf Theorem",
    "Ham Sandwich Theorem",
    "Desargues' Theorem",
    "Pappus' Theorem",
    "Euler characteristic of surfaces",
    "Classification of compact surfaces",
    "Uniformization Theorem",
    "Nash Embedding Theorem",
    # Number Theory
    "Fermat's Little Theorem",
    "Wilson's Theorem",
    "Quadratic Reciprocity",
    "Prime Number Theorem",
    "Dirichlet's Theorem on primes in arithmetic progressions",
    "Bertrand's Postulate",
    "Vinogradov's Theorem",
    "Waring's Problem solution",
    "Fermat's Last Theorem",
    "Mordell Conjecture (Faltings' Theorem)",
    "Catalan's Conjecture (Mihailescu's Theorem)",
    # Combinatorics and Graph Theory
    "Ramsey's Theorem",
    "Hall's Marriage Theorem",
    "Dilworth's Theorem",
    "Konig's Theorem",
    "Menger's Theorem",
    "Sperner's Lemma",
    "Van der Waerden's Theorem",
    "Hales-Jewett Theorem",
    "Turan's Theorem",
    "Four Color Theorem",
    "Kuratowski's Theorem",
    "Cayley's Formula for labeled trees",
    # Probability and Statistics
    "Central Limit Theorem",
    "Law of Large Numbers",
    "Bayes' Theorem",
    "Chebyshev's Inequality",
    "Markov's Inequality",
    "Berry-Esseen Theorem",
    "De Finetti's Theorem",
    # Dynamical Systems — theorems about trajectories, recurrence, stability, chaos
    "Poincare Recurrence Theorem",
    "Birkhoff Ergodic Theorem",
    "Sharkovskii's Theorem",
    "Takens' Embedding Theorem",
    "KAM Theorem",
    "Poincare-Bendixson Theorem",
    "Hartman-Grobman Theorem",
    "Center Manifold Theorem",
    "Lyapunov Stability Theorem",
    # Logic and Foundations
    "Godel's Incompleteness Theorem",
    "Cantor's Theorem on cardinality",
    "Zorn's Lemma",
    "Tychonoff's Theorem",
    # Differential Equations
    "Picard-Lindelof Existence and Uniqueness Theorem",
    "Sturm-Liouville Theory",
    "Noether's Theorem on symmetry and conservation",
    # Information Theory
    "Shannon's Source Coding Theorem",
    "Shannon's Channel Coding Theorem",
    # Linear Algebra
    "Perron-Frobenius Theorem",
    "Singular Value Decomposition existence",
    "Courant-Fischer Minimax Theorem",
]


def make_prompt(theorem):
    """Build the LLM prompt for a given theorem seed.

    The prompt specifically asks for structural principles: symmetries,
    invariants, fixed points, transformations, relationships. These are
    the mathematical concepts most likely to map onto dynamical system
    properties (which is what the weight vector determines).

    Note: the theorem name is quoted in the prompt, signaling to the LLM
    that it should treat it as a specific reference, not a generic phrase.
    """
    return (
        f"Generate 6 synapse weights for a 3-link walking robot given the theorem: "
        f'"{theorem}". The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. '
        f"Translate the structural principle of this theorem — its symmetries, invariants, "
        f"fixed points, transformations, or key relationships — into weight patterns. "
        f'Return ONLY a JSON object like '
        f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
        f"with no other text."
    )


def main():
    random.shuffle(THEOREMS)
    seeds = THEOREMS[:100]  # sample 100 from 108 available
    run_structured_search("theorems", seeds, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
