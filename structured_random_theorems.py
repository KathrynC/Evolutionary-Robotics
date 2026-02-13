#!/usr/bin/env python3
"""
structured_random_theorems.py

Structured random search condition #2: Random mathematical theorems.

Selects 100 random theorems spanning algebra, geometry, analysis, topology,
number theory, and combinatorics, asks a local LLM to translate each theorem's
structural principle into 6 synapse weights, then runs headless simulations
with Beer-framework analytics.

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

THEOREMS = [
    # Analysis
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
    # Algebra
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
    # Geometry and Topology
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
    # Dynamical Systems
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
    seeds = THEOREMS[:100]
    run_structured_search("theorems", seeds, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
