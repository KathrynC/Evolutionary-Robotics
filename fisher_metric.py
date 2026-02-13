#!/usr/bin/env python3
"""
fisher_metric.py — Phase 7 / A1: Fisher Metric Estimation

Calls Ollama N times per seed to measure the LLM's output variance,
building the statistical manifold on Sem needed for Hilbert formalization.

Saves partial results after each seed so interruption loses minimal work.

Output: artifacts/fisher_metric_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import ask_ollama

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
OUT_PATH = PROJECT / "artifacts" / "fisher_metric_results.json"
N_REPEATS = 10  # calls per seed

# Stratified sample: seeds from each condition that appeared in the original data
SEEDS = {
    "verbs": [
        ("stumble (English, to trip and nearly fall)", "verb"),
        ("stolpern (German, to stumble)", "verb"),
        ("cascade (English, to fall in a series of stages)", "verb"),
        ("lurch (English, to move with sudden unsteady movements)", "verb"),
        ("sprint (English, to run at full speed)", "verb"),
        ("tiptoe (English, to walk quietly on the toes)", "verb"),
        ("crawl (English, to move on hands and knees)", "verb"),
        ("leap (English, to jump a long distance)", "verb"),
        ("waddle (English, to walk with short steps swaying side to side)", "verb"),
        ("zou (Mandarin, to walk)", "verb"),
        ("prygat (Russian, to jump)", "verb"),
        ("schwanken (German, to sway or waver)", "verb"),
    ],
    "theorems": [
        ("Noether's Theorem", "theorem"),
        ("KAM Theorem", "theorem"),
        ("Konig's Theorem", "theorem"),
        ("Pythagorean Theorem", "theorem"),
        ("Sturm-Liouville Theory", "theorem"),
        ("Euler's Identity", "theorem"),
    ],
    "bible": [
        ("Revelation 6:8 — And I looked, and behold a pale horse: and his name that sat on him was Death.", "verse"),
        ("Ecclesiastes 1:6 — The wind goeth toward the south, and turneth about unto the north; it whirleth about continually.", "verse"),
        ("Psalm 29:3 — The voice of the LORD is upon the waters.", "verse"),
        ("1 Corinthians 15:52 — In a moment, in the twinkling of an eye, at the last trump.", "verse"),
        ("Exodus 14:21 — And the LORD caused the sea to go back by a strong east wind all that night.", "verse"),
        ("Isaiah 40:31 — But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles.", "verse"),
    ],
    "places": [
        ("Svalbard", "place"),
        ("Chichen Itza, Mexico", "place"),
        ("Bialowieza Forest, Poland", "place"),
        ("Mariana Trench", "place"),
        ("Sahara Desert", "place"),
        ("Mount Fuji, Japan", "place"),
    ],
}


def make_prompt(seed, seed_type):
    """Build prompt matching original condition scripts."""
    if seed_type == "verb":
        return (
            f"Generate 6 synapse weights for a 3-link walking robot given the verb: "
            f"{seed}. The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. "
            f"Translate the action quality, intensity, and movement character of this "
            f"verb into weight magnitudes, signs, and symmetry patterns. "
            f'Return ONLY a JSON object like '
            f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
            f"with no other text."
        )
    elif seed_type == "theorem":
        return (
            f"Generate 6 synapse weights for a 3-link walking robot given the theorem: "
            f"{seed}. The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. "
            f"Translate the structural principles, symmetries, and dynamics of this "
            f"theorem into weight magnitudes, signs, and symmetry patterns. "
            f'Return ONLY a JSON object like '
            f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
            f"with no other text."
        )
    elif seed_type == "verse":
        return (
            f"Generate 6 synapse weights for a 3-link walking robot given the verse: "
            f"{seed}. The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. "
            f"Translate the imagery, energy, and movement quality of this "
            f"verse into weight magnitudes, signs, and symmetry patterns. "
            f'Return ONLY a JSON object like '
            f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
            f"with no other text."
        )
    else:  # place
        return (
            f"Generate 6 synapse weights for a 3-link walking robot given the place: "
            f"{seed}. The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. "
            f"Translate the terrain, climate, and energy of this "
            f"place into weight magnitudes, signs, and symmetry patterns. "
            f'Return ONLY a JSON object like '
            f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
            f"with no other text."
        )


def parse_weights(text):
    """Extract 6 weights from LLM response. Returns dict or None."""
    import re
    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if all(k in obj for k in WEIGHT_NAMES):
            return {k: float(obj[k]) for k in WEIGHT_NAMES}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Try to find JSON in text
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            obj = json.loads(match.group())
            if all(k in obj for k in WEIGHT_NAMES):
                return {k: float(obj[k]) for k in WEIGHT_NAMES}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None


def main():
    print("=" * 60)
    print("FISHER METRIC ESTIMATION")
    print(f"  {N_REPEATS} Ollama calls per seed")
    print("=" * 60)

    # Load partial results if they exist
    results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"  Resuming: {len(results.get('seeds', {}))} seeds already done")

    if "seeds" not in results:
        results["seeds"] = {}
    results["meta"] = {
        "n_repeats": N_REPEATS,
        "model": "qwen3-coder:30b",
        "temperature": 0.8,
    }

    total_seeds = sum(len(v) for v in SEEDS.values())
    done = 0

    for condition, seed_list in SEEDS.items():
        for seed, seed_type in seed_list:
            key = f"{condition}::{seed}"
            if key in results["seeds"]:
                done += 1
                continue

            print(f"\n  [{done+1}/{total_seeds}] {condition}: {seed[:50]}...")
            prompt = make_prompt(seed, seed_type)
            weight_samples = []
            failures = 0

            for rep in range(N_REPEATS):
                try:
                    response = ask_ollama(prompt, temperature=0.8)
                    wt = parse_weights(response)
                    if wt:
                        weight_samples.append([wt[k] for k in WEIGHT_NAMES])
                        print(f"    rep {rep}: {[wt[k] for k in WEIGHT_NAMES]}")
                    else:
                        failures += 1
                        print(f"    rep {rep}: PARSE FAIL")
                except Exception as e:
                    failures += 1
                    print(f"    rep {rep}: ERROR {e}")

            if len(weight_samples) >= 2:
                wmat = np.array(weight_samples)
                mu = wmat.mean(axis=0).tolist()
                cov = np.cov(wmat.T).tolist()
                std = wmat.std(axis=0).tolist()

                results["seeds"][key] = {
                    "condition": condition,
                    "seed": seed,
                    "n_success": len(weight_samples),
                    "n_failures": failures,
                    "mean": mu,
                    "std": std,
                    "covariance": cov,
                    "samples": [s for s in weight_samples],
                    "all_identical": all(
                        np.allclose(weight_samples[0], s) for s in weight_samples[1:]
                    ),
                }
            else:
                results["seeds"][key] = {
                    "condition": condition,
                    "seed": seed,
                    "n_success": len(weight_samples),
                    "n_failures": failures,
                    "error": "insufficient samples",
                }

            # Save after each seed (interruptible)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

            done += 1

    # Summary
    print("\n" + "=" * 60)
    print("FISHER METRIC SUMMARY")
    print("=" * 60)
    n_identical = sum(1 for s in results["seeds"].values()
                      if s.get("all_identical", False))
    n_varied = sum(1 for s in results["seeds"].values()
                   if not s.get("all_identical", True) and s.get("n_success", 0) >= 2)
    print(f"  Seeds completed: {len(results['seeds'])}/{total_seeds}")
    print(f"  All-identical (deterministic): {n_identical}")
    print(f"  Varied (stochastic): {n_varied}")

    # Per-condition variance summary
    for condition in SEEDS:
        cond_seeds = [s for s in results["seeds"].values()
                      if s.get("condition") == condition and "std" in s]
        if cond_seeds:
            mean_std = np.mean([np.mean(s["std"]) for s in cond_seeds])
            print(f"  {condition}: mean per-weight std = {mean_std:.4f}")

    print(f"\nWROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
