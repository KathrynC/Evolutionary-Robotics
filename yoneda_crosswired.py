#!/usr/bin/env python3
"""
yoneda_crosswired.py — Phase 7 / A2: Yoneda Crosswired Test

Tests whether expanding from 6-synapse to 10-synapse topology increases
faithfulness (more distinct weight vectors from the LLM).

Selects seeds that collapsed to identical 6-synapse weights (the 39-seed
walk cluster and 20-seed run cluster) and asks Ollama for 10-synapse
weights. If previously-collapsed seeds now produce distinct weight vectors,
this validates the Yoneda-inspired prediction that enriching the target
category increases functor faithfulness.

Saves partial results after each seed (interruptible).

Output: artifacts/yoneda_crosswired_results.json
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import ask_ollama

WEIGHT_NAMES_6 = ["w03", "w04", "w13", "w14", "w23", "w24"]
WEIGHT_NAMES_10 = ["w03", "w04", "w13", "w14", "w23", "w24",
                    "w33", "w34", "w43", "w44"]
OUT_PATH = PROJECT / "artifacts" / "yoneda_crosswired_results.json"
N_SEEDS = 20  # seeds per cluster to test


def make_crosswired_prompt(seed, seed_type):
    """Build prompt requesting 10 weights for crosswired topology."""
    type_desc = {
        "verb": ("the verb", "action quality, intensity, and movement character"),
        "theorem": ("the theorem", "structural principles, symmetries, and dynamics"),
        "verse": ("the verse", "imagery, energy, and movement quality"),
        "place": ("the place", "terrain, climate, and energy"),
    }
    article, qualities = type_desc.get(seed_type, ("the concept", "character and energy"))

    return (
        f"Generate 10 synapse weights for a 3-link walking robot given {article}: "
        f"{seed}. The robot has 3 sensor neurons (0,1,2) and 2 motor neurons (3,4). "
        f"The 10 weights are: w03, w04, w13, w14, w23, w24 (sensor-to-motor), "
        f"w33 (motor 3 self-feedback), w34 (motor 3 to motor 4 cross-wiring), "
        f"w43 (motor 4 to motor 3 cross-wiring), w44 (motor 4 self-feedback). "
        f"Each weight is in [-1, 1]. "
        f"Translate the {qualities} of this {seed_type} into weight magnitudes, "
        f"signs, and symmetry patterns. The motor-to-motor weights (w33, w34, w43, w44) "
        f"control central pattern generator dynamics: self-feedback creates oscillation, "
        f"cross-wiring creates coupling between joints. "
        f'Return ONLY a JSON object like '
        f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2, '
        f'"w33": 0.3, "w34": -0.5, "w43": 0.4, "w44": -0.3}} '
        f"with no other text."
    )


def parse_weights_10(text):
    """Extract 10 weights from LLM response. Returns dict or None."""
    import re
    try:
        obj = json.loads(text)
        if all(k in obj for k in WEIGHT_NAMES_10):
            return {k: float(obj[k]) for k in WEIGHT_NAMES_10}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            obj = json.loads(match.group())
            if all(k in obj for k in WEIGHT_NAMES_10):
                return {k: float(obj[k]) for k in WEIGHT_NAMES_10}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None


def find_collapsed_clusters():
    """Find seeds that collapsed to identical 6-synapse weights."""
    path = PROJECT / "artifacts" / "structured_random_verbs.json"
    if not path.exists():
        return {}

    with open(path) as f:
        trials = json.load(f)

    groups = defaultdict(list)
    for t in trials:
        key = tuple(t["weights"][k] for k in WEIGHT_NAMES_6)
        groups[key].append(t["seed"])

    # Return clusters with >= 5 seeds, sorted by size
    clusters = {}
    for key, seeds in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(seeds) >= 5:
            clusters[str(list(key))] = {
                "weights_6": {k: v for k, v in zip(WEIGHT_NAMES_6, key)},
                "seeds": seeds,
                "n_seeds": len(seeds),
            }
    return clusters


def main():
    print("=" * 60)
    print("YONEDA CROSSWIRED TEST")
    print("  Do previously-collapsed seeds differentiate in 10-synapse space?")
    print("=" * 60)

    clusters = find_collapsed_clusters()
    print(f"\n  Found {len(clusters)} collapsed clusters (>= 5 seeds each):")
    for name, c in clusters.items():
        print(f"    {c['n_seeds']} seeds -> {name[:50]}...")

    # Load partial results
    results = {"clusters": {}, "meta": {}}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"\n  Resuming from partial results")

    results["meta"] = {
        "n_seeds_per_cluster": N_SEEDS,
        "model": "qwen3-coder:30b",
        "temperature": 0.8,
        "topology": "crosswired_10",
    }

    total_calls = 0
    for cluster_key, cluster in clusters.items():
        if cluster_key in results["clusters"]:
            print(f"\n  Cluster {cluster_key[:40]}... already done, skipping")
            continue

        seeds = cluster["seeds"][:N_SEEDS]
        print(f"\n  Testing {len(seeds)} seeds from cluster {cluster_key[:40]}...")
        print(f"  6-synapse weights: {cluster['weights_6']}")

        seed_results = []
        failures = 0

        for i, seed in enumerate(seeds):
            prompt = make_crosswired_prompt(seed, "verb")
            try:
                response = ask_ollama(prompt, temperature=0.8)
                wt = parse_weights_10(response)
                if wt:
                    seed_results.append({
                        "seed": seed,
                        "weights_10": wt,
                        "weights_10_list": [wt[k] for k in WEIGHT_NAMES_10],
                    })
                    # Show just the new 4 weights
                    new4 = [wt["w33"], wt["w34"], wt["w43"], wt["w44"]]
                    print(f"    [{i+1}/{len(seeds)}] {seed[:40]}: "
                          f"new4={new4}")
                    total_calls += 1
                else:
                    failures += 1
                    print(f"    [{i+1}/{len(seeds)}] {seed[:40]}: PARSE FAIL")
                    total_calls += 1
            except Exception as e:
                failures += 1
                print(f"    [{i+1}/{len(seeds)}] {seed[:40]}: ERROR {e}")
                total_calls += 1

        # Analyze this cluster
        if seed_results:
            # Count unique 10-weight vectors
            unique_10 = set()
            for sr in seed_results:
                unique_10.add(tuple(sr["weights_10_list"]))

            # Count unique 6-weight vectors (should be 1 by definition)
            unique_6 = set()
            for sr in seed_results:
                wt = sr["weights_10"]
                unique_6.add(tuple(wt[k] for k in WEIGHT_NAMES_6))

            # Count unique new-4 vectors
            unique_new4 = set()
            for sr in seed_results:
                wt = sr["weights_10"]
                unique_new4.add(tuple(wt[k] for k in ["w33", "w34", "w43", "w44"]))

            faithfulness_6 = 1 / len(seed_results)  # always 1 unique / N
            faithfulness_10 = len(unique_10) / len(seed_results)

            print(f"\n  Cluster analysis:")
            print(f"    Seeds tested: {len(seed_results)} (failures: {failures})")
            print(f"    Unique 6-synapse vectors: {len(unique_6)} "
                  f"(faithfulness = {faithfulness_6:.0%})")
            print(f"    Unique 10-synapse vectors: {len(unique_10)} "
                  f"(faithfulness = {faithfulness_10:.0%})")
            print(f"    Unique new-4 patterns: {len(unique_new4)}")

            # Show the new-4 distribution
            new4_counts = defaultdict(int)
            for sr in seed_results:
                wt = sr["weights_10"]
                new4 = tuple(wt[k] for k in ["w33", "w34", "w43", "w44"])
                new4_counts[new4] += 1
            print(f"    New-4 pattern distribution:")
            for pattern, count in sorted(new4_counts.items(),
                                          key=lambda x: -x[1]):
                print(f"      [{count}x] {list(pattern)}")

            results["clusters"][cluster_key] = {
                "weights_6": cluster["weights_6"],
                "n_seeds": len(seed_results),
                "n_failures": failures,
                "n_unique_6": len(unique_6),
                "n_unique_10": len(unique_10),
                "n_unique_new4": len(unique_new4),
                "faithfulness_6": float(faithfulness_6),
                "faithfulness_10": float(faithfulness_10),
                "seed_results": seed_results,
                "new4_patterns": {str(list(k)): v
                                   for k, v in new4_counts.items()},
            }
        else:
            results["clusters"][cluster_key] = {
                "weights_6": cluster["weights_6"],
                "error": "no successful responses",
                "n_failures": failures,
            }

        # Save after each cluster
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("YONEDA CROSSWIRED SUMMARY")
    print("=" * 60)
    print(f"  Total Ollama calls: {total_calls}")

    for cluster_key, cr in results["clusters"].items():
        if "error" in cr:
            print(f"\n  Cluster {cluster_key[:40]}...: ERROR")
            continue
        print(f"\n  Cluster (6-synapse): {cluster_key[:40]}...")
        print(f"    Faithfulness 6-synapse: {cr['faithfulness_6']:.0%}")
        print(f"    Faithfulness 10-synapse: {cr['faithfulness_10']:.0%}")
        print(f"    Improvement factor: {cr['faithfulness_10']/max(cr['faithfulness_6'], 0.01):.1f}x")
        print(f"    Unique new-4 patterns: {cr['n_unique_new4']}")

    print(f"\n  WROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
