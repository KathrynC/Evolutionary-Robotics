#!/usr/bin/env python3
"""Re-run only the failed trials from motion_seed_experiment.json."""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import parse_weights, run_trial_inmemory, WEIGHT_NAMES, OLLAMA_URL
from compute_beer_analytics import NumpyEncoder
from motion_seed_experiment import (
    build_seeds, MODELS, PROMPT_TEMPLATE, CORE_CONCEPTS,
    query_ollama, query_openai, REASONING_MODELS,
)

def query_model(model_info, prompt):
    if model_info["type"] == "ollama":
        return query_ollama(model_info["name"], prompt)
    elif model_info["type"] == "openai":
        return query_openai(model_info["name"], prompt)
    return None, "unknown model type"


def main():
    results_path = PROJECT / "artifacts" / "motion_seed_experiment.json"
    with open(results_path) as f:
        data = json.load(f)

    existing = data["results"]
    seeds = build_seeds()
    seed_lookup = {s["seed"]: s for s in seeds}

    # Find failed trials
    failed = [r for r in existing if not r["success"]]
    print(f"Found {len(failed)} failed trials to re-run")
    print(f"  deepseek-r1:8b:  {sum(1 for r in failed if r['model'] == 'deepseek-r1:8b')}")
    print(f"  gpt-oss:20b:     {sum(1 for r in failed if r['model'] == 'gpt-oss:20b')}")
    print(f"  gpt-4.1-mini:    {sum(1 for r in failed if r['model'] == 'gpt-4.1-mini')}")
    print()

    model_lookup = {m["name"]: m for m in MODELS}
    new_results = []
    start = time.time()

    for i, trial in enumerate(failed):
        seed_word = trial["seed"]
        model_name = trial["model"]
        concept = trial["concept"]
        lang = trial["language"]
        model_info = model_lookup[model_name]
        prompt = PROMPT_TEMPLATE.format(seed=seed_word)

        print(f"[{i+1}/{len(failed)}] {model_name} | {lang}:{seed_word}", end=" ", flush=True)

        weights, raw_resp = query_model(model_info, prompt)

        if weights is None:
            print(f"-> STILL FAIL")
            new_results.append(trial)  # keep original failure record
            continue

        try:
            analytics = run_trial_inmemory(weights)
        except Exception as e:
            print(f"-> SIM ERROR: {e}")
            new_results.append(trial)
            continue

        # Score semantic match
        seed_info = seed_lookup.get(seed_word)
        try:
            match = seed_info["criteria_fn"](analytics) if seed_info else None
        except Exception:
            match = None

        dx = analytics["outcome"]["dx"]
        dy = analytics["outcome"]["dy"]
        yaw = analytics["outcome"]["yaw_net_rad"]
        print(f"-> DX={dx:+.2f} DY={dy:+.2f} YAW={yaw:+.1f} match={match}")

        new_results.append({
            "seed": seed_word, "concept": concept, "language": lang,
            "model": model_name, "success": True,
            "weights": weights, "analytics": analytics, "match": match,
        })

    elapsed = time.time() - start
    print(f"\nRe-run done in {elapsed:.1f}s")

    # Merge: replace failed trials with new results
    success_existing = [r for r in existing if r["success"]]
    merged = success_existing + new_results
    print(f"Merged: {len(success_existing)} original successes + {len(new_results)} re-run results = {len(merged)} total")

    # Count improvements
    new_successes = sum(1 for r in new_results if r["success"])
    still_failed = sum(1 for r in new_results if not r["success"])
    print(f"  New successes: {new_successes}, still failed: {still_failed}")

    # Save merged results
    data["results"] = merged
    data["metadata"]["rerun_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["metadata"]["rerun_elapsed_seconds"] = elapsed
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"Saved merged results to {results_path}")

    # Print updated summary
    print("\n" + "="*70)
    print("UPDATED PER-MODEL SUMMARY")
    print("="*70)
    for model_info in MODELS:
        mname = model_info["name"]
        m_results = [r for r in merged if r["model"] == mname]
        successes = [r for r in m_results if r["success"]]
        core_results = [r for r in successes if r["concept"] in CORE_CONCEPTS and r["match"] is not None]
        core_matches = [r for r in core_results if r["match"]]

        print(f"\n{mname}:")
        print(f"  Parse success: {len(successes)}/{len(m_results)}")
        if core_results:
            print(f"  Semantic match (core 12 concepts): {len(core_matches)}/{len(core_results)} "
                  f"({100*len(core_matches)/len(core_results):.0f}%)")
        dxs = [r["analytics"]["outcome"]["dx"] for r in successes if r.get("analytics")]
        if dxs:
            print(f"  DX: median={np.median(dxs):+.2f}, max |DX|={max(abs(d) for d in dxs):.2f}")

    # Updated match grid
    print("\n" + "="*70)
    print("UPDATED MATCH GRID: concept (rows) × model (columns)")
    print("="*70)
    model_names = [m["name"] for m in MODELS]
    header = f"{'concept':20s} | " + " | ".join(f"{m[:12]:>12s}" for m in model_names)
    print(header)
    print("-" * len(header))
    for concept_id in CORE_CONCEPTS:
        row = f"{concept_id:20s} | "
        for mname in model_names:
            cm = [r for r in merged if r["concept"] == concept_id and r["model"] == mname
                  and r["success"] and r.get("match") is not None]
            cm_match = [r for r in cm if r["match"]]
            if cm:
                row += f"{len(cm_match):>2d}/{len(cm):>2d} ({100*len(cm_match)/len(cm):3.0f}%) | "
            else:
                row += f"{'---':>12s} | "
        print(row)


if __name__ == "__main__":
    main()
