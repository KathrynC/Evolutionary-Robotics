#!/usr/bin/env python3
"""
oeis_retry_deepseek.py

Retry all deepseek-r1:8b parse failures from the OEIS seed experiment
with more tokens and a firmer prompt that front-loads the JSON request.

Reads from the checkpoint or final results, retries failures, and merges
the successful retries back into the results file.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import (
    parse_weights, run_trial_inmemory, OLLAMA_URL
)
from compute_beer_analytics import NumpyEncoder

MODEL = "deepseek-r1:8b"
MAX_TOKENS = 4000  # 2x the original budget

WEIGHT_GRID = [-1.0, -0.7, -0.4, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0]

# Firmer prompt that tells deepseek to keep reasoning very short
PROMPT_TEMPLATE = (
    "Output a JSON object with 6 weights for a walking robot controller. "
    "Keep your reasoning to 1 sentence MAX, then output the JSON.\n\n"
    "Robot: 3-link (Torso, BackLeg, FrontLeg), 2 hinge joints.\n"
    "Control: m3 = tanh(w03*s0 + w13*s1 + w23*s2), m4 = tanh(w04*s0 + w14*s1 + w24*s2)\n"
    "Sensors: s0=torso_contact, s1=back_foot, s2=front_foot (0/1)\n\n"
    "Translate this integer sequence into movement weights:\n"
    "  {seq_id} — {seq_name}\n"
    "  Terms: {terms}\n\n"
    "Choose each weight from: -1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0\n\n"
    "ONE sentence about the movement quality, then the JSON:\n"
    '{{\"w03\": ..., \"w04\": ..., \"w13\": ..., \"w14\": ..., \"w23\": ..., \"w24\": ...}}'
)


def snap_to_grid(weights):
    return {k: min(WEIGHT_GRID, key=lambda g: abs(g - v)) for k, v in weights.items()}


def query_deepseek(prompt, timeout=180):
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": MAX_TOKENS}
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=timeout
        )
        if r.returncode != 0:
            return None, f"curl error: {r.stderr}"
        data = json.loads(r.stdout)
        if "error" in data:
            return None, f"ollama error: {data['error']}"
        resp = data["response"]
        weights = parse_weights(resp)
        if weights is not None:
            weights = snap_to_grid(weights)
        return weights, resp
    except Exception as e:
        return None, str(e)


def main():
    # Load results (prefer final, fall back to checkpoint)
    final = PROJECT / "artifacts" / "oeis_seed_experiment.json"
    checkpoint = PROJECT / "artifacts" / "oeis_seed_experiment_checkpoint.json"
    results_path = final if final.exists() else checkpoint

    if not results_path.exists():
        print("No OEIS experiment results found. Run oeis_seed_experiment.py first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} results from {results_path.name}")

    # Find deepseek failures
    failures = [r for r in results if r["model"] == MODEL and not r["success"]]
    ds_total = [r for r in results if r["model"] == MODEL]
    print(f"deepseek-r1: {len(ds_total)} total, {len(failures)} failures ({100*len(failures)/max(len(ds_total),1):.0f}%)")

    if not failures:
        print("No failures to retry!")
        return

    # Load OEIS cache for term data
    cache_dir = PROJECT / "artifacts" / "oeis_cache"

    print(f"\nRetrying {len(failures)} failures with {MAX_TOKENS} max tokens...")
    print()

    fixed = 0
    still_failed = 0

    for i, fail in enumerate(failures):
        seq_id = fail["seq_id"]
        seq_name = fail["seq_name"]
        terms = fail.get("seq_terms", [])

        # If terms not in the failure record, try the cache
        if not terms:
            cache_file = cache_dir / f"{seq_id}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                terms_str = cached.get("data", "")
                terms = [int(t) for t in terms_str.split(",")[:16] if t.strip()]

        terms_str = ", ".join(str(t) for t in terms[:16])
        prompt = PROMPT_TEMPLATE.format(
            seq_id=seq_id,
            seq_name=seq_name,
            terms=terms_str,
        )

        print(f"[{i+1}/{len(failures)}] {seq_id} {seq_name[:50]}...", end=" ", flush=True)

        weights, raw_resp = query_deepseek(prompt)

        if weights is None:
            still_failed += 1
            print("STILL FAILED")
            continue

        # Simulate
        try:
            analytics = run_trial_inmemory(weights)
        except Exception as e:
            still_failed += 1
            print(f"SIM ERROR: {e}")
            continue

        dx = analytics["outcome"]["dx"]
        dy = analytics["outcome"]["dy"]
        speed = analytics["outcome"]["mean_speed"]
        print(f"FIXED! DX={dx:+.2f} DY={dy:+.2f} spd={speed:.2f}")

        # Update the result in-place
        idx = results.index(fail)
        results[idx] = {
            "seq_id": seq_id,
            "seq_name": seq_name,
            "seq_terms": terms[:16],
            "model": MODEL,
            "success": True,
            "weights": weights,
            "analytics": analytics,
            "retry": True,
        }
        fixed += 1

    # Save updated results
    data["results"] = results
    data["metadata"]["deepseek_retry"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attempted": len(failures),
        "fixed": fixed,
        "still_failed": still_failed,
        "max_tokens": MAX_TOKENS,
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*50}")
    print(f"RETRY COMPLETE")
    print(f"{'='*50}")
    print(f"Attempted: {len(failures)}")
    print(f"Fixed: {fixed}")
    print(f"Still failed: {still_failed}")
    print(f"Updated: {results_path}")


if __name__ == "__main__":
    main()
