#!/usr/bin/env python3
"""
decollapse_weights.py

Detect weight-vector collapse in experiment results and retry collapsed trials
with enriched prompts. Weight collapse is when an LLM produces identical weight
vectors for semantically distinct inputs — e.g., the same 6 weights for both
"Saul Goodman" and "Applejack".

Strategy:
  1. Load experiment results and identify collapsed clusters (same weight vector
     used for multiple distinct inputs)
  2. For each collapsed trial, use an LLM to generate a rich distinguishing
     description of the input (character personality, sequence mathematical
     properties, etc.)
  3. Re-prompt the ORIGINAL model with this enriched context
  4. Re-simulate any trial that produces a NEW weight vector
  5. Merge new results back with a "decollapsed" tag

Works with: character_seed_experiment, oeis_seed_experiment, and future
experiment types that follow the same checkpoint format.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import (
    parse_weights, run_trial_inmemory,
    WEIGHT_NAMES, OLLAMA_URL
)
from compute_beer_analytics import NumpyEncoder

ARTIFACTS = PROJECT / "artifacts"

# Model used to generate enrichment context (different from weight-gen models
# to avoid self-reinforcing the same collapse patterns)
ENRICHMENT_MODEL = "llama3.1:latest"

# Weight grid (must match the experiment's grid)
WEIGHT_GRID = [-1.0, -0.7, -0.4, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0]


def snap_to_grid(weights):
    return {k: min(WEIGHT_GRID, key=lambda g: abs(g - v))
            for k, v in weights.items()}


def query_ollama(model, prompt, temperature=0.8, max_tokens=500, timeout=120):
    """Query Ollama and return (weights_dict, raw_response) or (None, error)."""
    reasoning_models = {"deepseek-r1:8b", "gpt-oss:20b"}
    effective_max = 2000 if model in reasoning_models else max_tokens
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": effective_max}
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
        # Strip <think> tags for reasoning models
        if "</think>" in resp:
            resp = resp.split("</think>")[-1]
        weights = parse_weights(resp)
        if weights is not None:
            weights = snap_to_grid(weights)
        return weights, resp
    except Exception as e:
        return None, str(e)


def weight_key(weights):
    """Convert weight dict to a hashable tuple."""
    return tuple(weights[k] for k in sorted(weights.keys()))


# ── Collapse detection ───────────────────────────────────────────────────────

def find_collapsed_clusters(results, min_cluster_size=2):
    """Find weight vectors used by multiple distinct inputs.

    Returns dict mapping weight_tuple -> list of result indices.
    Only includes clusters with min_cluster_size or more members.
    """
    wt_to_indices = defaultdict(list)
    for i, r in enumerate(results):
        if r.get("weights") and r.get("success", True):
            wt = weight_key(r["weights"])
            wt_to_indices[wt].append(i)

    return {wt: indices for wt, indices in wt_to_indices.items()
            if len(indices) >= min_cluster_size}


# ── Enrichment prompts ───────────────────────────────────────────────────────

def enrich_character(character, story):
    """Ask an LLM for a rich character description to break collapse."""
    prompt = (
        f"Describe the character {character} from {story} in 3-4 vivid sentences. "
        f"Focus on their PHYSICAL presence and movement style — how they walk, "
        f"their posture, their energy level, their nervous habits, their gait. "
        f"Also mention their emotional core and how their personality manifests "
        f"physically. Be specific and distinctive — what makes THIS character's "
        f"movement different from everyone else's?\n\n"
        f"Write ONLY the description, nothing else."
    )
    payload = json.dumps({
        "model": ENRICHMENT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300}
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=60
        )
        data = json.loads(r.stdout)
        return data.get("response", "").strip()
    except Exception:
        return ""


def enrich_sequence(seq_id, seq_name, terms):
    """Ask an LLM for a rich mathematical characterization."""
    prompt = (
        f"Describe the mathematical personality of OEIS sequence {seq_id} "
        f"({seq_name}) in 3-4 sentences. First terms: {terms}\n\n"
        f"Focus on what makes this sequence DISTINCTIVE: its growth behavior, "
        f"its rhythm, whether it's smooth or jagged, explosive or gentle, "
        f"regular or chaotic. Compare it to physical movement or music if helpful. "
        f"What FEELING does this sequence evoke?\n\n"
        f"Write ONLY the description."
    )
    payload = json.dumps({
        "model": ENRICHMENT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300}
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=60
        )
        data = json.loads(r.stdout)
        return data.get("response", "").strip()
    except Exception:
        return ""


# ── Enriched retry prompts ───────────────────────────────────────────────────

CHARACTER_RETRY_PROMPT = (
    "You are designing a neural controller for a 3-link walking robot "
    "(Torso, BackLeg, FrontLeg) with two hinge joints.\n\n"
    "Sensors: s0=torso_contact, s1=back_foot_contact, s2=front_foot_contact\n"
    "Motors: m3=back_joint_angle, m4=front_joint_angle\n"
    "Control law: m3 = tanh(w03*s0 + w13*s1 + w23*s2), "
    "m4 = tanh(w04*s0 + w14*s1 + w24*s2)\n\n"
    "Weight roles:\n"
    "  w03, w04: torso touch → motors (balance response)\n"
    "  w13, w24: foot → same leg (local reflex)\n"
    "  w14, w23: foot → opposite leg (cross-coupling)\n\n"
    "Here is a detailed description of the character whose movement you must "
    "capture:\n\n"
    "CHARACTER: {character} from {story}\n"
    "{enrichment}\n\n"
    "IMPORTANT: Think carefully about what makes THIS character physically "
    "distinctive. A nervous character should have different weights than a "
    "confident one. A heavy character different from a light one. A villain "
    "different from a hero.\n\n"
    "Previously, you gave this character IDENTICAL weights to very different "
    "characters. This time, focus on what makes {character} UNIQUE.\n\n"
    "Choose each weight from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].\n"
    "Output ONLY the JSON: {{\"w03\": ..., \"w04\": ..., \"w13\": ..., "
    "\"w14\": ..., \"w23\": ..., \"w24\": ...}}"
)

OEIS_RETRY_PROMPT = (
    "You are designing a neural controller for a 3-link walking robot "
    "(Torso, BackLeg, FrontLeg) with two hinge joints.\n\n"
    "Sensors: s0=torso_contact, s1=back_foot_contact, s2=front_foot_contact\n"
    "Motors: m3=back_joint_angle, m4=front_joint_angle\n"
    "Control law: m3 = tanh(w03*s0 + w13*s1 + w23*s2), "
    "m4 = tanh(w04*s0 + w14*s1 + w24*s2)\n\n"
    "Weight roles:\n"
    "  w03, w04: torso touch → motors (balance response)\n"
    "  w13, w24: foot → same leg (local reflex)\n"
    "  w14, w23: foot → opposite leg (cross-coupling)\n\n"
    "Translate this mathematical sequence into robot movement:\n\n"
    "Sequence: {seq_id} — {seq_name}\n"
    "First terms: {terms}\n\n"
    "Mathematical character:\n{enrichment}\n\n"
    "IMPORTANT: Think carefully about what makes THIS sequence distinctive. "
    "Previously, you gave this sequence the same weights as a very different "
    "sequence. This time, focus on its unique mathematical personality.\n\n"
    "Choose each weight from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].\n"
    "Output ONLY the JSON: {{\"w03\": ..., \"w04\": ..., \"w13\": ..., "
    "\"w14\": ..., \"w23\": ..., \"w24\": ...}}"
)


# ── Experiment type detection ────────────────────────────────────────────────

def detect_experiment_type(results):
    """Determine experiment type from result keys."""
    if not results:
        return "unknown"
    r = results[0]
    if "character" in r and "story" in r:
        return "character"
    if "seq_id" in r:
        return "oeis"
    if "mathematician" in r or "name" in r:
        return "mathematician"
    return "unknown"


def get_input_label(result, exp_type):
    """Get a human-readable label for a result."""
    if exp_type == "character":
        return f"{result['character']} ({result['story']})"
    elif exp_type == "oeis":
        return f"{result.get('seq_id', '?')} {result.get('seq_name', '')[:40]}"
    else:
        return str(result.get("name", result.get("seed", "?")))


# ── Main decollapse pipeline ─────────────────────────────────────────────────

def decollapse(input_file, output_file=None, min_cluster=2,
               max_retries_per_trial=2, temperature=0.9,
               skip_models=None, only_models=None):
    """
    Main decollapse pipeline.

    Args:
        input_file: Path to experiment checkpoint/results JSON
        output_file: Path for decollapse report (default: auto-generated)
        min_cluster: Minimum cluster size to consider collapsed
        max_retries_per_trial: Retry attempts per collapsed trial
        temperature: Higher temp for retry (default 0.9 vs original 0.8)
        skip_models: Set of model names to skip
        only_models: If set, only retry these models
    """
    with open(input_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    exp_type = detect_experiment_type(results)
    print(f"Experiment type: {exp_type}")
    print(f"Total results: {len(results)}")

    # Find collapsed clusters
    clusters = find_collapsed_clusters(results, min_cluster)
    total_collapsed = sum(len(indices) for indices in clusters.values())
    print(f"Collapsed clusters: {len(clusters)}")
    print(f"Total trials in collapsed clusters: {total_collapsed}")
    print()

    if not clusters:
        print("No collapsed clusters found. Nothing to do.")
        return

    # Sort clusters by size (worst first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    # Report
    print("Top collapsed clusters:")
    for wt, indices in sorted_clusters[:10]:
        w_dict = dict(zip(sorted(WEIGHT_NAMES), wt))
        models = Counter(results[i]["model"] for i in indices)
        labels = [get_input_label(results[i], exp_type) for i in indices[:3]]
        print(f"  {len(indices)}x: {w_dict}")
        print(f"    models: {dict(models)}")
        print(f"    examples: {labels}")
    print()

    # Build retry list: for each cluster, keep one "anchor" (first occurrence)
    # and retry all others
    retry_list = []
    for wt, indices in sorted_clusters:
        # Keep the first occurrence as-is; retry the rest
        anchor_idx = indices[0]
        anchor = results[anchor_idx]
        for idx in indices[1:]:
            r = results[idx]
            model = r["model"]
            if skip_models and model in skip_models:
                continue
            if only_models and model not in only_models:
                continue
            retry_list.append({
                "index": idx,
                "model": model,
                "original_weights": r["weights"],
                "collapsed_with": get_input_label(anchor, exp_type),
                "result": r,
            })

    print(f"Trials to retry: {len(retry_list)}")
    print()

    # Cache enrichment descriptions to avoid duplicate LLM calls
    enrichment_cache = {}
    decollapse_results = []
    broken = 0
    still_collapsed = 0
    parse_fail = 0
    t0 = time.time()

    for i, trial in enumerate(retry_list):
        r = trial["result"]
        model = trial["model"]
        label = get_input_label(r, exp_type)
        old_wt = weight_key(r["weights"])

        # Generate enrichment if not cached
        if exp_type == "character":
            cache_key = (r["character"], r["story"])
            if cache_key not in enrichment_cache:
                enrichment_cache[cache_key] = enrich_character(
                    r["character"], r["story"])
            enrichment = enrichment_cache[cache_key]

            prompt = CHARACTER_RETRY_PROMPT.format(
                character=r["character"],
                story=r["story"],
                enrichment=enrichment,
            )

        elif exp_type == "oeis":
            cache_key = r.get("seq_id", "?")
            if cache_key not in enrichment_cache:
                enrichment_cache[cache_key] = enrich_sequence(
                    r.get("seq_id", "?"),
                    r.get("seq_name", ""),
                    r.get("terms", []),
                )
            enrichment = enrichment_cache[cache_key]

            prompt = OEIS_RETRY_PROMPT.format(
                seq_id=r.get("seq_id", "?"),
                seq_name=r.get("seq_name", ""),
                terms=r.get("terms", []),
                enrichment=enrichment,
            )
        else:
            continue  # Unknown experiment type

        # Retry with enriched prompt
        new_weights = None
        raw_resp = None
        for attempt in range(max_retries_per_trial):
            new_weights, raw_resp = query_ollama(
                model, prompt,
                temperature=temperature,
                max_tokens=500,
            )
            if new_weights is not None:
                break
            time.sleep(0.5)

        if new_weights is None:
            parse_fail += 1
            status = "parse_fail"
            print(f"  [{i+1}/{len(retry_list)}] {model[:15]} | {label[:40]} -> PARSE FAIL")
        else:
            new_wt = weight_key(new_weights)
            if new_wt == old_wt:
                still_collapsed += 1
                status = "still_collapsed"
                print(f"  [{i+1}/{len(retry_list)}] {model[:15]} | {label[:40]} -> SAME (still collapsed)")
            else:
                broken += 1
                status = "broken"
                # Run simulation with new weights
                try:
                    analytics = run_trial_inmemory(new_weights)
                    dx = analytics["outcome"]["displacement"]["dx"]
                    dy = analytics["outcome"]["displacement"]["dy"]
                    spd = analytics["outcome"]["speed"]["mean"]
                    print(f"  [{i+1}/{len(retry_list)}] {model[:15]} | {label[:40]} -> "
                          f"BROKEN! DX={dx:+.2f} DY={dy:+.2f} spd={spd:.2f}")
                except Exception as e:
                    analytics = None
                    print(f"  [{i+1}/{len(retry_list)}] {model[:15]} | {label[:40]} -> "
                          f"BROKEN (new weights) but sim failed: {e}")

        entry = {
            "index": trial["index"],
            "model": model,
            "label": label,
            "status": status,
            "old_weights": r["weights"],
            "new_weights": dict(zip(sorted(WEIGHT_NAMES),
                                    weight_key(new_weights))) if new_weights else None,
            "enrichment": enrichment,
            "collapsed_with": trial["collapsed_with"],
        }
        if status == "broken" and analytics:
            entry["analytics"] = analytics
            entry["new_dx"] = analytics["outcome"]["displacement"]["dx"]
            entry["new_dy"] = analytics["outcome"]["displacement"]["dy"]
            entry["new_speed"] = analytics["outcome"]["speed"]["mean"]

        decollapse_results.append(entry)

        # Progress checkpoint every 50 trials
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(retry_list) - i - 1) / rate if rate > 0 else 0
            print(f"  [checkpoint] {i+1}/{len(retry_list)} done, "
                  f"{elapsed:.0f}s elapsed, ~{remaining/60:.0f}min remaining")
            print(f"    broken={broken}, still_collapsed={still_collapsed}, "
                  f"parse_fail={parse_fail}")

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    print("DECOLLAPSE COMPLETE")
    print("=" * 60)
    print(f"Total retried: {len(retry_list)}")
    print(f"  Broken (new weights): {broken} ({100*broken/max(1,len(retry_list)):.1f}%)")
    print(f"  Still collapsed:      {still_collapsed} ({100*still_collapsed/max(1,len(retry_list)):.1f}%)")
    print(f"  Parse failures:       {parse_fail} ({100*parse_fail/max(1,len(retry_list)):.1f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Enrichment cache size: {len(enrichment_cache)} descriptions")

    # Per-model breakdown
    model_stats = defaultdict(lambda: {"broken": 0, "same": 0, "fail": 0, "total": 0})
    for dr in decollapse_results:
        m = dr["model"]
        model_stats[m]["total"] += 1
        if dr["status"] == "broken":
            model_stats[m]["broken"] += 1
        elif dr["status"] == "still_collapsed":
            model_stats[m]["same"] += 1
        else:
            model_stats[m]["fail"] += 1

    print("\nPer-model breakdown:")
    for m, s in sorted(model_stats.items()):
        pct = 100 * s["broken"] / max(1, s["total"])
        print(f"  {m}: {s['broken']}/{s['total']} broken ({pct:.0f}%), "
              f"{s['same']} still collapsed, {s['fail']} parse fail")

    # Save report
    if output_file is None:
        stem = Path(input_file).stem.replace("_checkpoint", "")
        output_file = ARTIFACTS / f"{stem}_decollapse.json"

    report = {
        "source_file": str(input_file),
        "experiment_type": exp_type,
        "total_results": len(results),
        "total_collapsed_trials": total_collapsed,
        "total_retried": len(retry_list),
        "broken": broken,
        "still_collapsed": still_collapsed,
        "parse_fail": parse_fail,
        "elapsed_seconds": elapsed,
        "temperature": temperature,
        "enrichment_model": ENRICHMENT_MODEL,
        "enrichment_cache_size": len(enrichment_cache),
        "model_stats": dict(model_stats),
        "trials": decollapse_results,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {output_file}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and break weight-vector collapse in experiment results")
    parser.add_argument("input", help="Experiment checkpoint JSON file")
    parser.add_argument("--output", help="Output file (default: auto)")
    parser.add_argument("--min-cluster", type=int, default=2,
                        help="Minimum cluster size to consider collapsed (default: 2)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="LLM temperature for retry (default: 0.9, higher than original)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries per trial on parse failure")
    parser.add_argument("--skip-models", nargs="+",
                        help="Skip these models")
    parser.add_argument("--only-models", nargs="+",
                        help="Only retry these models")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Limit number of trials to retry (for testing)")

    args = parser.parse_args()

    decollapse(
        input_file=args.input,
        output_file=args.output,
        min_cluster=args.min_cluster,
        temperature=args.temperature,
        max_retries_per_trial=args.max_retries,
        skip_models=set(args.skip_models) if args.skip_models else None,
        only_models=set(args.only_models) if args.only_models else None,
    )
