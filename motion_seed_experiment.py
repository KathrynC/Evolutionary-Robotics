#!/usr/bin/env python3
"""
motion_seed_experiment.py

Batch test: 100 motion seeds across 5 LLMs.
Tests whether LLM-generated weights produce gaits that match the semantic
motion instruction. Each seed is a motion word (e.g., "stumble", "glide")
in English or another language.

Pipeline per trial:
  1. Prompt LLM: "Generate weights that embody [motion word]"
  2. Parse 6 weights from LLM response
  3. Run headless PyBullet simulation (4000 steps @ 240 Hz)
  4. Compute Beer-framework analytics
  5. Score: does the behavior match the semantic intent?

Uses 4 local Ollama models + OpenAI API.
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
    parse_weights, write_brain, run_trial_inmemory,
    WEIGHT_NAMES, OLLAMA_URL
)
from compute_beer_analytics import NumpyEncoder

# ── Motion seed lexicon ──────────────────────────────────────────────────────
# 12 core concepts × 5 languages = 60, plus 40 English synonyms/variations = 100

CORE_CONCEPTS = {
    "forward_walk": {
        "en": "forward walk", "de": "Vorwärtsgang", "zh": "向前走",
        "fr": "marche avant", "fi": "eteenpäin kävely",
        "criteria": lambda a: a["outcome"]["dx"] > 5 and abs(a["outcome"]["dy"]) < abs(a["outcome"]["dx"]),
        "description": "positive DX > 5, |DY| < |DX|"
    },
    "backward_walk": {
        "en": "backward walk", "de": "Rückwärtsgang", "zh": "向后走",
        "fr": "marche arrière", "fi": "taaksepäin kävely",
        "criteria": lambda a: a["outcome"]["dx"] < -5 and abs(a["outcome"]["dy"]) < abs(a["outcome"]["dx"]),
        "description": "negative DX < -5, |DY| < |DX|"
    },
    "crab_walk": {
        "en": "crab walk", "de": "Seitwärtsgang", "zh": "横着走",
        "fr": "marche latérale", "fi": "ravun kävely",
        "criteria": lambda a: abs(a["outcome"]["dy"]) > 3 and abs(a["outcome"]["dy"]) > abs(a["outcome"]["dx"]),
        "description": "|DY| > 3 and |DY| > |DX|"
    },
    "spin": {
        "en": "spin", "de": "Drehung", "zh": "旋转",
        "fr": "rotation", "fi": "pyörähdys",
        "criteria": lambda a: abs(a["outcome"]["yaw_net_rad"]) > 3.0,
        "description": "|yaw| > 3 radians (nearly a full turn)"
    },
    "walk_and_spin": {
        "en": "walk and spin", "de": "Gehen mit Drehung", "zh": "边走边转",
        "fr": "marche avec rotation", "fi": "kävely ja pyöriminen",
        "criteria": lambda a: (abs(a["outcome"]["dx"]) + abs(a["outcome"]["dy"])) > 3 and abs(a["outcome"]["yaw_net_rad"]) > 1.5,
        "description": "displacement > 3m AND |yaw| > 1.5 rad"
    },
    "stumble": {
        "en": "stumble", "de": "Stolpern", "zh": "蹒跚",
        "fr": "trébucher", "fi": "kompastua",
        "criteria": lambda a: a["outcome"]["speed_cv"] > 1.0 or a["contact"]["contact_entropy_bits"] > 1.5,
        "description": "high speed CV > 1.0 or high contact entropy > 1.5"
    },
    "limp": {
        "en": "limp", "de": "Hinken", "zh": "跛行",
        "fr": "boiter", "fi": "ontuminen",
        "criteria": lambda a: abs(a["coordination"]["delta_phi_rad"]) > 0.5 and a["coordination"]["phase_lock_score"] < 0.7,
        "description": "asymmetric phase (|delta_phi| > 0.5, phase_lock < 0.7)"
    },
    "glide": {
        "en": "glide", "de": "Gleiten", "zh": "滑行",
        "fr": "glisser", "fi": "liukuminen",
        "criteria": lambda a: a["outcome"]["distance_per_work"] > 0.002 and a["outcome"]["speed_cv"] < 0.8,
        "description": "high efficiency > 0.002 and low speed CV < 0.8"
    },
    "bounce": {
        "en": "bounce", "de": "Hüpfen", "zh": "弹跳",
        "fr": "rebondir", "fi": "pomppiminen",
        "criteria": lambda a: abs(a["outcome"]["dx"]) < 5 and a["outcome"]["work_proxy"] > 2000,
        "description": "low displacement < 5m but high work > 2000 (oscillatory)"
    },
    "shuffle": {
        "en": "shuffle", "de": "Schlurfen", "zh": "曳步",
        "fr": "traîner les pieds", "fi": "laahustaminen",
        "criteria": lambda a: abs(a["outcome"]["dx"]) < 3 and a["outcome"]["mean_speed"] < 1.0,
        "description": "very low displacement < 3m and low speed < 1.0"
    },
    "lurch": {
        "en": "lurch", "de": "Schlingern", "zh": "突然前冲",
        "fr": "embardée", "fi": "heittäytyminen",
        "criteria": lambda a: a["outcome"]["speed_cv"] > 1.2 and abs(a["outcome"]["dx"]) > 2,
        "description": "high speed CV > 1.2 with some displacement > 2m"
    },
    "stand_still": {
        "en": "stand still", "de": "Stillstand", "zh": "静止",
        "fr": "immobile", "fi": "paikallaan seisominen",
        "criteria": lambda a: abs(a["outcome"]["dx"]) < 1 and abs(a["outcome"]["dy"]) < 1,
        "description": "|DX| < 1 and |DY| < 1"
    },
}

# Additional English motion words (synonyms/variations) to reach 100
EXTRA_ENGLISH = [
    "march", "stride", "trot", "gallop", "crawl", "creep", "slide",
    "sway", "rock", "wobble", "drift", "dash", "rush", "plod",
    "stagger", "waddle", "scurry", "amble", "saunter", "sprint",
    "tiptoe", "stomp", "drag", "skid", "prance", "hop", "skip",
    "lunge", "zigzag", "circle", "weave", "twist", "roll", "fall",
    "charge", "retreat", "patrol", "wander", "roam", "prowl",
]

# ── Build the full seed list ─────────────────────────────────────────────────

def build_seeds():
    """Build list of 100 seeds: 12 concepts × 5 langs + 40 English extras."""
    seeds = []
    lang_keys = ["en", "de", "zh", "fr", "fi"]

    for concept_id, concept in CORE_CONCEPTS.items():
        for lang in lang_keys:
            seeds.append({
                "seed": concept[lang],
                "concept": concept_id,
                "language": lang,
                "criteria_fn": concept["criteria"],
                "criteria_desc": concept["description"],
            })

    # Extra English words — scored loosely (any non-trivial motion counts)
    for word in EXTRA_ENGLISH:
        seeds.append({
            "seed": word,
            "concept": word,
            "language": "en",
            "criteria_fn": lambda a: True,  # no specific behavioral criterion
            "criteria_desc": "exploratory — no specific criterion",
        })

    return seeds[:100]


# ── LLM query functions ─────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are designing a neural controller for a 3-link robot (Torso, BackLeg, FrontLeg) "
    "with two joints. The controller has six weights [w03, w13, w23, w04, w14, w24] "
    "mapping touch sensors to motors. Generate a weight vector that makes the robot "
    "move in a way that captures the essence of the word '{seed}'. "
    "Output ONLY a JSON object with keys w03, w04, w13, w14, w23, w24, "
    "each a float in [-1, 1]. No explanation."
)


REASONING_MODELS = {"deepseek-r1:8b", "gpt-oss:20b"}  # models that use hidden <think> tokens

def query_ollama(model, prompt, temperature=0.8, max_tokens=200, timeout=120):
    """Query a local Ollama model."""
    # Reasoning models need more tokens because hidden <think> tokens count against num_predict
    effective_max = 1000 if model in REASONING_MODELS else max_tokens
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
        weights = parse_weights(resp)
        return weights, resp
    except Exception as e:
        return None, str(e)


def query_openai(model, prompt, temperature=0.7, max_tokens=200):
    """Query OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        resp = r.choices[0].message.content
        weights = parse_weights(resp)
        return weights, resp
    except Exception as e:
        return None, str(e)


# ── Models ───────────────────────────────────────────────────────────────────

MODELS = [
    {"name": "qwen3-coder:30b", "type": "ollama"},
    {"name": "deepseek-r1:8b", "type": "ollama"},
    {"name": "llama3.1:latest", "type": "ollama"},
    {"name": "gpt-oss:20b", "type": "ollama"},
    {"name": "gpt-4.1-mini", "type": "openai"},
]


def query_model(model_info, prompt):
    """Dispatch to the appropriate LLM backend."""
    if model_info["type"] == "ollama":
        return query_ollama(model_info["name"], prompt)
    elif model_info["type"] == "openai":
        return query_openai(model_info["name"], prompt)
    return None, "unknown model type"


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    seeds = build_seeds()
    results = []
    n_total = len(seeds) * len(MODELS)
    print(f"Motion Seed Experiment: {len(seeds)} seeds × {len(MODELS)} models = {n_total} trials")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print()

    start_time = time.time()
    trial_num = 0

    for seed_info in seeds:
        seed_word = seed_info["seed"]
        concept = seed_info["concept"]
        lang = seed_info["language"]
        prompt = PROMPT_TEMPLATE.format(seed=seed_word)

        for model_info in MODELS:
            trial_num += 1
            model_name = model_info["name"]
            print(f"[{trial_num}/{n_total}] {model_name} | {lang}:{seed_word}", end=" ", flush=True)

            # Query LLM
            weights, raw_resp = query_model(model_info, prompt)

            if weights is None:
                print("-> PARSE FAIL")
                results.append({
                    "seed": seed_word, "concept": concept, "language": lang,
                    "model": model_name, "success": False,
                    "raw_response": raw_resp[:500],
                    "weights": None, "analytics": None, "match": None,
                })
                continue

            # Simulate
            try:
                analytics = run_trial_inmemory(weights)
            except Exception as e:
                print(f"-> SIM ERROR: {e}")
                results.append({
                    "seed": seed_word, "concept": concept, "language": lang,
                    "model": model_name, "success": False,
                    "raw_response": raw_resp[:500],
                    "weights": weights, "analytics": None, "match": None,
                })
                continue

            # Score semantic match
            try:
                match = seed_info["criteria_fn"](analytics)
            except Exception:
                match = None

            dx = analytics["outcome"]["dx"]
            dy = analytics["outcome"]["dy"]
            yaw = analytics["outcome"]["yaw_net_rad"]
            print(f"-> DX={dx:+.2f} DY={dy:+.2f} YAW={yaw:+.1f} match={match}")

            results.append({
                "seed": seed_word, "concept": concept, "language": lang,
                "model": model_name, "success": True,
                "weights": weights,
                "analytics": analytics,
                "match": match,
            })

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/max(trial_num,1):.2f}s/trial)")

    # ── Summary ──────────────────────────────────────────────────────────────

    # Per-model stats
    print("\n" + "="*70)
    print("PER-MODEL SUMMARY")
    print("="*70)
    for model_info in MODELS:
        mname = model_info["name"]
        m_results = [r for r in results if r["model"] == mname]
        successes = [r for r in m_results if r["success"]]
        matches = [r for r in successes if r["match"] is True]
        core_results = [r for r in successes if r["concept"] in CORE_CONCEPTS and r["match"] is not None]
        core_matches = [r for r in core_results if r["match"]]

        print(f"\n{mname}:")
        print(f"  Parse success: {len(successes)}/{len(m_results)}")
        if core_results:
            print(f"  Semantic match (core 12 concepts): {len(core_matches)}/{len(core_results)} "
                  f"({100*len(core_matches)/len(core_results):.0f}%)")

        # DX distribution
        dxs = [r["analytics"]["outcome"]["dx"] for r in successes if r["analytics"]]
        if dxs:
            print(f"  DX: median={np.median(dxs):+.2f}, max |DX|={max(abs(d) for d in dxs):.2f}")

    # Per-concept match rates across all models
    print("\n" + "="*70)
    print("PER-CONCEPT MATCH RATES (across all models)")
    print("="*70)
    for concept_id in CORE_CONCEPTS:
        c_results = [r for r in results if r["concept"] == concept_id and r["success"] and r["match"] is not None]
        c_matches = [r for r in c_results if r["match"]]
        if c_results:
            print(f"  {concept_id:20s}: {len(c_matches):2d}/{len(c_results):2d} "
                  f"({100*len(c_matches)/len(c_results):5.1f}%) — {CORE_CONCEPTS[concept_id]['description']}")

    # Per-concept × per-model match grid
    print("\n" + "="*70)
    print("MATCH GRID: concept (rows) × model (columns)")
    print("="*70)
    model_names = [m["name"] for m in MODELS]
    header = f"{'concept':20s} | " + " | ".join(f"{m[:12]:>12s}" for m in model_names)
    print(header)
    print("-" * len(header))
    for concept_id in CORE_CONCEPTS:
        row = f"{concept_id:20s} | "
        for mname in model_names:
            cm = [r for r in results if r["concept"] == concept_id and r["model"] == mname
                  and r["success"] and r["match"] is not None]
            cm_match = [r for r in cm if r["match"]]
            if cm:
                row += f"{len(cm_match):>2d}/{len(cm):>2d} ({100*len(cm_match)/len(cm):3.0f}%) | "
            else:
                row += f"{'---':>12s} | "
        print(row)

    # Save full results
    out_path = PROJECT / "artifacts" / "motion_seed_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_seeds": len(seeds),
                "n_models": len(MODELS),
                "models": [m["name"] for m in MODELS],
                "elapsed_seconds": elapsed,
            },
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
