#!/usr/bin/env python3
"""
character_seed_experiment.py

Translate 2000 fictional character names into neural network weights via LLMs,
then simulate each and compute Beer-framework analytics.

Same pipeline as the Motion Gait Dictionary but with character identities as seeds
instead of motion verbs. No foreign language translations (English names only).

Pipeline per trial:
  1. Prompt LLM: "Generate weights that capture the essence of [Character/Story]"
  2. Parse 6 weights from LLM response
  3. Run headless PyBullet simulation (4000 steps @ 240 Hz)
  4. Compute Beer-framework analytics
  5. Record everything

Scale: 2000 characters × 4 local Ollama models = 8000 trials
Estimated time: ~4-5 hours (good overnight run)
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import (
    parse_weights, run_trial_inmemory,
    WEIGHT_NAMES, OLLAMA_URL
)
from compute_beer_analytics import NumpyEncoder

# ── Prompt template ──────────────────────────────────────────────────────────
# Same robot description and weight semantics as motion_seed_experiment_v2,
# but the seed is a fictional character name instead of a motion word.

PROMPT_TEMPLATE = (
    "You are designing a neural controller for a 3-link walking robot "
    "(Torso, BackLeg, FrontLeg) with two hinge joints.\n\n"
    "Sensors: s0=torso_contact, s1=back_foot_contact, s2=front_foot_contact (binary: 0 or 1)\n"
    "Motors: m3=back_joint_angle, m4=front_joint_angle\n"
    "Control law: m3 = tanh(w03*s0 + w13*s1 + w23*s2), m4 = tanh(w04*s0 + w14*s1 + w24*s2)\n"
    "Positive motor value = extend leg forward. Negative = pull leg back.\n\n"
    "Weight roles:\n"
    "  w03, w04: torso touch → motors (balance response when body tilts)\n"
    "  w13, w24: foot touch → same leg motor (local reflex, stride timing)\n"
    "  w14, w23: foot touch → opposite leg motor (cross-coupling, coordination)\n\n"
    "Verified examples (character → weights → measured outcome):\n"
    "  calm stoic character → {{\"w03\":0.1, \"w04\":-0.1, \"w13\":0.4, \"w14\":0.4, \"w23\":0.4, \"w24\":0.4}}\n"
    "    measured: DX=+3.2m, steady march, phase_lock=0.85\n"
    "  chaotic trickster → {{\"w03\":0.7, \"w04\":-0.7, \"w13\":-0.4, \"w14\":1.0, \"w23\":0.7, \"w24\":-0.4}}\n"
    "    measured: DX=-2.1m, DY=+3.8m, erratic stagger, high contact entropy\n"
    "  frozen with fear → {{\"w03\":0, \"w04\":0, \"w13\":0, \"w14\":0, \"w23\":0, \"w24\":0}}\n"
    "    measured: DX=0.0m, perfectly still\n\n"
    "Now design weights that capture the essence of how '{character}' "
    "from '{story}' would move. Think about their personality, energy level, "
    "and movement style.\n\n"
    "Choose each weight from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].\n"
    "In 1-2 sentences, note the movement quality. Then output ONLY the JSON object "
    "with keys w03, w04, w13, w14, w23, w24. Keep reasoning SHORT."
)

# ── Weight discretization ────────────────────────────────────────────────────

WEIGHT_GRID = [-1.0, -0.7, -0.4, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0]

def snap_to_grid(weights):
    """Snap each weight to nearest value in WEIGHT_GRID."""
    snapped = {}
    for k, v in weights.items():
        snapped[k] = min(WEIGHT_GRID, key=lambda g: abs(g - v))
    return snapped

# ── LLM query ────────────────────────────────────────────────────────────────

REASONING_MODELS = {"deepseek-r1:8b", "gpt-oss:20b"}

MODELS = [
    {"name": "qwen3-coder:30b", "type": "ollama"},
    {"name": "deepseek-r1:8b", "type": "ollama"},
    {"name": "llama3.1:latest", "type": "ollama"},
    {"name": "gpt-oss:20b", "type": "ollama"},
]

def query_ollama(model, prompt, temperature=0.8, max_tokens=500, timeout=120):
    """Query a local Ollama model."""
    effective_max = 2000 if model in REASONING_MODELS else max_tokens
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
        if weights is not None:
            weights = snap_to_grid(weights)
        return weights, resp
    except Exception as e:
        return None, str(e)

# ── Load characters ──────────────────────────────────────────────────────────

def load_characters():
    """Load the 2000 fictional characters from the archetypometrics TSV."""
    tsv_path = PROJECT / "artifacts" / "archetypometrics_characters.tsv"
    characters = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Parse "Character/Story" into separate fields
            char_story = row.get("character/story", "")
            parts = char_story.split("/", 1)
            character = parts[0].strip() if parts else row.get("character", "")
            story = parts[1].strip() if len(parts) > 1 else "Unknown"
            characters.append({
                "index": int(row.get("index", 0)),
                "character": character,
                "story": story,
                "character_story": char_story,
                "card_url": row.get("card url", ""),
            })
    return characters

# ── Checkpoint support ───────────────────────────────────────────────────────
# With 8000 trials over ~5 hours, we need to save progress periodically
# so the run can be resumed if interrupted.

def load_checkpoint(checkpoint_path):
    """Load existing results from checkpoint file."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        return data.get("results", []), data.get("completed_keys", set())
    return [], set()

def save_checkpoint(checkpoint_path, results, metadata):
    """Save results to checkpoint file."""
    completed_keys = {f"{r['character_story']}|{r['model']}" for r in results}
    with open(checkpoint_path, "w") as f:
        json.dump({
            "metadata": metadata,
            "completed_keys": list(completed_keys),
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)

# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    characters = load_characters()
    print(f"Loaded {len(characters)} characters from archetypometrics TSV")

    out_path = PROJECT / "artifacts" / "character_seed_experiment.json"
    checkpoint_path = PROJECT / "artifacts" / "character_seed_experiment_checkpoint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if exists
    results = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        completed_keys = set(ckpt.get("completed_keys", []))
        print(f"Resumed from checkpoint: {len(results)} trials already done")

    n_total = len(characters) * len(MODELS)
    n_remaining = n_total - len(completed_keys)
    print(f"Character Seed Experiment: {len(characters)} characters × {len(MODELS)} models = {n_total} trials")
    print(f"Remaining: {n_remaining} trials")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print()

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "character_seed",
        "prompt_version": "character essence (weight semantics + few-shot + personality)",
        "n_characters": len(characters),
        "n_models": len(MODELS),
        "models": [m["name"] for m in MODELS],
        "source": "artifacts/archetypometrics_characters.tsv (2000 characters from 341 stories)",
    }

    start_time = time.time()
    trial_num = len(completed_keys)
    failures = 0
    checkpoint_interval = 50  # save every 50 trials

    for char_info in characters:
        character = char_info["character"]
        story = char_info["story"]
        char_story = char_info["character_story"]

        prompt = PROMPT_TEMPLATE.format(character=character, story=story)

        for model_info in MODELS:
            model_name = model_info["name"]
            key = f"{char_story}|{model_name}"

            # Skip if already done
            if key in completed_keys:
                continue

            trial_num += 1
            print(f"[{trial_num}/{n_total}] {model_name} | {character} ({story})", end=" ", flush=True)

            # Query LLM
            weights, raw_resp = query_ollama(model_name, prompt)

            if weights is None:
                failures += 1
                print("-> PARSE FAIL")
                results.append({
                    "character": character,
                    "story": story,
                    "character_story": char_story,
                    "character_index": char_info["index"],
                    "card_url": char_info["card_url"],
                    "model": model_name,
                    "success": False,
                    "raw_response": raw_resp[:500] if raw_resp else "",
                    "weights": None,
                    "analytics": None,
                })
                completed_keys.add(key)
                continue

            # Simulate
            try:
                analytics = run_trial_inmemory(weights)
            except Exception as e:
                failures += 1
                print(f"-> SIM ERROR: {e}")
                results.append({
                    "character": character,
                    "story": story,
                    "character_story": char_story,
                    "character_index": char_info["index"],
                    "card_url": char_info["card_url"],
                    "model": model_name,
                    "success": False,
                    "raw_response": raw_resp[:500] if raw_resp else "",
                    "weights": weights,
                    "analytics": None,
                })
                completed_keys.add(key)
                continue

            dx = analytics["outcome"]["dx"]
            dy = analytics["outcome"]["dy"]
            speed = analytics["outcome"]["mean_speed"]
            print(f"-> DX={dx:+.2f} DY={dy:+.2f} spd={speed:.2f}")

            results.append({
                "character": character,
                "story": story,
                "character_story": char_story,
                "character_index": char_info["index"],
                "card_url": char_info["card_url"],
                "model": model_name,
                "success": True,
                "weights": weights,
                "analytics": analytics,
            })
            completed_keys.add(key)

            # Periodic checkpoint
            if len(completed_keys) % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, results, metadata)
                elapsed = time.time() - start_time
                rate = (trial_num - (n_total - n_remaining)) / max(elapsed, 1)
                remaining_time = (n_total - len(completed_keys)) / max(rate, 0.01)
                print(f"  [checkpoint] {len(completed_keys)}/{n_total} done, "
                      f"{elapsed:.0f}s elapsed, ~{remaining_time/60:.0f}min remaining")

    total_elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CHARACTER SEED EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f} hours)")
    print(f"Trials: {len(results)} ({failures} failures)")

    successes = [r for r in results if r["success"]]
    if successes:
        dxs = [abs(r["analytics"]["outcome"]["dx"]) for r in successes]
        dead = sum(1 for d in dxs if d < 1.0)
        print(f"\nOverall:")
        print(f"  Dead (|DX|<1m): {dead}/{len(successes)} ({100*dead/len(successes):.1f}%)")
        print(f"  Median |DX|: {np.median(dxs):.2f}m")
        print(f"  Max |DX|: {max(dxs):.2f}m")

        # Per-model summary
        print(f"\nPer-model:")
        for model_info in MODELS:
            mname = model_info["name"]
            m_results = [r for r in successes if r["model"] == mname]
            if m_results:
                m_dxs = [abs(r["analytics"]["outcome"]["dx"]) for r in m_results]
                m_dead = sum(1 for d in m_dxs if d < 1.0)
                print(f"  {mname:20s}: {len(m_results):4d} trials, "
                      f"dead={m_dead} ({100*m_dead/len(m_results):.0f}%), "
                      f"median |DX|={np.median(m_dxs):.2f}m")

        # Per-story summary (top 10 most interesting stories by median displacement)
        from collections import defaultdict
        story_dxs = defaultdict(list)
        for r in successes:
            story_dxs[r["story"]].append(abs(r["analytics"]["outcome"]["dx"]))

        print(f"\nTop 10 stories by median |DX|:")
        sorted_stories = sorted(story_dxs.items(),
                                key=lambda x: np.median(x[1]), reverse=True)
        for story, dxs_list in sorted_stories[:10]:
            print(f"  {story:40s}: median |DX|={np.median(dxs_list):.2f}m "
                  f"({len(dxs_list)} characters)")

        # Most mobile characters
        print(f"\nTop 10 most mobile characters (by max |DX| across models):")
        char_best = defaultdict(lambda: 0)
        char_info_map = {}
        for r in successes:
            dx_abs = abs(r["analytics"]["outcome"]["dx"])
            if dx_abs > char_best[r["character_story"]]:
                char_best[r["character_story"]] = dx_abs
                char_info_map[r["character_story"]] = r
        sorted_chars = sorted(char_best.items(), key=lambda x: x[1], reverse=True)
        for char_story, best_dx in sorted_chars[:10]:
            r = char_info_map[char_story]
            print(f"  {char_story:50s}: |DX|={best_dx:.2f}m ({r['model']})")

        # Most frozen characters
        print(f"\nTop 10 most frozen characters (lowest max |DX|):")
        for char_story, best_dx in sorted_chars[-10:]:
            r = char_info_map[char_story]
            print(f"  {char_story:50s}: |DX|={best_dx:.2f}m ({r['model']})")

    # Save final results
    metadata["elapsed_seconds"] = total_elapsed
    metadata["n_results"] = len(results)
    metadata["n_failures"] = failures

    with open(out_path, "w") as f:
        json.dump({
            "metadata": metadata,
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint removed")


if __name__ == "__main__":
    run_experiment()
