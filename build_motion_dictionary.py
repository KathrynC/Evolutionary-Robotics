#!/usr/bin/env python3
"""
build_motion_dictionary.py

Build a comprehensive motion-to-gait dictionary from experiment results.
Includes ALL viable synonyms (every trial that matched its concept criteria),
not just the best exemplar. Each entry cites the descriptive word(s),
the model, and the language.

Reads from:
  - artifacts/motion_seed_experiment.json   (v1: 12 concepts, 500 trials)
  - artifacts/motion_seed_experiment_v2.json (v2: 28 concepts, 700 trials)

Writes to:
  - artifacts/motion_gait_dictionary_v2.json
"""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

ARTIFACTS = PROJECT / "artifacts"


def load_matches(path):
    """Load all successful, matching trials from an experiment JSON."""
    if not path.exists():
        print(f"  {path.name}: not found, skipping")
        return []
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    matches = []
    for r in results:
        if r.get("success") and r.get("match") is True and r.get("weights"):
            matches.append(r)
    print(f"  {path.name}: {len(matches)} matches from {len(results)} trials")
    return matches


def build_dictionary():
    print("Loading experiment results...")
    v1_matches = load_matches(ARTIFACTS / "motion_seed_experiment.json")
    v2_matches = load_matches(ARTIFACTS / "motion_seed_experiment_v2.json")

    all_matches = v1_matches + v2_matches
    print(f"\nTotal matches: {len(all_matches)}")

    # Group by concept
    concepts = {}
    for m in all_matches:
        concept = m["concept"]
        if concept not in concepts:
            concepts[concept] = []
        concepts[concept].append(m)

    # Build dictionary
    dictionary = {}
    for concept_id in sorted(concepts.keys()):
        entries = concepts[concept_id]

        # Deduplicate: same weights from same model/seed count once
        seen = set()
        unique_entries = []
        for e in entries:
            w = e["weights"]
            key = (e["seed"], e["model"], tuple(sorted(w.items())))
            if key not in seen:
                seen.add(key)
                unique_entries.append(e)

        # Build the flat list of all viable synonyms
        synonyms = []
        for e in unique_entries:
            w = e["weights"]
            a = e.get("analytics", {})
            outcome = a.get("outcome", {})
            contact = a.get("contact", {})
            coord = a.get("coordination", {})

            synonyms.append({
                "word": e["seed"],
                "model": e["model"],
                "language": e["language"],
                "weights": {
                    "w03": w.get("w03", 0),
                    "w04": w.get("w04", 0),
                    "w13": w.get("w13", 0),
                    "w14": w.get("w14", 0),
                    "w23": w.get("w23", 0),
                    "w24": w.get("w24", 0),
                },
                "behavior": {
                    "dx": round(outcome.get("dx", 0), 2),
                    "dy": round(outcome.get("dy", 0), 2),
                    "yaw_rad": round(outcome.get("yaw_net_rad", 0), 2),
                    "mean_speed": round(outcome.get("mean_speed", 0), 3),
                    "speed_cv": round(outcome.get("speed_cv", 0), 3),
                    "work": round(outcome.get("work_proxy", 0), 1),
                    "efficiency": round(outcome.get("distance_per_work", 0), 6),
                    "phase_lock": round(coord.get("phase_lock_score", 0), 3),
                    "contact_entropy": round(contact.get("contact_entropy_bits", 0), 3),
                    "torso_duty": round(contact.get("duty_torso", 0), 3),
                },
            })

        # Sort: by model name, then language, then word
        synonyms.sort(key=lambda s: (s["model"], s["language"], s["word"]))

        # Collect coverage stats
        models_seen = sorted(set(s["model"] for s in synonyms))
        langs_seen = sorted(set(s["language"] for s in synonyms))
        words_seen = sorted(set(s["word"] for s in synonyms))

        dictionary[concept_id] = {
            "n_matches": len(synonyms),
            "models": models_seen,
            "languages": langs_seen,
            "words": words_seen,
            "synonyms": synonyms,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"MOTION GAIT DICTIONARY: {len(dictionary)} concepts")
    print(f"{'='*60}")
    for concept_id, entry in sorted(dictionary.items(), key=lambda x: -x[1]["n_matches"]):
        n = entry["n_matches"]
        m = len(entry["models"])
        l = len(entry["languages"])
        w = len(entry["words"])
        print(f"  {concept_id:25s}: {n:3d} matches, {w:2d} words, {m} models, {l} languages")

    total_entries = sum(e["n_matches"] for e in dictionary.values())
    print(f"\n  Total entries: {total_entries}")

    # Write
    out = {
        "metadata": {
            "generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "sources": [
                "motion_seed_experiment.json (v1: 12 concepts, 500 trials)",
                "motion_seed_experiment_v2.json (v2: 28 concepts, 700 trials)",
            ],
            "description": (
                "Comprehensive motion-to-gait dictionary. Maps semantic motion concepts "
                "to neural network weight vectors for a 3-link PyBullet robot. "
                "Includes ALL viable synonyms — every trial that matched its concept criteria. "
                "Each entry cites the descriptive word, the LLM model, and the language."
            ),
            "robot": "3-link (Torso + BackLeg + FrontLeg), 2 hinge joints, 3 touch sensors, 2 motors",
            "weight_keys": {
                "w03": "torso touch → back leg motor (balance feedback)",
                "w04": "torso touch → front leg motor (balance feedback)",
                "w13": "back leg touch → back leg motor (local reflex)",
                "w14": "back leg touch → front leg motor (cross-coupling)",
                "w23": "front leg touch → back leg motor (cross-coupling)",
                "w24": "front leg touch → front leg motor (local reflex)",
            },
            "simulation": "4000 steps @ 240 Hz, MAX_FORCE=150N, headless PyBullet DIRECT mode",
            "n_concepts": len(dictionary),
            "n_total_entries": total_entries,
        },
        "concepts": dictionary,
    }

    out_path = ARTIFACTS / "motion_gait_dictionary_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    build_dictionary()
