#!/usr/bin/env python3
"""
structured_random_celebrities_multimodel.py

Multi-LLM Celebrity Experiment
==============================

HYPOTHESIS
----------
Different LLMs encode different implicit theories of personhood. When asked
to translate a celebrity name into 6 synapse weights, each model's training
data, architecture, and tokenization scheme should produce a different mapping
from Name → Weight Vector → Robot Gait. This experiment runs the same 132
celebrity names through all 4 locally available LLMs to answer:

KEY QUESTIONS
  1. Do different LLMs produce different sign-pattern distributions?
     (i.e., different archetypal categorizations of the same people)
  2. Do they agree on who should walk vs who should be inert?
  3. Do they land in different cliffiness zones?
  4. Which model produces the most unique weight vectors (pre-perturbation)?
  5. Is there a "consensus archetype" — celebrities that ALL models agree on?
  6. How does model size (8B vs 20B vs 30B) affect the mapping?

MODELS
------
  qwen3-coder:30b  — 30B parameter code-specialized model (baseline, used in prior runs)
  gpt-oss:20b      — 20B parameter general-purpose model
  deepseek-r1:8b   — 8B parameter reasoning model (chain-of-thought)
  llama3.1:latest  — 8B parameter general-purpose model (Meta)

Each model gets the same 132 prompts with the same temperature (1.5) and
the same deterministic perturbation (±0.05 via SHA-256 hash). The perturbation
seed includes the model name so that the same celebrity gets a different nudge
from each model, preventing artificial convergence.

Usage:
    python3 structured_random_celebrities_multimodel.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import structured_random_common as src
src.NUM_TRIALS = 200  # Allow all 132 seeds

from structured_random_common import (
    run_structured_search, WEIGHT_NAMES, generate_weights, run_trial_inmemory,
    write_brain
)

# Import celebrity seed lists from the original script
from structured_random_celebrities import GROUPS, SEEDS

OUT_DIR = PROJECT / "artifacts"

MODELS = [
    "qwen3-coder:30b",
    "gpt-oss:20b",
    "deepseek-r1:8b",
    "llama3.1:latest",
]

TEMPERATURE = 1.5
PERTURB_RADIUS = 0.05


def perturb_weights(weights, seed):
    """Deterministic perturbation. Seed includes model name for cross-model uniqueness."""
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(h)
    perturbed = {}
    for wn in WEIGHT_NAMES:
        delta = rng.uniform(-PERTURB_RADIUS, PERTURB_RADIUS)
        perturbed[wn] = max(-1.0, min(1.0, weights[wn] + delta))
    return perturbed


def make_prompt(seed):
    """Build the LLM prompt for a celebrity seed (model-agnostic)."""
    name = seed.split(" [")[0]
    return (
        f"Generate 6 synapse weights for a 3-link walking robot that embodies "
        f"the public figure {name}. "
        f"The 6 weights are: w03, w04, w13, w14, w23, w24. "
        f"Each is a float in [-1, 1] with exactly 3 decimal places. "
        f"Think about what makes {name} DISTINCT: their energy, aggression, "
        f"grace, authority, and movement style. Politicians and athletes should "
        f"differ. Entertainers and intellectuals should differ. Brash and calm should differ. "
        f"Return ONLY a JSON object: "
        f'{{"w03": <float>, "w04": <float>, "w13": <float>, '
        f'"w14": <float>, "w23": <float>, "w24": <float>}}'
    )


def run_model(model_name):
    """Run all 132 celebrities through one model."""
    # Override the global model
    src.OLLAMA_MODEL = model_name

    # Seeds include model name in the perturbation seed for cross-model uniqueness
    def model_perturb(weights, seed):
        return perturb_weights(weights, f"{model_name}:{seed}")

    out_json = OUT_DIR / f"multimodel_celebrities_{model_name.replace(':', '_').replace('.', '_')}.json"

    print(f"\n{'#'*70}")
    print(f"# MODEL: {model_name}")
    print(f"{'#'*70}")

    run_structured_search(
        f"celebrities_{model_name}",
        SEEDS,
        make_prompt,
        out_json,
        temperature=TEMPERATURE,
        weight_transform=model_perturb,
    )
    return out_json


def main():
    print(f"Multi-LLM Celebrity Experiment: {len(SEEDS)} celebrities × {len(MODELS)} models")
    print(f"Temperature: {TEMPERATURE}, Perturbation: ±{PERTURB_RADIUS}")
    print(f"Models: {', '.join(MODELS)}")

    t0 = time.perf_counter()
    result_files = {}
    for model in MODELS:
        out = run_model(model)
        result_files[model] = out

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"ALL MODELS COMPLETE — {elapsed:.0f}s total")
    print(f"{'='*70}")
    for model, path in result_files.items():
        print(f"  {model}: {path}")


if __name__ == "__main__":
    main()
