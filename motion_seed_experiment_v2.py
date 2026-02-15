#!/usr/bin/env python3
"""
motion_seed_experiment_v2.py

25 new motion concepts with improved prompting:
  - Weight semantics (what each weight physically does)
  - Few-shot examples from v1 verified matches
  - Behavioral criteria in the prompt
  - 5 languages per concept (en/de/zh/fr/fi)

Pipeline per trial:
  1. Prompt LLM with enriched context + motion word
  2. Parse 6 weights from LLM response
  3. Run headless PyBullet simulation (4000 steps @ 240 Hz)
  4. Compute Beer-framework analytics
  5. Score: does the behavior match the semantic intent?
"""

import json
import math
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

# ── Improved prompt template ─────────────────────────────────────────────────
# Synthesized from DeepSeek + GPT-4.1-mini + GPT-5.2 consultations:
# - Control law equation so LLM can reason about the math (GPT-5.2)
# - Functional labels on weights (DeepSeek)
# - 3 contrasting few-shot with measured outcomes (GPT-5.2: "seed → metrics → weights")
# - Qualitative feature card per concept (DeepSeek + GPT-5.2)
# - Structured brief CoT before JSON (DeepSeek)
# - Discretization suggestion to fight collapse (GPT-5.2)
# - Anti-symmetric hint for directional control (DeepSeek)

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
    "Design priors:\n"
    "  Coordinated gait: w13 ≈ w24 (symmetric reflexes), w14 ≈ w23 (symmetric coupling)\n"
    "  Directional bias: w03 and w04 with opposite signs\n"
    "  Rotation: strong cross-coupling dominance (w14, w23 >> w13, w24)\n"
    "  Stillness: all weights near zero\n\n"
    "Verified examples (word → weights → measured outcome):\n"
    "  \"stand still\" → {{\"w03\":0, \"w04\":0, \"w13\":0, \"w14\":0, \"w23\":0, \"w24\":0}}\n"
    "    measured: DX=0.0m, DY=0.0m, speed=0.0 (perfectly still)\n"
    "  \"stumble\" → {{\"w03\":0.8, \"w04\":-0.6, \"w13\":-0.3, \"w14\":0.9, \"w23\":0.5, \"w24\":-0.7}}\n"
    "    measured: DX=-2.3m, DY=+2.6m, speed=1.2, irregular and erratic\n"
    "  \"spin\" → {{\"w03\":-0.7, \"w04\":0.6, \"w13\":0.8, \"w14\":0.3, \"w23\":-0.4, \"w24\":0.9}}\n"
    "    measured: DX=-1.5m, DY=-0.7m, yaw=+4.2 radians, rotates in place\n\n"
    "For '{seed}': {criteria_hint}\n\n"
    "Choose each weight from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].\n"
    "In 1-2 sentences, note the motor pattern. Then output ONLY the JSON object "
    "with keys w03, w04, w13, w14, w23, w24. Keep reasoning SHORT."
)

# ── 25 new motion concepts ──────────────────────────────────────────────────

def _disp(a):
    """Total displacement from origin."""
    return math.sqrt(a["outcome"]["dx"]**2 + a["outcome"]["dy"]**2)

CORE_CONCEPTS = {
    "gallop": {
        "en": "gallop", "de": "Galopp", "zh": "飞奔",
        "fr": "galop", "fi": "laukka",
        "criteria": lambda a: a["outcome"]["mean_speed"] > 1.5 and a["coordination"]["phase_lock_score"] > 0.6,
        "description": "fast and rhythmic (speed > 1.5, phase_lock > 0.6)",
        "hint": "fast rhythmic locomotion, legs moving in coordinated alternating pattern"
    },
    "tiptoe": {
        "en": "tiptoe", "de": "Zehenspitzen", "zh": "踮脚走",
        "fr": "marcher sur la pointe des pieds", "fi": "varpaillaan kävely",
        "criteria": lambda a: a["outcome"]["mean_speed"] < 0.3 and _disp(a) > 0.5,
        "description": "very slow but moving (speed < 0.3, displacement > 0.5m)",
        "hint": "extremely slow cautious movement, barely moving but not still"
    },
    "zigzag": {
        "en": "zigzag", "de": "Zickzack", "zh": "之字形行走",
        "fr": "zigzag", "fi": "siksak",
        "criteria": lambda a: abs(a["outcome"]["dy"]) > 2 and a["outcome"]["speed_cv"] > 0.8,
        "description": "|DY| > 2 and speed_cv > 0.8 (erratic lateral motion)",
        "hint": "erratic path weaving side to side, significant lateral deviation"
    },
    "circle": {
        "en": "walk in a circle", "de": "im Kreis gehen", "zh": "走圈",
        "fr": "tourner en rond", "fi": "kävellä ympyrää",
        "criteria": lambda a: _disp(a) < 5 and abs(a["outcome"]["yaw_net_rad"]) > 4,
        "description": "returns near start after rotating (displacement < 5, |yaw| > 4 rad)",
        "hint": "walking in a circular path, continuous turning while moving forward"
    },
    "slide": {
        "en": "slide", "de": "Gleiten", "zh": "滑动",
        "fr": "glisser doucement", "fi": "liukua",
        "criteria": lambda a: a["outcome"]["distance_per_work"] > 0.003 and a["outcome"]["speed_cv"] < 0.5,
        "description": "very efficient (distance_per_work > 0.003, speed_cv < 0.5)",
        "hint": "smooth efficient gliding, minimal wasted energy, consistent speed"
    },
    "freeze": {
        "en": "freeze", "de": "Einfrieren", "zh": "冻住",
        "fr": "geler", "fi": "jäätyä",
        "criteria": lambda a: abs(a["outcome"]["dx"]) < 0.3 and abs(a["outcome"]["dy"]) < 0.3 and a["outcome"]["work_proxy"] < 100,
        "description": "|DX| < 0.3, |DY| < 0.3, work < 100 (absolute stillness)",
        "hint": "absolute stillness, locked in place, no energy expenditure at all"
    },
    "dash": {
        "en": "dash", "de": "Spurt", "zh": "冲刺",
        "fr": "sprint court", "fi": "spurtti",
        "criteria": lambda a: _disp(a) > 10 and a["outcome"]["mean_speed"] > 1.5,
        "description": "far and fast (displacement > 10m, speed > 1.5)",
        "hint": "explosive fast movement, covering a large distance quickly"
    },
    "wobble": {
        "en": "wobble", "de": "Wackeln", "zh": "摇晃",
        "fr": "chanceler", "fi": "huojua",
        "criteria": lambda a: a["contact"]["contact_entropy_bits"] > 1.5 and abs(a["outcome"]["dx"]) < 3,
        "description": "high contact entropy > 1.5 with low displacement < 3m",
        "hint": "unstable oscillation, constantly shifting balance without going anywhere"
    },
    "march": {
        "en": "march", "de": "Marschieren", "zh": "行军",
        "fr": "marcher au pas", "fi": "marssia",
        "criteria": lambda a: a["coordination"]["phase_lock_score"] > 0.7 and a["outcome"]["speed_cv"] < 0.6 and _disp(a) > 2,
        "description": "steady symmetric gait (phase_lock > 0.7, speed_cv < 0.6, displacement > 2m)",
        "hint": "steady regular march, legs alternating symmetrically at constant pace"
    },
    "crawl": {
        "en": "crawl", "de": "Kriechen", "zh": "爬行",
        "fr": "ramper", "fi": "ryömiä",
        "criteria": lambda a: a["outcome"]["mean_speed"] < 0.5 and _disp(a) > 2,
        "description": "very slow but travels (speed < 0.5, displacement > 2m)",
        "hint": "very slow ground-hugging locomotion, patient and deliberate"
    },
    "sprint": {
        "en": "sprint", "de": "Volles Tempo", "zh": "全速冲刺",
        "fr": "sprint", "fi": "pikajuoksu",
        "criteria": lambda a: a["outcome"]["mean_speed"] > 2.0,
        "description": "maximum speed > 2.0",
        "hint": "absolute maximum speed, fastest possible locomotion"
    },
    "sway": {
        "en": "sway", "de": "Schwanken", "zh": "摇摆",
        "fr": "se balancer", "fi": "keinua",
        "criteria": lambda a: abs(a["outcome"]["dy"]) > abs(a["outcome"]["dx"]) and abs(a["outcome"]["dy"]) > 1,
        "description": "|DY| > |DX| and |DY| > 1 (lateral dominant)",
        "hint": "lateral swaying motion, drifting more sideways than forward"
    },
    "drag": {
        "en": "drag", "de": "Schleppen", "zh": "拖行",
        "fr": "traîner", "fi": "raahata",
        "criteria": lambda a: a["outcome"]["mean_speed"] < 0.8 and a["outcome"]["work_proxy"] > 2000,
        "description": "slow with high effort (speed < 0.8, work > 2000)",
        "hint": "slow effortful movement, like dragging a heavy load, lots of energy for little progress"
    },
    "twirl": {
        "en": "twirl", "de": "Wirbeln", "zh": "旋转跳跃",
        "fr": "tourbillonner", "fi": "pyörähtelyä",
        "criteria": lambda a: abs(a["outcome"]["yaw_net_rad"]) > 5,
        "description": "|yaw| > 5 radians (extensive rotation)",
        "hint": "extensive spinning, rotating well past a full revolution"
    },
    "stomp": {
        "en": "stomp", "de": "Stampfen", "zh": "跺脚",
        "fr": "piétiner", "fi": "tömistää",
        "criteria": lambda a: a["outcome"]["work_proxy"] > 3000 and abs(a["outcome"]["dx"]) < 2,
        "description": "high energy (work > 3000) but stays put (|DX| < 2)",
        "hint": "vigorous stomping in place, lots of energy but staying put"
    },
    "drift": {
        "en": "drift", "de": "Treiben", "zh": "漂移",
        "fr": "dériver", "fi": "ajelehtia",
        "criteria": lambda a: a["outcome"]["speed_cv"] < 0.5 and _disp(a) > 3,
        "description": "smooth travel (speed_cv < 0.5, displacement > 3m)",
        "hint": "smooth passive-feeling movement, as if carried by a current"
    },
    "hop": {
        "en": "hop", "de": "Hüpfer", "zh": "单脚跳",
        "fr": "sautiller", "fi": "hyppiä",
        "criteria": lambda a: a["outcome"]["work_proxy"] > 1500 and 2 < _disp(a) < 8,
        "description": "energetic with moderate travel (work > 1500, 2m < displacement < 8m)",
        "hint": "energetic bouncing locomotion, rhythmic vertical energy with some forward travel"
    },
    "scurry": {
        "en": "scurry", "de": "Huschen", "zh": "急匆匆地跑",
        "fr": "trotter", "fi": "vilistää",
        "criteria": lambda a: a["outcome"]["mean_speed"] > 1.0 and _disp(a) < 5,
        "description": "fast but short distance (speed > 1.0, displacement < 5m)",
        "hint": "fast frantic movement, lots of speed but not covering much ground"
    },
    "plod": {
        "en": "plod", "de": "Trotten", "zh": "沉重地走",
        "fr": "marcher lourdement", "fi": "tallustaa",
        "criteria": lambda a: a["outcome"]["speed_cv"] < 0.4 and a["outcome"]["mean_speed"] < 0.8 and _disp(a) > 1,
        "description": "slow and steady (speed_cv < 0.4, speed < 0.8, displacement > 1m)",
        "hint": "slow heavy plodding, very consistent monotonous pace"
    },
    "stagger": {
        "en": "stagger", "de": "Taumeln", "zh": "踉跄",
        "fr": "tituber", "fi": "horjua",
        "criteria": lambda a: a["outcome"]["speed_cv"] > 1.0 and _disp(a) > 5,
        "description": "irregular with far travel (speed_cv > 1.0, displacement > 5m)",
        "hint": "irregular unsteady locomotion, lurching forward despite instability"
    },
    "pivot": {
        "en": "pivot", "de": "Schwenk", "zh": "原地转身",
        "fr": "pivoter", "fi": "kääntyä",
        "criteria": lambda a: abs(a["outcome"]["yaw_net_rad"]) > 2 and _disp(a) < 3,
        "description": "rotate in place (|yaw| > 2 rad, displacement < 3m)",
        "hint": "turning in place, rotating body without traveling forward"
    },
    "charge": {
        "en": "charge", "de": "Angriff", "zh": "猛冲",
        "fr": "charger", "fi": "rynnätä",
        "criteria": lambda a: _disp(a) > 8 and a["outcome"]["mean_speed"] > 1.5 and abs(a["outcome"]["dy"]) < abs(a["outcome"]["dx"]),
        "description": "fast straight line (displacement > 8m, speed > 1.5, |DY| < |DX|)",
        "hint": "fast aggressive straight-line charge, moving far in one direction"
    },
    "retreat": {
        "en": "retreat", "de": "Rückzug", "zh": "后退",
        "fr": "reculer", "fi": "perääntyä",
        "criteria": lambda a: a["outcome"]["dx"] < -3 and abs(a["outcome"]["dy"]) < abs(a["outcome"]["dx"]),
        "description": "backward motion (DX < -3, |DY| < |DX|)",
        "hint": "backward movement, moving away from the forward direction"
    },
    "patrol": {
        "en": "patrol", "de": "Patrouille", "zh": "巡逻",
        "fr": "patrouiller", "fi": "partioida",
        "criteria": lambda a: a["outcome"]["speed_cv"] < 0.6 and 3 < _disp(a) < 15,
        "description": "moderate steady locomotion (speed_cv < 0.6, 3m < displacement < 15m)",
        "hint": "steady moderate-pace patrolling, consistent speed covering moderate distance"
    },
    "rock": {
        "en": "rock back and forth", "de": "Schaukeln", "zh": "前后摇摆",
        "fr": "se bercer", "fi": "keinua edestakaisin",
        "criteria": lambda a: abs(a["outcome"]["dx"]) < 1.5 and abs(a["outcome"]["dy"]) < 1.5 and a["outcome"]["work_proxy"] > 500,
        "description": "oscillatory in place (|DX| < 1.5, |DY| < 1.5, work > 500)",
        "hint": "rhythmic rocking in place, oscillating body without traveling"
    },
    "turn_left": {
        "en": "turn left", "de": "links abbiegen", "zh": "左转",
        "fr": "tourner à gauche", "fi": "kääntyä vasemmalle",
        "criteria": lambda a: a["outcome"]["yaw_net_rad"] < -1.5 and _disp(a) > 1,
        "description": "turn leftward (yaw < -1.5 rad with some displacement)",
        "hint": "veering or turning to the left while moving, curving leftward"
    },
    "turn_right": {
        "en": "turn right", "de": "rechts abbiegen", "zh": "右转",
        "fr": "tourner à droite", "fi": "kääntyä oikealle",
        "criteria": lambda a: a["outcome"]["yaw_net_rad"] > 1.5 and _disp(a) > 1,
        "description": "turn rightward (yaw > +1.5 rad with some displacement)",
        "hint": "veering or turning to the right while moving, curving rightward"
    },
    "headstand": {
        "en": "stand on your head", "de": "Kopfstand", "zh": "倒立",
        "fr": "faire le poirier", "fi": "seisoa päällään",
        "criteria": lambda a: a["contact"]["duty_torso"] > 0.6 and abs(a["outcome"]["dx"]) < 3,
        "description": "inverted (torso_duty > 0.6, robot upside down, |DX| < 3)",
        "hint": "flip upside down so the body is on the ground and legs are in the air"
    },
}


# ── Build the full seed list ─────────────────────────────────────────────────

def build_seeds():
    """Build list of seeds: 28 concepts × 5 languages = 140."""
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
                "criteria_hint": concept["hint"],
            })

    return seeds


# ── Weight discretization (GPT-5.2 suggestion) ──────────────────────────────
# Snap continuous weights to nearest grid point to fight collapse and increase diversity

WEIGHT_GRID = [-1.0, -0.7, -0.4, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0]

def snap_to_grid(weights):
    """Snap each weight to nearest value in WEIGHT_GRID."""
    snapped = {}
    for k, v in weights.items():
        snapped[k] = min(WEIGHT_GRID, key=lambda g: abs(g - v))
    return snapped


# ── LLM query functions ─────────────────────────────────────────────────────

REASONING_MODELS = {"deepseek-r1:8b", "gpt-oss:20b"}

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


def query_openai(model, prompt, temperature=0.7, max_tokens=500):
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
        if weights is not None:
            weights = snap_to_grid(weights)
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
    print(f"Motion Seed Experiment v2: {len(seeds)} seeds × {len(MODELS)} models = {n_total} trials")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print(f"Prompt includes: weight semantics, {5} few-shot examples, behavioral criteria")
    print()

    start_time = time.time()
    trial_num = 0

    for seed_info in seeds:
        seed_word = seed_info["seed"]
        concept = seed_info["concept"]
        lang = seed_info["language"]
        hint = seed_info["criteria_hint"]
        prompt = PROMPT_TEMPLATE.format(seed=seed_word, criteria_hint=hint)

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

    print("\n" + "="*70)
    print("PER-MODEL SUMMARY")
    print("="*70)
    for model_info in MODELS:
        mname = model_info["name"]
        m_results = [r for r in results if r["model"] == mname]
        successes = [r for r in m_results if r["success"]]
        core_results = [r for r in successes if r["concept"] in CORE_CONCEPTS and r["match"] is not None]
        core_matches = [r for r in core_results if r["match"]]

        print(f"\n{mname}:")
        print(f"  Parse success: {len(successes)}/{len(m_results)}")
        if core_results:
            print(f"  Semantic match: {len(core_matches)}/{len(core_results)} "
                  f"({100*len(core_matches)/len(core_results):.0f}%)")

        dxs = [r["analytics"]["outcome"]["dx"] for r in successes if r["analytics"]]
        if dxs:
            print(f"  DX: median={np.median(dxs):+.2f}, max |DX|={max(abs(d) for d in dxs):.2f}")

    print("\n" + "="*70)
    print("PER-CONCEPT MATCH RATES")
    print("="*70)
    for concept_id in CORE_CONCEPTS:
        c_results = [r for r in results if r["concept"] == concept_id and r["success"] and r["match"] is not None]
        c_matches = [r for r in c_results if r["match"]]
        if c_results:
            print(f"  {concept_id:20s}: {len(c_matches):2d}/{len(c_results):2d} "
                  f"({100*len(c_matches)/len(c_results):5.1f}%) — {CORE_CONCEPTS[concept_id]['description']}")

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

    # Save results
    out_path = PROJECT / "artifacts" / "motion_seed_experiment_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "experiment": "v2",
                "prompt_version": "enriched (weight semantics + few-shot + criteria hints)",
                "n_seeds": len(seeds),
                "n_models": len(MODELS),
                "n_concepts": len(CORE_CONCEPTS),
                "models": [m["name"] for m in MODELS],
                "languages": ["en", "de", "zh", "fr", "fi"],
                "elapsed_seconds": elapsed,
            },
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
