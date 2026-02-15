#!/usr/bin/env python3
"""
oeis_seed_experiment.py

Use notable integer sequences from OEIS as semantic seeds for the LLM→weight
pipeline. Each sequence is presented with its ID, name, and first 16 terms.
The LLM translates the mathematical "energy" of the sequence into 6 synaptic
weights for the 3-link walking robot.

Pipeline per trial:
  1. Fetch sequence from OEIS (cached locally)
  2. Prompt LLM with sequence ID, name, description, first 16 terms
  3. Parse 6 weights from LLM response
  4. Run headless PyBullet simulation (4000 steps @ 240 Hz)
  5. Compute Beer-framework analytics
  6. Record everything

Scale: ~200 sequences × 4 local Ollama models = ~800 trials
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
    parse_weights, run_trial_inmemory,
    WEIGHT_NAMES, OLLAMA_URL
)
from compute_beer_analytics import NumpyEncoder

# ── Curated OEIS sequences ──────────────────────────────────────────────────
# Organized by mathematical flavor to probe different kinds of structure.

SEQUENCES = {
    # === Growth & Accumulation ===
    "A000027": "The positive integers (natural numbers)",
    "A000079": "Powers of 2",
    "A000142": "Factorial numbers",
    "A000290": "The squares",
    "A000578": "The cubes",
    "A001477": "The nonneg integers (0,1,2,3,...)",
    "A000217": "Triangular numbers",
    "A000292": "Tetrahedral numbers",
    "A000326": "Pentagonal numbers",
    "A000384": "Hexagonal numbers",

    # === Fibonacci & Golden Ratio ===
    "A000045": "Fibonacci numbers",
    "A000032": "Lucas numbers",
    "A001622": "Decimal expansion of golden ratio",
    "A000931": "Padovan sequence",
    "A001608": "Perrin sequence",
    "A000073": "Tribonacci numbers",
    "A003714": "Fibbinary numbers",

    # === Primes & Divisibility ===
    "A000040": "The prime numbers",
    "A001358": "Semiprimes",
    "A000961": "Prime powers",
    "A002808": "Composite numbers",
    "A000005": "d(n) — number of divisors",
    "A000010": "Euler totient function",
    "A000203": "sigma(n) — sum of divisors",
    "A008683": "Moebius function",
    "A000720": "pi(n) — prime counting function",
    "A001223": "Prime gaps",
    "A002385": "Palindromic primes",
    "A000668": "Mersenne primes",
    "A006882": "Double factorials",

    # === Combinatorial ===
    "A000108": "Catalan numbers",
    "A000110": "Bell numbers",
    "A000670": "Fubini numbers (ordered Bell)",
    "A001006": "Motzkin numbers",
    "A000984": "Central binomial coefficients",
    "A000041": "Partitions of n",
    "A000129": "Pell numbers",
    "A001700": "3^n - 2^n",

    # === Recursive & Self-referential ===
    "A005132": "Recamán's sequence",
    "A006577": "Steps to reach 1 in Collatz (3n+1)",
    "A003215": "Hex (centered hexagonal) numbers",
    "A002262": "Triangle read by rows: T(n,k) = k",
    "A007318": "Pascal's triangle read by rows",
    "A000120": "Binary weight of n (number of 1s)",
    "A000002": "Kolakoski sequence",
    "A001462": "Golomb's sequence",
    "A006519": "Highest power of 2 dividing n",

    # === Chaotic & Pseudorandom ===
    "A000796": "Decimal expansion of Pi",
    "A001113": "Decimal expansion of e",
    "A002193": "Decimal expansion of sqrt(2)",
    "A000583": "Fourth powers",
    "A001511": "2-ruler sequence",
    "A030101": "Binary reversal of n",
    "A036044": "Irregular triangle from Stern-Brocot",
    "A005408": "The odd numbers",
    "A005843": "The even numbers",

    # === Oscillating & Periodic ===
    "A000035": "Period 2: 0,1,0,1,...",
    "A000034": "Period 2: 1,2,1,2,...",
    "A011655": "Period 3: 0,1,2,0,1,2,...",
    "A010060": "Thue-Morse sequence",
    "A001285": "Thue-Morse (1,2 version)",
    "A004718": "Aperiodic binary sequence",
    "A014577": "Regular paper-folding (dragon curve)",
    "A005614": "Fibonacci word (binary)",

    # === Sparse & Explosive ===
    "A000079": "Powers of 2",
    "A000244": "Powers of 3",
    "A000400": "Powers of 6",
    "A001146": "2^(2^n)",
    "A007953": "Digital sum of n",
    "A055642": "Number of digits of n",
    "A000120": "Hamming weight",

    # === Number Theory Exotica ===
    "A000169": "n^(n-1)",
    "A001333": "Numerators of convergents to sqrt(2)",
    "A000225": "2^n - 1 (Mersenne numbers)",
    "A000051": "2^n + 1",
    "A000043": "Mersenne exponents",
    "A000396": "Perfect numbers",
    "A005100": "Deficient numbers",
    "A005101": "Abundant numbers",
    "A000219": "Number of planar partitions",

    # === Geometry & Space ===
    "A000124": "Lazy caterer's sequence",
    "A000127": "Regions of a circle (Motzkin)",
    "A006003": "n(n^2+1)/2 — centered octahedral",
    "A000330": "Sum of squares",
    "A000537": "Sum of cubes",
    "A002378": "Oblong (pronic) numbers",

    # === Music & Signal ===
    "A005187": "a(n) = a(floor(n/2)) + n",
    "A000201": "Beatty sequence for sqrt(2)",
    "A001950": "Upper Wythoff sequence",
    "A000149": "Related to continued fractions",

    # === Sequences about sequences ===
    "A007947": "Largest squarefree factor of n",
    "A001221": "omega(n) — number of prime factors",
    "A001222": "Omega(n) — with multiplicity",
    "A003418": "lcm(1,...,n)",
    "A000793": "Landau's function",

    # === Famous Constants (digit sequences) ===
    "A007376": "Digits of the Champernowne constant",
    "A000796": "Digits of pi",
    "A001113": "Digits of e",
    "A010815": "Related to Jacobi theta",

    # === Deceptively simple ===
    "A000012": "The all 1's sequence",
    "A000004": "The all 0's sequence",
    "A000007": "The characteristic function of 0",
    "A000290": "Perfect squares",
    "A005117": "Squarefree numbers",
}

# Deduplicate (some appear under multiple categories)
SEQUENCES = dict(sorted(set(SEQUENCES.items())))


# ── OEIS fetch ──────────────────────────────────────────────────────────────

def fetch_oeis_sequence(seq_id, cache_dir):
    """Fetch a sequence from OEIS, caching locally."""
    cache_file = cache_dir / f"{seq_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"https://oeis.org/search?q=id:{seq_id}&fmt=json"
    try:
        r = subprocess.run(
            ["curl", "-s", url],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        seq = data[0]
        # Cache it
        with open(cache_file, "w") as f:
            json.dump(seq, f, indent=2)
        return seq
    except Exception as e:
        print(f"  OEIS fetch error for {seq_id}: {e}")
        return None


def get_first_terms(seq_data, n=16):
    """Extract first n terms from OEIS data string."""
    terms_str = seq_data.get("data", "")
    terms = []
    for t in terms_str.split(","):
        t = t.strip()
        if t == "" or t.startswith("-"):
            try:
                terms.append(int(t))
            except ValueError:
                break
        else:
            try:
                terms.append(int(t))
            except ValueError:
                break
        if len(terms) >= n:
            break
    return terms


# ── LLM prompt ──────────────────────────────────────────────────────────────

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
    "I want you to translate a mathematical integer sequence into movement.\n\n"
    "Sequence: {seq_id} — {seq_name}\n"
    "First 16 terms: {terms}\n\n"
    "Think about the character of this sequence — its growth rate, its rhythm, "
    "its regularity or chaos, its mathematical personality. A sequence that grows "
    "explosively might produce aggressive, lunging movement. A periodic sequence "
    "might produce steady, rhythmic walking. A chaotic sequence might produce "
    "erratic stumbling. A constant sequence might produce stillness.\n\n"
    "Choose each weight from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].\n"
    "In 1-2 sentences, describe the movement quality you're going for. "
    "Then output ONLY the JSON object with keys w03, w04, w13, w14, w23, w24. "
    "Keep reasoning SHORT."
)

# ── Weight discretization ────────────────────────────────────────────────────

WEIGHT_GRID = [-1.0, -0.7, -0.4, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0]

def snap_to_grid(weights):
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


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(path, results, metadata):
    completed_keys = {f"{r['seq_id']}|{r['model']}" for r in results}
    with open(path, "w") as f:
        json.dump({
            "metadata": metadata,
            "completed_keys": list(completed_keys),
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    cache_dir = PROJECT / "artifacts" / "oeis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = PROJECT / "artifacts" / "oeis_seed_experiment.json"
    checkpoint_path = PROJECT / "artifacts" / "oeis_seed_experiment_checkpoint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch all sequences from OEIS
    print(f"Fetching {len(SEQUENCES)} sequences from OEIS...")
    seq_data = {}
    for seq_id in sorted(SEQUENCES.keys()):
        data = fetch_oeis_sequence(seq_id, cache_dir)
        if data:
            terms = get_first_terms(data)
            if len(terms) >= 6:
                seq_data[seq_id] = {
                    "id": seq_id,
                    "name": data["name"],
                    "terms": terms,
                    "description": SEQUENCES[seq_id],
                }
            else:
                print(f"  {seq_id}: too few terms ({len(terms)}), skipping")
        else:
            print(f"  {seq_id}: fetch failed, skipping")
        time.sleep(0.3)  # Be polite to OEIS

    print(f"Fetched {len(seq_data)} sequences successfully")

    # Load checkpoint
    results = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        completed_keys = set(ckpt.get("completed_keys", []))
        print(f"Resumed from checkpoint: {len(results)} trials already done")

    n_total = len(seq_data) * len(MODELS)
    n_remaining = n_total - len(completed_keys)
    print(f"\nOEIS Seed Experiment: {len(seq_data)} sequences × {len(MODELS)} models = {n_total} trials")
    print(f"Remaining: {n_remaining} trials")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print()

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "oeis_seed",
        "n_sequences": len(seq_data),
        "n_models": len(MODELS),
        "models": [m["name"] for m in MODELS],
    }

    start_time = time.time()
    trial_num = len(completed_keys)
    failures = 0
    checkpoint_interval = 50

    for seq_id in sorted(seq_data.keys()):
        seq = seq_data[seq_id]
        terms_str = ", ".join(str(t) for t in seq["terms"][:16])

        prompt = PROMPT_TEMPLATE.format(
            seq_id=seq_id,
            seq_name=seq["name"],
            terms=terms_str,
        )

        for model_info in MODELS:
            model_name = model_info["name"]
            key = f"{seq_id}|{model_name}"

            if key in completed_keys:
                continue

            trial_num += 1
            print(f"[{trial_num}/{n_total}] {model_name} | {seq_id} {seq['name'][:50]}", end=" ", flush=True)

            weights, raw_resp = query_ollama(model_name, prompt)

            if weights is None:
                failures += 1
                print("-> PARSE FAIL")
                results.append({
                    "seq_id": seq_id,
                    "seq_name": seq["name"],
                    "seq_terms": seq["terms"][:16],
                    "model": model_name,
                    "success": False,
                    "raw_response": raw_resp[:500] if raw_resp else "",
                    "weights": None,
                    "analytics": None,
                })
                completed_keys.add(key)
                continue

            try:
                analytics = run_trial_inmemory(weights)
            except Exception as e:
                failures += 1
                print(f"-> SIM ERROR: {e}")
                results.append({
                    "seq_id": seq_id,
                    "seq_name": seq["name"],
                    "seq_terms": seq["terms"][:16],
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
                "seq_id": seq_id,
                "seq_name": seq["name"],
                "seq_terms": seq["terms"][:16],
                "model": model_name,
                "success": True,
                "weights": weights,
                "analytics": analytics,
            })
            completed_keys.add(key)

            if len(completed_keys) % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, results, metadata)
                elapsed = time.time() - start_time
                rate = (trial_num - (n_total - n_remaining)) / max(elapsed, 1)
                remaining_time = (n_total - len(completed_keys)) / max(rate, 0.01)
                print(f"  [checkpoint] {len(completed_keys)}/{n_total} done, "
                      f"{elapsed:.0f}s elapsed, ~{remaining_time/60:.0f}min remaining")

    total_elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"OEIS SEED EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Trials: {len(results)} ({failures} failures)")

    successes = [r for r in results if r["success"]]
    if successes:
        dxs = [abs(r["analytics"]["outcome"]["dx"]) for r in successes]
        dead = sum(1 for d in dxs if d < 1.0)
        print(f"\nOverall:")
        print(f"  Dead (|DX|<1m): {dead}/{len(successes)} ({100*dead/len(successes):.1f}%)")
        print(f"  Median |DX|: {np.median(dxs):.2f}m")
        print(f"  Max |DX|: {max(dxs):.2f}m")

        # Per-model
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

        # Top 10 most mobile sequences
        from collections import defaultdict
        seq_best = defaultdict(lambda: 0)
        seq_info = {}
        for r in successes:
            dx_abs = abs(r["analytics"]["outcome"]["dx"])
            if dx_abs > seq_best[r["seq_id"]]:
                seq_best[r["seq_id"]] = dx_abs
                seq_info[r["seq_id"]] = r

        print(f"\nTop 10 most mobile sequences:")
        sorted_seqs = sorted(seq_best.items(), key=lambda x: x[1], reverse=True)
        for sid, best_dx in sorted_seqs[:10]:
            r = seq_info[sid]
            print(f"  {sid} {r['seq_name'][:45]:45s}: |DX|={best_dx:.2f}m ({r['model']})")

        # Top 10 most frozen
        print(f"\nTop 10 most frozen sequences:")
        for sid, best_dx in sorted_seqs[-10:]:
            r = seq_info[sid]
            print(f"  {sid} {r['seq_name'][:45]:45s}: |DX|={best_dx:.2f}m ({r['model']})")

    # Save final
    metadata["elapsed_seconds"] = total_elapsed
    metadata["n_results"] = len(results)
    metadata["n_failures"] = failures

    with open(out_path, "w") as f:
        json.dump({
            "metadata": metadata,
            "results": results,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint removed")


if __name__ == "__main__":
    run_experiment()
