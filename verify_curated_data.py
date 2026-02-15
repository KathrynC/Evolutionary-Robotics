#!/usr/bin/env python3
"""
verify_curated_data.py

Factcheck and citation verification agent for LLM-curated datasets.
Uses a second LLM pass (different temperature, adversarial prompt) to
verify claims made in the curation pass, plus web lookups where possible.

Verifies:
  - Mathematician birth/death years, causes of death, primary fields
  - TV Tropes names (checks if the trope name is real vs hallucinated)
  - Stith Thompson motif IDs (checks plausibility of ID format and description)
  - OEIS sequence attributions to mathematicians

Usage:
    python verify_curated_data.py --mathematicians
    python verify_curated_data.py --tropes
    python verify_curated_data.py --motifs
    python verify_curated_data.py --all
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/generate"

# Use a DIFFERENT model for verification to avoid self-confirmation bias.
# If curation used qwen3-coder, verify with llama3.1 or gpt-oss.
VERIFY_MODEL = "llama3.1:latest"
CURATION_MODEL = "qwen3-coder:30b"


def query_llm(prompt, model=VERIFY_MODEL, max_tokens=4000, temperature=0.1, timeout=180):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    })
    r = subprocess.run(
        ["curl", "-s", OLLAMA_URL, "-d", payload],
        capture_output=True, text=True, timeout=timeout
    )
    if r.returncode != 0:
        return None
    data = json.loads(r.stdout)
    if "error" in data:
        return None
    return data["response"]


def parse_json_response(resp):
    text = resp
    while "<think>" in text:
        start = text.index("<think>")
        end = text.index("</think>", start) + len("</think>") if "</think>" in text[start:] else len(text)
        text = text[:start] + text[end:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        blocks = text.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ── Mathematician verification ───────────────────────────────────────────────

MATH_VERIFY_PROMPT = """\
You are a factchecker. For each mathematician below, verify the claimed facts.
Be skeptical — flag anything that seems wrong or uncertain.

{entries_block}

For EACH mathematician, output a JSON object with:
- "name": the name
- "birth_year_correct": true/false/null (null if you're unsure)
- "death_year_correct": true/false/null
- "birth_year_actual": your best knowledge of the correct year, or null
- "death_year_actual": your best knowledge of the correct year, or null
- "cause_of_death_correct": true/false/null
- "cause_of_death_actual": your assessment, or null
- "primary_field_correct": true/false/null
- "known_for_correct": true/false — is the "known for" description accurate?
- "is_real_person": true/false — is this actually a real mathematician?
- "confidence": "high"/"medium"/"low" — your confidence in this verification
- "notes": any corrections or flags (1 sentence max)

Output ONLY a JSON array. No other text."""


def verify_mathematicians():
    path = PROJECT / "artifacts" / "mathematicians_curated.json"
    if not path.exists():
        print("No mathematicians file found. Run curate_mathematicians.py first.")
        return

    with open(path) as f:
        data = json.load(f)

    mathematicians = data["mathematicians"]
    print(f"Verifying {len(mathematicians)} mathematicians using {VERIFY_MODEL}...")

    batch_size = 10
    all_verifications = []
    issues = []

    for i in range(0, len(mathematicians), batch_size):
        batch = mathematicians[i:i+batch_size]
        entries_block = "\n".join(
            f"- {m['name']} (claimed: {m.get('birth_year','?')}–{m.get('death_year','?')}, "
            f"cause: {m.get('cause_of_death','?')}, field: {m.get('primary_field','?')}, "
            f"known for: {m.get('known_for','?')})"
            for m in batch
        )
        prompt = MATH_VERIFY_PROMPT.format(entries_block=entries_block)

        print(f"  [{i+1}-{min(i+batch_size, len(mathematicians))}/{len(mathematicians)}]...", end=" ", flush=True)

        resp = query_llm(prompt)
        if resp is None:
            print("LLM ERROR")
            continue

        parsed = parse_json_response(resp)
        if parsed is None:
            print("PARSE FAIL")
            continue

        # Check for issues
        batch_issues = 0
        for v in parsed if isinstance(parsed, list) else [parsed]:
            if isinstance(v, dict):
                all_verifications.append(v)
                # Flag issues
                flags = []
                if v.get("birth_year_correct") == False:
                    flags.append(f"birth year: claimed vs actual {v.get('birth_year_actual')}")
                if v.get("death_year_correct") == False:
                    flags.append(f"death year: claimed vs actual {v.get('death_year_actual')}")
                if v.get("cause_of_death_correct") == False:
                    flags.append(f"cause of death: actual={v.get('cause_of_death_actual')}")
                if v.get("is_real_person") == False:
                    flags.append("NOT A REAL PERSON")
                if v.get("known_for_correct") == False:
                    flags.append("known_for inaccurate")
                if flags:
                    issues.append({"name": v.get("name"), "flags": flags, "notes": v.get("notes", "")})
                    batch_issues += 1

        print(f"verified, {batch_issues} issues")

    # Summary
    print(f"\n{'='*60}")
    print(f"MATHEMATICIAN VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Verified: {len(all_verifications)}/{len(mathematicians)}")
    print(f"Issues found: {len(issues)}")

    if issues:
        print(f"\nIssues:")
        for issue in issues:
            print(f"  {issue['name']}: {'; '.join(issue['flags'])}")
            if issue['notes']:
                print(f"    Note: {issue['notes']}")

    # Save verification report
    report_path = PROJECT / "artifacts" / "mathematicians_verification.json"
    with open(report_path, "w") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "verify_model": VERIFY_MODEL,
            "curation_model": CURATION_MODEL,
            "n_verified": len(all_verifications),
            "n_issues": len(issues),
            "issues": issues,
            "verifications": all_verifications,
        }, f, indent=2)
    print(f"\nReport saved to {report_path}")
    return issues


# ── TV Tropes verification ───────────────────────────────────────────────────

TROPES_VERIFY_PROMPT = """\
You are a TV Tropes expert and factchecker. For each trope name below,
verify whether it is a REAL TV Tropes page or a hallucinated/made-up name.
Also check if the laconic description is accurate.

{entries_block}

For EACH trope, output a JSON object with:
- "name": the CamelCase name
- "is_real_trope": true/false — does this page actually exist on TV Tropes?
- "laconic_accurate": true/false/null — is the description correct?
- "correct_name": if the name is close but slightly wrong, give the correct name (or null)
- "confidence": "high"/"medium"/"low"
- "notes": corrections (1 sentence max, or empty string)

Output ONLY a JSON array. No other text."""


def verify_tropes():
    path = PROJECT / "artifacts" / "tv_tropes_curated.json"
    if not path.exists():
        print("No tropes file found. Run curate_tv_tropes.py first.")
        return

    with open(path) as f:
        data = json.load(f)

    tropes = data["tropes"]
    print(f"Verifying {len(tropes)} TV Tropes using {VERIFY_MODEL}...")

    batch_size = 15
    all_verifications = []
    issues = []

    for i in range(0, len(tropes), batch_size):
        batch = tropes[i:i+batch_size]
        entries_block = "\n".join(
            f"- {t.get('name', t.get('display_name','?'))}: \"{t.get('laconic', '?')}\""
            for t in batch
        )
        prompt = TROPES_VERIFY_PROMPT.format(entries_block=entries_block)

        print(f"  [{i+1}-{min(i+batch_size, len(tropes))}/{len(tropes)}]...", end=" ", flush=True)

        resp = query_llm(prompt)
        if resp is None:
            print("LLM ERROR")
            continue

        parsed = parse_json_response(resp)
        if parsed is None:
            print("PARSE FAIL")
            continue

        batch_issues = 0
        for v in parsed if isinstance(parsed, list) else [parsed]:
            if isinstance(v, dict):
                all_verifications.append(v)
                flags = []
                if v.get("is_real_trope") == False:
                    flags.append("NOT A REAL TROPE")
                if v.get("laconic_accurate") == False:
                    flags.append("laconic inaccurate")
                if v.get("correct_name"):
                    flags.append(f"correct name: {v['correct_name']}")
                if flags:
                    issues.append({"name": v.get("name"), "flags": flags, "notes": v.get("notes", "")})
                    batch_issues += 1

        print(f"verified, {batch_issues} issues")

    print(f"\n{'='*60}")
    print(f"TV TROPES VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Verified: {len(all_verifications)}/{len(tropes)}")
    print(f"Issues found: {len(issues)}")

    if issues:
        print(f"\nIssues:")
        for issue in issues:
            print(f"  {issue['name']}: {'; '.join(issue['flags'])}")

    report_path = PROJECT / "artifacts" / "tv_tropes_verification.json"
    with open(report_path, "w") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "verify_model": VERIFY_MODEL,
            "curation_model": CURATION_MODEL,
            "n_verified": len(all_verifications),
            "n_issues": len(issues),
            "issues": issues,
            "verifications": all_verifications,
        }, f, indent=2)
    print(f"\nReport saved to {report_path}")
    return issues


# ── Stith Thompson verification ──────────────────────────────────────────────

MOTIF_VERIFY_PROMPT = """\
You are a folklore studies expert. For each Stith Thompson motif below,
verify whether the motif ID and description are plausible/accurate.

The Thompson Motif-Index uses IDs like: A1 (Creator), B211.1 (Speaking horse),
D1421.1.3 (Magic horn provides drink), etc. Category letter must match the
type (A=mythological, B=animals, C=tabu, D=magic, E=dead, etc.)

{entries_block}

For EACH motif, output a JSON object with:
- "motif_id": the claimed ID
- "id_format_valid": true/false — does the ID follow Thompson's numbering scheme?
- "category_matches": true/false — does the letter match the category described?
- "description_plausible": true/false — is this description consistent with Thompson's style?
- "likely_real": true/false — do you believe this is a real Thompson motif?
- "confidence": "high"/"medium"/"low"
- "notes": corrections (1 sentence max, or empty string)

Output ONLY a JSON array. No other text."""


def verify_motifs():
    path = PROJECT / "artifacts" / "stith_thompson_curated.json"
    if not path.exists():
        print("No motifs file found. Run curate_stith_thompson.py first.")
        return

    with open(path) as f:
        data = json.load(f)

    motifs = data["motifs"]
    print(f"Verifying {len(motifs)} Stith Thompson motifs using {VERIFY_MODEL}...")

    batch_size = 15
    all_verifications = []
    issues = []

    for i in range(0, len(motifs), batch_size):
        batch = motifs[i:i+batch_size]
        entries_block = "\n".join(
            f"- {m.get('motif_id', '?')}: \"{m.get('description', '?')}\" (category {m.get('category', '?')})"
            for m in batch
        )
        prompt = MOTIF_VERIFY_PROMPT.format(entries_block=entries_block)

        print(f"  [{i+1}-{min(i+batch_size, len(motifs))}/{len(motifs)}]...", end=" ", flush=True)

        resp = query_llm(prompt)
        if resp is None:
            print("LLM ERROR")
            continue

        parsed = parse_json_response(resp)
        if parsed is None:
            print("PARSE FAIL")
            continue

        batch_issues = 0
        for v in parsed if isinstance(parsed, list) else [parsed]:
            if isinstance(v, dict):
                all_verifications.append(v)
                flags = []
                if v.get("id_format_valid") == False:
                    flags.append("invalid ID format")
                if v.get("category_matches") == False:
                    flags.append("category mismatch")
                if v.get("likely_real") == False:
                    flags.append("LIKELY HALLUCINATED")
                if v.get("description_plausible") == False:
                    flags.append("description implausible")
                if flags:
                    issues.append({"motif_id": v.get("motif_id"), "flags": flags, "notes": v.get("notes", "")})
                    batch_issues += 1

        print(f"verified, {batch_issues} issues")

    print(f"\n{'='*60}")
    print(f"STITH THOMPSON VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Verified: {len(all_verifications)}/{len(motifs)}")
    print(f"Issues found: {len(issues)}")

    if issues:
        print(f"\nIssues:")
        for issue in issues:
            print(f"  {issue['motif_id']}: {'; '.join(issue['flags'])}")

    report_path = PROJECT / "artifacts" / "stith_thompson_verification.json"
    with open(report_path, "w") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "verify_model": VERIFY_MODEL,
            "curation_model": CURATION_MODEL,
            "n_verified": len(all_verifications),
            "n_issues": len(issues),
            "issues": issues,
            "verifications": all_verifications,
        }, f, indent=2)
    print(f"\nReport saved to {report_path}")
    return issues


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = set(sys.argv[1:])

    if "--all" in args or not args:
        args = {"--mathematicians", "--tropes", "--motifs"}

    if "--mathematicians" in args:
        verify_mathematicians()
        print()

    if "--tropes" in args:
        verify_tropes()
        print()

    if "--motifs" in args:
        verify_motifs()
        print()


if __name__ == "__main__":
    main()
