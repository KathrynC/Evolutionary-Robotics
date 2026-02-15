#!/usr/bin/env python3
"""
curate_mathematicians.py

Use a local LLM to build a curated, metadata-rich list of ~200 mathematicians
for the mathematician seed experiment. Generates the list in batches by era,
then annotates each with structured metadata.

Output: artifacts/mathematicians_curated.json
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-coder:30b"


def query_llm(prompt, max_tokens=4000, temperature=0.3, timeout=180):
    payload = json.dumps({
        "model": MODEL,
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
    # Strip <think>...</think>
    while "<think>" in text:
        start = text.index("<think>")
        end = text.index("</think>", start) + len("</think>") if "</think>" in text[start:] else len(text)
        text = text[:start] + text[end:]
    text = text.strip()
    # Try direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try code block
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
    # Try { ... }
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


# ── Step 1: Generate names by era ────────────────────────────────────────────

ERAS = [
    ("ancient", "Ancient & Classical (before 500 CE)", 20,
     "Euclid, Archimedes, Pythagoras, Hypatia, Brahmagupta, Al-Khwarizmi, etc."),
    ("renaissance", "Renaissance & Early Modern (1400-1700)", 25,
     "Descartes, Fermat, Pascal, Leibniz, Newton, Euler, Cardano, etc."),
    ("19th_century", "19th Century (1700-1900)", 50,
     "Gauss, Cauchy, Abel, Galois, Riemann, Cantor, Poincaré, Hilbert, Noether, Ramanujan, etc."),
    ("early_20th", "Early-Mid 20th Century (1900-1970)", 50,
     "Gödel, Turing, von Neumann, Kolmogorov, Erdos, Grothendieck, Nash, etc."),
    ("late_20th", "Late 20th Century & Living (1950-present)", 40,
     "Thurston, Perelman, Wiles, Tao, Mirzakhani, Conway, Knuth, Scholze, etc."),
    ("collectives", "Mathematical Collectives & Pseudonyms", 5,
     "Bourbaki, Polymath, Pythagorean Brotherhood, Kerala School, Bletchley Park"),
]

NAME_PROMPT = """\
List exactly {count} notable mathematicians from the era: {era_name}.

Include a diverse mix of:
- Universally famous (Euler, Gauss, Newton level)
- Famous within mathematics (Noether, Ramanujan, Galois level)
- Important but less well-known (Wedderburn, Zorn, Dilworth level)
- Women mathematicians and non-Western mathematicians where historically relevant

Examples to include (but don't limit to these): {examples}

Output a JSON array of objects, each with:
- "name": full name as commonly known
- "birth_year": integer or null if unknown
- "death_year": integer or null if alive/unknown

Output ONLY the JSON array. No other text."""


# ── Step 2: Annotate metadata ────────────────────────────────────────────────

ANNOTATE_PROMPT = """\
For each mathematician below, provide structured metadata.

Mathematicians:
{names_block}

For EACH mathematician, output a JSON object with these fields:
- "name": the name (as given)
- "birth_year": int or null
- "death_year": int or null
- "age_at_death": int or null (null if alive)
- "died_young": true if died before age 45, false otherwise, null if unknown
- "cause_of_death": one of "natural", "disease", "tragic", "violent", "suicide", "accident", "alive", "unknown"
- "primary_field": one of "algebra", "analysis", "geometry", "topology", "number_theory", "logic", "combinatorics", "probability", "applied", "CS", "physics_math", "multiple"
- "secondary_field": same options or null
- "style": one of "systematic", "intuitive", "visual", "algebraic", "computational", "applied", "eclectic"
- "fame_level": one of "household" (Newton, Einstein level), "educated_public" (Euler, Turing), "math_famous" (Noether, Galois), "specialist" (Wedderburn, Zorn)
- "known_for": 1-sentence description of main contribution
- "associated_sequences": list of OEIS IDs if any famous sequences bear their name (e.g., ["A000045"] for Fibonacci), empty list otherwise

Output a JSON array of these objects. Keep "known_for" to ONE concise sentence. Output ONLY the JSON array."""


def generate_names_for_era(era_id, era_name, count, examples):
    prompt = NAME_PROMPT.format(
        count=count, era_name=era_name, examples=examples
    )
    print(f"  Generating {count} names for {era_name}...", end=" ", flush=True)
    resp = query_llm(prompt, max_tokens=6000)
    if resp is None:
        print("LLM ERROR")
        return []
    parsed = parse_json_response(resp)
    if parsed is None:
        print("PARSE FAIL")
        return []
    print(f"got {len(parsed)}")
    # Tag with era
    for entry in parsed:
        entry["era"] = era_id
    return parsed


def annotate_batch(mathematicians):
    names_block = "\n".join(
        f"- {m['name']} ({m.get('birth_year','?')}–{m.get('death_year','?')})"
        for m in mathematicians
    )
    prompt = ANNOTATE_PROMPT.format(names_block=names_block)
    resp = query_llm(prompt, max_tokens=8000, timeout=300)
    if resp is None:
        return None
    return parse_json_response(resp)


def main():
    out_path = PROJECT / "artifacts" / "mathematicians_curated.json"

    # Step 1: Generate names by era
    print("Step 1: Generating mathematician names by era\n")
    all_mathematicians = []
    for era_id, era_name, count, examples in ERAS:
        names = generate_names_for_era(era_id, era_name, count, examples)
        all_mathematicians.extend(names)

    print(f"\nTotal names generated: {len(all_mathematicians)}")

    # Deduplicate by name
    seen = set()
    unique = []
    for m in all_mathematicians:
        name = m["name"].lower().strip()
        if name not in seen:
            seen.add(name)
            unique.append(m)
    all_mathematicians = unique
    print(f"After dedup: {len(all_mathematicians)}")

    # Step 2: Annotate in batches of 15
    print("\nStep 2: Annotating metadata\n")
    batch_size = 15
    annotated = []
    for i in range(0, len(all_mathematicians), batch_size):
        batch = all_mathematicians[i:i+batch_size]
        batch_names = [m["name"] for m in batch]
        print(f"  [{i+1}-{min(i+batch_size, len(all_mathematicians))}/{len(all_mathematicians)}] "
              f"{batch_names[0]}...{batch_names[-1]}", end=" ", flush=True)

        result = annotate_batch(batch)
        if result is None:
            print("ANNOTATION FAIL — keeping names only")
            for m in batch:
                m["annotation_failed"] = True
            annotated.extend(batch)
            continue

        # Merge annotations with era info
        result_by_name = {}
        for r in result:
            result_by_name[r["name"].lower().strip()] = r

        matched = 0
        for m in batch:
            key = m["name"].lower().strip()
            if key in result_by_name:
                ann = result_by_name[key]
                ann["era"] = m["era"]
                annotated.append(ann)
                matched += 1
            else:
                # Try fuzzy match
                found = False
                for rkey, rval in result_by_name.items():
                    if key.split()[-1].lower() in rkey.lower() or rkey.split()[-1].lower() in key.lower():
                        rval["era"] = m["era"]
                        annotated.append(rval)
                        matched += 1
                        found = True
                        break
                if not found:
                    m["annotation_failed"] = True
                    annotated.append(m)

        print(f"matched {matched}/{len(batch)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"MATHEMATICIAN CURATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(annotated)}")

    # Stats
    died_young = sum(1 for m in annotated if m.get("died_young") == True)
    alive = sum(1 for m in annotated if m.get("cause_of_death") == "alive")
    tragic = sum(1 for m in annotated if m.get("cause_of_death") in ("tragic", "violent", "suicide"))
    fields = {}
    for m in annotated:
        f = m.get("primary_field", "unknown")
        fields[f] = fields.get(f, 0) + 1
    fame = {}
    for m in annotated:
        f = m.get("fame_level", "unknown")
        fame[f] = fame.get(f, 0) + 1

    print(f"Died young (<45): {died_young}")
    print(f"Alive: {alive}")
    print(f"Tragic/violent/suicide: {tragic}")
    print(f"\nFields: {json.dumps(fields, indent=2)}")
    print(f"\nFame levels: {json.dumps(fame, indent=2)}")

    # Save
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "n_mathematicians": len(annotated),
        "mathematicians": annotated,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
