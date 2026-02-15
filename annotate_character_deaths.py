#!/usr/bin/env python3
"""
annotate_character_deaths.py

Use a local LLM to annotate whether each of the 2000 archetypometrics
characters dies in their story. Batches by story for efficiency (~341 calls).

Output: artifacts/character_death_annotations.json
"""

import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-coder:30b"

PROMPT_TEMPLATE = """\
For the story "{story}", classify whether each character dies during the events of the story.

Characters: {char_list}

For EACH character, respond with exactly one of:
- "dies" — the character dies during the story (killed, suicide, sacrifice, executed, etc.)
- "survives" — the character is alive at the end of the story
- "ambiguous" — death is uncertain, off-screen, or the story is ongoing/unresolved
- "unknown" — you don't have enough knowledge about this character

Respond ONLY with a JSON object mapping character names to their status. Example:
{{"Character A": "dies", "Character B": "survives", "Character C": "ambiguous"}}

Important: Base this on the canonical/most well-known version of the story. For TV series, consider the full run of the show. For book series, consider the full series. Output ONLY the JSON object, no other text."""


def query_ollama(prompt, timeout=120):
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2000}
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
        return data["response"], None
    except Exception as e:
        return None, str(e)


def parse_json_response(resp):
    """Extract JSON object from LLM response, handling think tags and markdown."""
    text = resp
    # Strip <think>...</think> blocks
    while "<think>" in text:
        start = text.index("<think>")
        end = text.index("</think>", start) + len("</think>") if "</think>" in text[start:] else len(text)
        text = text[:start] + text[end:]

    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```" in text:
        blocks = text.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i]
            # Strip language tag
            if block.startswith("json"):
                block = block[4:]
            block = block.strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Try finding { ... } substring
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def load_characters():
    tsv_path = PROJECT / "artifacts" / "archetypometrics_characters.tsv"
    stories = defaultdict(list)
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cs = row.get("character/story", "")
            parts = cs.split("/", 1)
            char = parts[0].strip()
            story = parts[1].strip() if len(parts) > 1 else "Unknown"
            stories[story].append(char)
    return stories


def load_checkpoint(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"annotations": {}, "completed_stories": []}


def save_checkpoint(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    stories = load_characters()
    print(f"Loaded {sum(len(v) for v in stories.values())} characters from {len(stories)} stories")

    out_path = PROJECT / "artifacts" / "character_death_annotations.json"
    checkpoint = load_checkpoint(out_path)
    completed = set(checkpoint.get("completed_stories", []))
    annotations = checkpoint.get("annotations", {})

    remaining = [s for s in stories if s not in completed]
    print(f"Already done: {len(completed)} stories, {len(annotations)} characters")
    print(f"Remaining: {len(remaining)} stories")
    print(f"Model: {MODEL}")
    print()

    start_time = time.time()
    failures = 0

    for i, story in enumerate(sorted(remaining)):
        chars = stories[story]
        char_list = ", ".join(chars)
        prompt = PROMPT_TEMPLATE.format(story=story, char_list=char_list)

        print(f"[{len(completed)+1}/{len(stories)}] {story} ({len(chars)} characters)...", end=" ", flush=True)

        resp, err = query_ollama(prompt)
        if err:
            print(f"ERROR: {err}")
            failures += 1
            continue

        parsed = parse_json_response(resp)
        if parsed is None:
            print(f"PARSE FAIL")
            failures += 1
            # Store raw response for debugging
            for char in chars:
                key = f"{char}/{story}"
                annotations[key] = {"status": "parse_error", "raw": resp[:200]}
            completed.add(story)
            continue

        # Map results
        matched = 0
        for char in chars:
            key = f"{char}/{story}"
            status = parsed.get(char, None)
            if status is None:
                # Try case-insensitive match
                for k, v in parsed.items():
                    if k.lower() == char.lower():
                        status = v
                        break
            if status and status in ("dies", "survives", "ambiguous", "unknown"):
                annotations[key] = {"status": status}
                matched += 1
            elif status:
                annotations[key] = {"status": status, "raw": True}
                matched += 1
            else:
                annotations[key] = {"status": "unmatched"}

        completed.add(story)

        # Count dies/survives for this story
        story_dies = sum(1 for c in chars if annotations.get(f"{c}/{story}", {}).get("status") == "dies")
        story_survives = sum(1 for c in chars if annotations.get(f"{c}/{story}", {}).get("status") == "survives")
        print(f"matched={matched}/{len(chars)}  dies={story_dies} survives={story_survives}")

        # Checkpoint every 20 stories
        if len(completed) % 20 == 0:
            save_checkpoint(out_path, {
                "model": MODEL,
                "annotations": annotations,
                "completed_stories": list(completed),
            })
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(remaining) - i - 1) / max(rate, 0.001)
            print(f"  [checkpoint] {len(completed)}/{len(stories)} stories, "
                  f"{elapsed:.0f}s elapsed, ~{eta/60:.1f}min remaining")

    # Final save
    total_elapsed = time.time() - start_time

    # Summary stats
    statuses = defaultdict(int)
    for v in annotations.values():
        s = v if isinstance(v, str) else v.get("status", "unknown")
        statuses[s] += 1

    save_checkpoint(out_path, {
        "model": MODEL,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": total_elapsed,
        "n_stories": len(stories),
        "n_characters": len(annotations),
        "summary": dict(statuses),
        "annotations": annotations,
        "completed_stories": list(completed),
    })

    print(f"\n{'='*60}")
    print(f"DEATH ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Stories: {len(completed)}/{len(stories)}")
    print(f"Characters: {len(annotations)}")
    print(f"Failures: {failures}")
    print(f"\nStatus breakdown:")
    for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
