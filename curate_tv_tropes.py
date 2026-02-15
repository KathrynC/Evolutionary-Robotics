#!/usr/bin/env python3
"""
curate_tv_tropes.py

Use a local LLM to build a curated list of ~200 TV Tropes with laconic
descriptions and metadata, organized by narrative function.

Output: artifacts/tv_tropes_curated.json
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-coder:30b"


def query_llm(prompt, max_tokens=6000, temperature=0.3, timeout=180):
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
    return None


CATEGORIES = [
    ("movement_physical", "Movement and Physical Action Tropes", 25,
     "Tropes specifically about how characters move, fight, walk, run, fall, or use their bodies. "
     "Examples: The Slow Walk, Unflinching Walk, Le Parkour, Death by Falling, Taking the Bullet, "
     "Walk and Talk, Dramatic Chase Opening, Roof Hopping, Stumbling, Power Walk, Unnecessary Combat Roll."),

    ("character_archetypes", "Character Archetype Tropes", 30,
     "Tropes that define character types — who they are, how they carry themselves. "
     "Examples: The Hero, The Mentor, The Dragon, Big Bad, Damsel in Distress, Byronic Hero, "
     "Action Girl, Gentle Giant, Dark Lord, Plucky Comic Relief, The Stoic, Berserker, "
     "The Lancer, Femme Fatale, Mad Scientist, Anti-Hero, Lovable Rogue, Wise Old Mentor."),

    ("death_sacrifice", "Death, Sacrifice, and Ending Tropes", 25,
     "Tropes about dying, killing, sacrifice, and how stories end. "
     "Examples: Heroic Sacrifice, Anyone Can Die, Kill the Cutie, Redemption Equals Death, "
     "Killed Off for Real, Disney Death, Happy Ending, Downer Ending, Bittersweet Ending, "
     "Dead Man Walking, The Dog Dies, Sudden Sequel Death, Death Is Dramatic."),

    ("plot_mechanics", "Plot Mechanic Tropes", 25,
     "Structural tropes about how stories work. "
     "Examples: Chekhov's Gun, Deus Ex Machina, Red Herring, Plot Twist, MacGuffin, "
     "Cliffhanger, Darkest Hour, The Reveal, Foreshadowing, Flashback, Montage, "
     "In Medias Res, Bottle Episode, The Stinger, Wham Episode."),

    ("tone_energy", "Tone and Energy Tropes", 25,
     "Tropes about mood, pacing, emotional register. "
     "Examples: Mood Whiplash, Cerebus Syndrome, Breather Episode, Rule of Cool, "
     "Rule of Funny, Nightmare Fuel, Heartwarming Moments, Tear Jerker, "
     "Narm, Bathos, Darker and Edgier, Lighter and Softer, Cringe Comedy."),

    ("conflict_tension", "Conflict and Tension Tropes", 25,
     "Tropes about confrontation, struggle, rivalry, escalation. "
     "Examples: The Rival, Escalating War, Mexican Standoff, Battle Royale, "
     "Curb-Stomp Battle, David vs. Goliath, Enemy Mine, Let's Fight Like Gentlemen, "
     "Villain Team-Up, Mirror Match, Final Boss, Unstoppable Rage."),

    ("comedy_absurdity", "Comedy and Absurdity Tropes", 20,
     "Tropes about humor, absurdity, and breaking expectations. "
     "Examples: Slapstick, Running Gag, Brick Joke, Deadpan Snarker, "
     "Butt-Monkey, Epic Fail, Refuge in Audacity, Insane Troll Logic, "
     "Straight Man, Fourth Wall Break, Comically Serious."),

    ("meta_structural", "Meta and Structural Tropes", 15,
     "Tropes about storytelling itself, genre awareness, narrative structure. "
     "Examples: Lampshade Hanging, Breaking the Fourth Wall, Genre Savvy, "
     "Medium Awareness, Leaning on the Fourth Wall, This Is Reality, "
     "Tropes Are Tools, Dead Horse Trope, Deconstruction, Reconstruction."),

    ("transformation_power", "Transformation and Power Tropes", 15,
     "Tropes about gaining, losing, or changing power and form. "
     "Examples: Power-Up, Superpowered Evil Side, Drunk with Power, "
     "Brought Down to Normal, One-Winged Angel, Shapeshifting, "
     "Evolution Power-Up, Limit Break, Deadly Upgrade, With Great Power."),
]

TROPE_PROMPT = """\
List exactly {count} well-known TV Tropes in the category: {category_name}

{category_description}

For each trope, provide:
- "name": the exact TV Tropes name (CamelCase as on the site, e.g. "TheSlowWalk")
- "display_name": human-readable name (e.g. "The Slow Walk")
- "laconic": a ONE-sentence description (the "laconic" version)
- "movement_association": how strongly this trope relates to physical movement (high/medium/low/none)
- "energy_level": the typical energy of this trope (explosive/high/moderate/low/still)
- "valence": emotional valence (positive/negative/neutral/mixed)

Pick tropes that are well-known and widely recognized — they should be tropes that
a well-read person would recognize by name. Prioritize tropes with vivid, evocative names.

Output ONLY a JSON array. No other text."""


def generate_tropes_for_category(cat_id, cat_name, count, cat_desc):
    prompt = TROPE_PROMPT.format(
        count=count, category_name=cat_name, category_description=cat_desc
    )
    print(f"  Generating {count} tropes for {cat_name}...", end=" ", flush=True)
    resp = query_llm(prompt, max_tokens=8000)
    if resp is None:
        print("LLM ERROR")
        return []
    parsed = parse_json_response(resp)
    if parsed is None:
        print(f"PARSE FAIL")
        return []
    print(f"got {len(parsed)}")
    for entry in parsed:
        entry["category"] = cat_id
        entry["category_name"] = cat_name
    return parsed


def main():
    out_path = PROJECT / "artifacts" / "tv_tropes_curated.json"

    print("Curating TV Tropes list\n")

    all_tropes = []
    for cat_id, cat_name, count, cat_desc in CATEGORIES:
        tropes = generate_tropes_for_category(cat_id, cat_name, count, cat_desc)
        all_tropes.extend(tropes)

    # Deduplicate by name
    seen = set()
    unique = []
    for t in all_tropes:
        key = t.get("name", t.get("display_name", "")).lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    all_tropes = unique

    # Summary
    print(f"\n{'='*60}")
    print(f"TV TROPES CURATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total tropes: {len(all_tropes)}")

    by_cat = {}
    for t in all_tropes:
        c = t.get("category", "unknown")
        by_cat[c] = by_cat.get(c, 0) + 1
    for c, n in sorted(by_cat.items()):
        print(f"  {c}: {n}")

    by_energy = {}
    for t in all_tropes:
        e = t.get("energy_level", "unknown")
        by_energy[e] = by_energy.get(e, 0) + 1
    print(f"\nEnergy levels: {json.dumps(by_energy, indent=2)}")

    by_movement = {}
    for t in all_tropes:
        m = t.get("movement_association", "unknown")
        by_movement[m] = by_movement.get(m, 0) + 1
    print(f"\nMovement association: {json.dumps(by_movement, indent=2)}")

    output = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "n_tropes": len(all_tropes),
        "categories": [{"id": c[0], "name": c[1], "count": by_cat.get(c[0], 0)} for c in CATEGORIES],
        "tropes": all_tropes,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
