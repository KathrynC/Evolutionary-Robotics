#!/usr/bin/env python3
"""
curate_stith_thompson.py

Use a local LLM to build a curated list of ~200 Stith Thompson folk-literature
motifs with structured metadata, spanning all 23 top-level categories.

The LLM's knowledge of the Thompson Motif-Index is extensive since it's a
foundational reference in folklore studies, widely discussed in academic literature.

Output: artifacts/stith_thompson_curated.json
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


# All 23 Thompson top-level categories with target counts
CATEGORIES = [
    ("A", "Mythological Motifs", 10,
     "Creator, creation, world elements, gods, demigods, culture heroes. "
     "Examples: A1 Creator, A15 Human creator, A100 God as creator, A700 Creation of sky, "
     "A1010 Deluge, A1415 Theft of fire."),

    ("B", "Animals", 10,
     "Mythical animals, speaking animals, helpful animals, animal marriages. "
     "Examples: B11 Dragon, B211 Speaking horse, B300 Helpful animal, B600 Animal marriage, "
     "B100 Treasure animal, B500 Magic animal."),

    ("C", "Tabu", 10,
     "Forbidden acts — looking, eating, speaking, touching. Breaking tabu. "
     "Examples: C30 Tabu: offending supernatural, C200 Eating tabu, C300 Looking tabu, "
     "C400 Speaking tabu, C900 Punishment for breaking tabu, C610 Forbidden chamber."),

    ("D", "Magic", 15,
     "Transformation, enchantment, magic objects, magic powers. "
     "Examples: D10 Transformation to animal, D100 Transformation to object, "
     "D700 Disenchantment, D800 Magic object, D1000 Magic power, D1400 Magic object overcomes."),

    ("E", "The Dead", 10,
     "Resuscitation, ghosts, revenants, the underworld. "
     "Examples: E1 Person resuscitated, E200 Ghost, E400 Ghost appearances, "
     "E500 Phantom hosts, E700 Soul, E750 Perils of the soul."),

    ("F", "Marvels", 10,
     "Fairies, otherworld, marvelous creatures, extraordinary places. "
     "Examples: F0 Otherworld journeys, F200 Fairies, F400 Spirits, F500 Remarkable persons, "
     "F700 Extraordinary places, F900 Extraordinary events."),

    ("G", "Ogres", 8,
     "Witches, devils, cannibals, giants, monsters. "
     "Examples: G10 Cannibalism, G200 Witches, G300 Giants, G500 Ogre defeated, "
     "G400 Person falls into ogre's power."),

    ("H", "Tests", 10,
     "Recognition, identity tests, riddles, tasks, quests. "
     "Examples: H0 Identity tests, H300 Tests of prowess, H500 Tests of cleverness, "
     "H900 Tasks, H1200 Quest, H1300 Quest for the best."),

    ("J", "The Wise and the Foolish", 10,
     "Wisdom, cleverness, foolishness, absurdity, choice. "
     "Examples: J0 Acquisition of wisdom, J200 Choices, J800 Fools, "
     "J1100 Cleverness, J1700 Fools, J2000 Absurd actions."),

    ("K", "Deceptions", 10,
     "Tricksters, disguises, escapes by deception, capture by deception. "
     "Examples: K0 Contests won by deception, K300 Thefts and cheats, "
     "K500 Escape by deception, K800 Killing by deception, K1800 Deception by disguise."),

    ("L", "Reversal of Fortune", 8,
     "Victorious youngest, unpromising hero, modesty rewarded. "
     "Examples: L0 Victorious youngest child, L100 Unpromising hero, "
     "L200 Modesty rewarded, L400 Pride brought low."),

    ("M", "Ordaining the Future", 8,
     "Bargains, vows, prophecies, fate, curses. "
     "Examples: M0 Bargains and promises, M200 Bargains with devil, "
     "M300 Prophecies, M400 Curses."),

    ("N", "Chance and Fate", 8,
     "Luck, gambling, accidents, treasure found by chance. "
     "Examples: N0 Wagers, N100 Nature of luck, N300 Unlucky accidents, "
     "N500 Treasure found by chance."),

    ("P", "Society", 8,
     "Royalty, nobility, warriors, customs, social classes. "
     "Examples: P0 Royalty, P200 The family, P300 Social classes, "
     "P400 Trades and professions, P500 Government."),

    ("Q", "Rewards and Punishments", 10,
     "Deeds rewarded, deeds punished, divine justice. "
     "Examples: Q0 Deeds rewarded, Q200 Deeds punished, Q400 Kinds of punishment, "
     "Q500 Tedious punishment, Q550 Mysterious punishments."),

    ("R", "Captives and Fugitives", 8,
     "Abduction, rescue, captivity, escape, pursuit. "
     "Examples: R0 Abduction, R100 Rescue, R200 Escape, R300 Refuge."),

    ("S", "Unnatural Cruelty", 8,
     "Murder, mutilation, abandonment, cruel persecution. "
     "Examples: S0 Cruel relatives, S100 Cruel persecutors, S200 Cruel sacrifices, "
     "S300 Abandoned children."),

    ("T", "Sex", 8,
     "Love, courtship, marriage, conception, birth. "
     "Examples: T0 Love, T100 Marriage, T200 Married life, T300 Chastity, "
     "T500 Conception and birth."),

    ("U", "The Nature of Life", 5,
     "Justice, truth, the nature of the world. "
     "Examples: U0 Justice and injustice, U100 The nature of life, U200 Ingratitude."),

    ("V", "Religion", 5,
     "Worship, saints, religious orders, religious objects. "
     "Examples: V0 Religious services, V200 Saints, V300 Religious orders, V400 Religious objects."),

    ("W", "Traits of Character", 8,
     "Favorable traits (generosity, patience) and unfavorable (greed, cruelty). "
     "Examples: W0 Favorable traits, W100 Unfavorable traits, W200 Humor of cleverness."),

    ("Z", "Miscellaneous", 5,
     "Formulae, symbolism, humor, heroes, cumulative tales. "
     "Examples: Z0 Formulae, Z100 Symbolism, Z200 Heroes, Z300 Unique exceptions."),
]

MOTIF_PROMPT = """\
List exactly {count} notable motifs from the Stith Thompson Motif-Index of Folk-Literature,
category {cat_id}: {cat_name}.

{cat_description}

Select motifs that are:
- Vivid and evocative (strong imagery or action)
- Well-known in folklore scholarship
- Diverse within the category (don't cluster in one subcategory)
- Mix of specific (e.g., D1421.1.3 "Magic horn which provides drink") and general (e.g., D10 "Transformation to animal")

For each motif, provide:
- "motif_id": the Thompson classification number (e.g., "B211.1", "D1421.1.3")
- "description": the motif description as Thompson wrote it (short phrase)
- "movement_quality": what kind of physical movement this motif evokes (e.g., "flight", "stillness", "struggle", "creeping", "running", "falling", "none")
- "energy": one of "explosive", "high", "moderate", "low", "still"
- "involves_death": true/false — does this motif typically involve death?
- "involves_transformation": true/false — does this motif involve changing form or state?

Use real Thompson motif IDs where you can recall them. If you cannot recall the exact ID,
use a plausible ID following the Thompson numbering scheme (letter + numbers + optional decimals).

Output ONLY a JSON array. No other text."""


def generate_motifs_for_category(cat_id, cat_name, count, cat_desc):
    prompt = MOTIF_PROMPT.format(
        count=count, cat_id=cat_id, cat_name=cat_name, cat_description=cat_desc
    )
    print(f"  [{cat_id}] {cat_name} ({count} motifs)...", end=" ", flush=True)
    resp = query_llm(prompt, max_tokens=8000)
    if resp is None:
        print("LLM ERROR")
        return []
    parsed = parse_json_response(resp)
    if parsed is None:
        print("PARSE FAIL")
        return []
    print(f"got {len(parsed)}")
    for entry in parsed:
        entry["category"] = cat_id
        entry["category_name"] = cat_name
    return parsed


def main():
    out_path = PROJECT / "artifacts" / "stith_thompson_curated.json"

    print("Curating Stith Thompson Motif-Index\n")

    all_motifs = []
    for cat_id, cat_name, count, cat_desc in CATEGORIES:
        motifs = generate_motifs_for_category(cat_id, cat_name, count, cat_desc)
        all_motifs.extend(motifs)

    # Deduplicate by motif_id
    seen = set()
    unique = []
    for m in all_motifs:
        mid = m.get("motif_id", "")
        if mid not in seen:
            seen.add(mid)
            unique.append(m)
    all_motifs = unique

    # Summary
    print(f"\n{'='*60}")
    print(f"STITH THOMPSON CURATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total motifs: {len(all_motifs)}")

    by_cat = {}
    for m in all_motifs:
        c = m.get("category", "?")
        by_cat[c] = by_cat.get(c, 0) + 1
    for c in sorted(by_cat.keys()):
        print(f"  {c}: {by_cat[c]}")

    death_count = sum(1 for m in all_motifs if m.get("involves_death"))
    transform_count = sum(1 for m in all_motifs if m.get("involves_transformation"))
    print(f"\nInvolves death: {death_count}")
    print(f"Involves transformation: {transform_count}")

    by_energy = {}
    for m in all_motifs:
        e = m.get("energy", "unknown")
        by_energy[e] = by_energy.get(e, 0) + 1
    print(f"\nEnergy: {json.dumps(by_energy, indent=2)}")

    output = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "n_motifs": len(all_motifs),
        "categories": [{"id": c[0], "name": c[1], "count": by_cat.get(c[0], 0)} for c in CATEGORIES],
        "motifs": all_motifs,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
