#!/usr/bin/env python3
"""
structured_random_verbs.py

Structured random search condition #1: Random verbs from multiple languages.

Selects 100 random verbs (spanning 15+ languages), asks a local LLM to
translate each verb's action quality into 6 synapse weights, then runs
headless simulations with Beer-framework analytics.

Usage:
    python3 structured_random_verbs.py
"""

import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import run_structured_search

OUT_JSON = PROJECT / "artifacts" / "structured_random_verbs.json"

# ── Seed list: verbs across languages ────────────────────────────────────────
# Format: "verb (language, meaning)" to give the LLM context

VERBS = [
    # English - motion
    "stumble (English, to trip and nearly fall)",
    "sprint (English, to run at full speed)",
    "crawl (English, to move on hands and knees)",
    "glide (English, to move smoothly and effortlessly)",
    "lurch (English, to move with sudden unsteady movements)",
    "waddle (English, to walk with short steps swaying side to side)",
    "stride (English, to walk with long decisive steps)",
    "shuffle (English, to walk dragging the feet)",
    "tiptoe (English, to walk quietly on the toes)",
    "stagger (English, to walk unsteadily as if about to fall)",
    "leap (English, to jump a long distance)",
    "creep (English, to move slowly and carefully)",
    "dash (English, to move suddenly and quickly)",
    "wobble (English, to move unsteadily from side to side)",
    "skid (English, to slide sideways uncontrollably)",
    # English - non-motion
    "shatter (English, to break into many pieces)",
    "whisper (English, to speak very softly)",
    "bloom (English, to produce flowers)",
    "ignite (English, to catch fire)",
    "dissolve (English, to become incorporated into a liquid)",
    "oscillate (English, to move back and forth regularly)",
    "cascade (English, to fall in a series of stages)",
    "resonate (English, to vibrate in response to a frequency)",
    "fracture (English, to break without complete separation)",
    "coalesce (English, to come together and form one mass)",
    # Spanish
    "correr (Spanish, to run)",
    "bailar (Spanish, to dance)",
    "tropezar (Spanish, to stumble)",
    "arrastrarse (Spanish, to drag oneself)",
    "saltar (Spanish, to jump)",
    "tambalearse (Spanish, to sway or stagger)",
    "deslizarse (Spanish, to slide or glide)",
    "temblar (Spanish, to tremble)",
    # French
    "marcher (French, to walk)",
    "bondir (French, to leap or bound)",
    "ramper (French, to crawl or creep)",
    "tourbillonner (French, to whirl or spin)",
    "chanceler (French, to stagger or totter)",
    "flotter (French, to float)",
    "glisser (French, to slide or slip)",
    "sautiller (French, to hop or skip)",
    # German
    "schleichen (German, to sneak or creep)",
    "stolpern (German, to stumble)",
    "schwanken (German, to sway or waver)",
    "humpeln (German, to limp or hobble)",
    "rennen (German, to run fast)",
    "kriechen (German, to crawl)",
    "taumeln (German, to stagger or reel)",
    "schlurfen (German, to shuffle or scuff)",
    # Japanese
    "aruku (Japanese, to walk)",
    "hashiru (Japanese, to run)",
    "tobu (Japanese, to fly or jump)",
    "oyogu (Japanese, to swim)",
    "suberu (Japanese, to slide or slip)",
    "korobu (Japanese, to fall down or tumble)",
    "odoru (Japanese, to dance)",
    "yureru (Japanese, to sway or shake)",
    # Arabic
    "yamshi (Arabic, to walk)",
    "yarkud (Arabic, to run)",
    "yaqfiz (Arabic, to jump)",
    "yazhaf (Arabic, to crawl)",
    "yataraddad (Arabic, to hesitate)",
    "yadur (Arabic, to rotate or spin)",
    # Mandarin
    "zou (Mandarin, to walk)",
    "pao (Mandarin, to run)",
    "tiao (Mandarin, to jump or leap)",
    "pa (Mandarin, to crawl or climb)",
    "hua (Mandarin, to slide or ski)",
    "zhuan (Mandarin, to rotate or turn)",
    # Russian
    "begat (Russian, to run)",
    "polzti (Russian, to crawl)",
    "prygat (Russian, to jump)",
    "shatat'sya (Russian, to stagger)",
    "kachat'sya (Russian, to swing or rock)",
    "skatit'sya (Russian, to slide down)",
    # Hindi
    "chalna (Hindi, to walk)",
    "daudna (Hindi, to run)",
    "kudna (Hindi, to jump)",
    "girna (Hindi, to fall)",
    "nachna (Hindi, to dance)",
    # Swahili
    "kutembea (Swahili, to walk)",
    "kukimbia (Swahili, to run)",
    "kuruka (Swahili, to fly or jump)",
    "kutambaa (Swahili, to crawl)",
    "kucheza (Swahili, to dance or play)",
    # Portuguese
    "tropecar (Portuguese, to stumble)",
    "rastejar (Portuguese, to crawl)",
    "pairar (Portuguese, to hover)",
    "escorregar (Portuguese, to slip)",
    # Korean
    "geotda (Korean, to walk)",
    "ttwida (Korean, to run)",
    "gureuda (Korean, to drag)",
    "heundeullida (Korean, to sway or shake)",
    # Turkish
    "yuvarlanmak (Turkish, to roll)",
    "sendelemek (Turkish, to stagger)",
    "surunmek (Turkish, to crawl or drag)",
    "ziplamak (Turkish, to jump or bounce)",
    # Latin
    "ambulare (Latin, to walk)",
    "currere (Latin, to run)",
    "serpere (Latin, to creep or crawl)",
    "volare (Latin, to fly)",
    # Greek
    "peripatein (Greek, to walk about)",
    "trechein (Greek, to run)",
    "choreuo (Greek, to dance)",
]


def make_prompt(verb):
    return (
        f"Generate 6 synapse weights for a 3-link walking robot given the verb: "
        f"{verb}. The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. "
        f"Translate the action quality, intensity, and movement character of this "
        f"verb into weight magnitudes, signs, and symmetry patterns. "
        f'Return ONLY a JSON object like '
        f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
        f"with no other text."
    )


def main():
    random.shuffle(VERBS)
    seeds = VERBS[:100]
    run_structured_search("verbs", seeds, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
