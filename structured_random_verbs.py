#!/usr/bin/env python3
"""
structured_random_verbs.py

Structured random search — Condition #1: Multilingual Verbs
============================================================

HYPOTHESIS
----------
Verbs encode action qualities — speed, direction, stability, rhythm — that
may map naturally onto locomotion parameters. A verb like "stumble" implies
asymmetric, unstable motion; "glide" implies smooth, effortless motion;
"oscillate" implies periodic, back-and-forth motion. The LLM should be able
to extract these kinematic qualities and encode them as weight patterns.

SEED DESIGN
-----------
150 verbs spanning 15+ languages (English, Spanish, French, German, Japanese,
Arabic, Mandarin, Russian, Hindi, Swahili, Portuguese, Korean, Turkish, Latin,
Greek). Each verb includes its language and English meaning to give the LLM
full context: "stolpern (German, to stumble)".

Two categories of English verbs are included:
  - Motion verbs: stumble, sprint, crawl, glide, lurch, waddle, stride, etc.
  - Non-motion verbs: shatter, bloom, ignite, dissolve, oscillate, cascade, etc.

The non-motion verbs test whether abstract action qualities (breaking apart,
coming together, vibrating) translate into locomotion as effectively as
explicit movement descriptions.

PROMPT STRATEGY
--------------
The prompt asks the LLM to translate "the action quality, intensity, and
movement character" of the verb into weight magnitudes, signs, and symmetry
patterns. This gives the LLM three orthogonal dimensions to encode:
  - Action quality → which sensors drive which motors (sign pattern)
  - Intensity → weight magnitudes
  - Movement character → symmetry/asymmetry between the two motors

KEY RESULTS (from 100-trial run)
---------------------------------
  Dead: 5% (vs 8% baseline)
  Median |DX|: 1.55m (vs 6.64m baseline — significantly lower)
  Max |DX|: 25.12m (from "fracture")
  Mean phase lock: 0.850 (vs 0.613 baseline — much more coordinated)

Notable: Stumble-synonyms across 4 languages (English, German, Portuguese,
Spanish) all mapped to identical weights → identical speed (2.010). The LLM
treats cross-linguistic synonyms as the same structural concept.

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
# Format: "verb (language, meaning)" to give the LLM context.
# Each entry provides the romanized verb, its source language, and an English
# gloss. The LLM uses all three to construct its weight mapping.
#
# The list is deliberately over-provisioned (150 verbs) so that random.shuffle
# + [:100] gives a different sample each run while maintaining language diversity.

VERBS = [
    # English - motion verbs (explicit locomotion descriptions)
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
    # English - non-motion verbs (abstract actions without explicit locomotion;
    # tests whether the LLM can map breaking, growing, vibrating, etc. into
    # locomotion parameters as effectively as explicit motion descriptions)
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
    """Build the LLM prompt for a given verb seed.

    The prompt explicitly names the 6 weights and their range [-1, 1], provides
    a concrete JSON example to anchor the output format, and asks the LLM to
    translate three aspects of the verb into weight properties:
      - Action quality → sign patterns (which sensors excite/inhibit which motors)
      - Intensity → magnitudes (how strongly each connection fires)
      - Movement character → symmetry patterns (are the two motors driven alike?)

    The instruction "Return ONLY a JSON object ... with no other text" minimizes
    parsing failures, though parse_weights() handles violations robustly.
    """
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
    # Shuffle and take 100 to get a random sample from the full 150-verb pool.
    # Different runs get different samples, providing some run-to-run variance.
    random.shuffle(VERBS)
    seeds = VERBS[:100]
    run_structured_search("verbs", seeds, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
