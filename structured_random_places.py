#!/usr/bin/env python3
"""
structured_random_places.py

Structured random search — Condition #4: Global Place Names
=============================================================

HYPOTHESIS
----------
Place names carry geographic and atmospheric associations — terrain type,
climate, energy, scale — that the LLM encodes from travel writing, geography
texts, cultural descriptions, and news. "Death Valley" evokes extreme heat
and flat desolation; "Mariana Trench" evokes crushing depth and darkness;
"Serengeti" evokes vast open plains with periodic migration.

This condition tests the weakest form of structural transfer: can the LLM
map a *name* (not even a description) of a geographic location into weight
patterns that produce locomotion? The place name is maximally indirect —
there is no action, no narrative, no mathematical structure, just a location
and whatever the LLM associates with it.

SEED DESIGN
-----------
114 places spanning all continents and terrain types:
  Cities (30):            Reykjavik, Mumbai, Kyoto, Marrakech, Venice, ...
  Mountains/peaks (10):   Everest, Kilimanjaro, Fuji, K2, ...
  Deserts (7):            Sahara, Atacama, Gobi, Namib, ...
  Water features (13):    Mariana Trench, Amazon, Victoria Falls, Baikal, ...
  Islands (10):           Madagascar, Borneo, Socotra, Galápagos, ...
  Forests (6):            Amazon Rainforest, Black Forest, Yakushima, ...
  Polar/extreme (6):      South Pole, North Pole, Oymyakon, Death Valley, ...
  Plains/steppes (5):     Mongolian Steppe, Serengeti, Great Plains, ...
  Geological features (8): Grand Canyon, Great Barrier Reef, Cappadocia, ...
  Cultural/historical (9): Angkor Wat, Machu Picchu, Stonehenge, ...

Each place includes its country/region for disambiguation and to provide the
LLM with additional geographic context.

PROMPT STRATEGY
--------------
The prompt asks the LLM to translate "the character of this place — its
terrain, climate, energy, rhythm, and physical quality" into weights. Five
dimensions:
  - Terrain → physical landscape type (flat, mountainous, aquatic)
  - Climate → temperature and weather patterns
  - Energy → how dynamic or static the place feels
  - Rhythm → seasonal, tidal, volcanic, or cultural cycles
  - Physical quality → hardness, fluidity, density, openness

KEY RESULTS (from 100-trial run)
---------------------------------
  Dead: 0% (zero dead gaits, like Bible)
  Median |DX|: 1.18m (lowest of all conditions)
  Max |DX|: 5.64m (Ulaanbaatar, Mongolia — the most constrained max)
  Mean phase lock: 0.884 (vs 0.613 baseline)

This is the most conservative condition: the LLM generates weights in the
tightest cluster, producing coordinated but uniformly modest gaits. Geographic
concepts translate into the safest, most central region of weight space.
The PCA diversity plot shows places (green) occupying the smallest area of
any condition in behavioral space.

Usage:
    python3 structured_random_places.py
"""

import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import run_structured_search

OUT_JSON = PROJECT / "artifacts" / "structured_random_places.json"

# ── Seed list: global place names ────────────────────────────────────────────
# Curated for maximum geographic diversity: all 7 continents, all major terrain
# types, extreme and moderate climates, natural and cultural sites, varying
# scales (from a single cave to an entire desert). Over-provisioned (114 places)
# so that shuffle + [:100] samples broadly.

PLACES = [
    # Cities — varied character, climate, energy, cultural density
    "Reykjavik, Iceland",
    "Mumbai, India",
    "Kyoto, Japan",
    "Marrakech, Morocco",
    "Venice, Italy",
    "Ulaanbaatar, Mongolia",
    "Havana, Cuba",
    "Bergen, Norway",
    "Cusco, Peru",
    "Dubrovnik, Croatia",
    "Zanzibar City, Tanzania",
    "Lhasa, Tibet",
    "Timbuktu, Mali",
    "Hanoi, Vietnam",
    "Bruges, Belgium",
    "Varanasi, India",
    "Petra, Jordan",
    "Fez, Morocco",
    "Cartagena, Colombia",
    "Samarkand, Uzbekistan",
    "Kathmandu, Nepal",
    "Irkutsk, Russia",
    "Tromsoe, Norway",
    "Oaxaca, Mexico",
    "Luang Prabang, Laos",
    "Aleppo, Syria",
    "Manaus, Brazil",
    "Svalbard, Norway",
    "Singapore",
    "Norilsk, Russia",
    # Mountains and peaks
    "Mount Everest",
    "Kilimanjaro, Tanzania",
    "Denali, Alaska",
    "Mount Fuji, Japan",
    "Matterhorn, Swiss Alps",
    "Cerro Torre, Patagonia",
    "K2, Karakoram",
    "Table Mountain, South Africa",
    "Mount Olympus, Greece",
    "Popocatepetl, Mexico",
    # Deserts
    "Sahara Desert",
    "Atacama Desert, Chile",
    "Gobi Desert, Mongolia",
    "Namib Desert, Namibia",
    "Rub al Khali, Arabian Peninsula",
    "White Sands, New Mexico",
    "Dasht-e Lut, Iran",
    # Water features
    "Mariana Trench",
    "Amazon River, Brazil",
    "Victoria Falls, Zimbabwe",
    "Dead Sea, Israel-Jordan",
    "Lake Baikal, Russia",
    "Iguazu Falls, Argentina-Brazil",
    "Niagara Falls",
    "Mekong Delta, Vietnam",
    "Congo River",
    "Ganges River Delta",
    "Sargasso Sea",
    "Lake Titicaca, Peru-Bolivia",
    "Strait of Magellan",
    # Islands
    "Madagascar",
    "Borneo",
    "Iceland",
    "Socotra, Yemen",
    "Easter Island, Chile",
    "Faroe Islands",
    "Svalbard",
    "Galapagos Islands",
    "Reunion Island",
    "Tasmania, Australia",
    # Forests and jungles
    "Amazon Rainforest",
    "Black Forest, Germany",
    "Bialowieza Forest, Poland",
    "Daintree Rainforest, Australia",
    "Tongass National Forest, Alaska",
    "Yakushima Forest, Japan",
    # Polar and extreme
    "South Pole",
    "North Pole",
    "McMurdo Station, Antarctica",
    "Oymyakon, Siberia",
    "Death Valley, California",
    "Danakil Depression, Ethiopia",
    # Plains and steppes
    "Mongolian Steppe",
    "Serengeti, Tanzania",
    "Great Plains, USA",
    "Pampas, Argentina",
    "Kazakh Steppe",
    # Geological features
    "Grand Canyon, Arizona",
    "Great Barrier Reef, Australia",
    "Giant's Causeway, Northern Ireland",
    "Cappadocia, Turkey",
    "Yellowstone Caldera",
    "Carlsbad Caverns, New Mexico",
    "Jeita Grotto, Lebanon",
    "Eisriesenwelt Ice Cave, Austria",
    # Cultural/historical
    "Angkor Wat, Cambodia",
    "Machu Picchu, Peru",
    "Great Wall of China",
    "Stonehenge, England",
    "Chichen Itza, Mexico",
    "Pompeii, Italy",
    "Bagan, Myanmar",
    "Lalibela, Ethiopia",
]


def make_prompt(place):
    """Build the LLM prompt for a given place name seed.

    The prompt provides five dimensions for the LLM to map from:
      - Terrain → landscape type (the physical surface the robot "walks on")
      - Climate → temperature/weather patterns (the environment's energy level)
      - Energy → how dynamic or static the place is (volcanic vs frozen)
      - Rhythm → cyclic patterns (tides, seasons, eruptions, migrations)
      - Physical quality → material properties (hard rock, soft sand, flowing water)

    Place names test the weakest structural transfer: the LLM must infer all
    of these qualities from a name alone, using its training-corpus associations.
    """
    return (
        f"Generate 6 synapse weights for a 3-link walking robot given the place: "
        f'"{place}". The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. '
        f"Translate the character of this place — its terrain, climate, energy, "
        f"rhythm, and physical quality — into weight magnitudes, signs, and patterns. "
        f'Return ONLY a JSON object like '
        f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
        f"with no other text."
    )


def main():
    random.shuffle(PLACES)
    seeds = PLACES[:100]  # sample 100 from 114 available
    run_structured_search("places", seeds, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
