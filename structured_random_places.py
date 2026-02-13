#!/usr/bin/env python3
"""
structured_random_places.py

Structured random search condition #4: Random place names from around the world.

Selects 100 random place names (cities, geographic features, regions)
spanning all continents and terrain types, asks a local LLM to translate
each place's character into 6 synapse weights, then runs headless
simulations with Beer-framework analytics.

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
# Diverse by: continent, terrain type, scale, climate, cultural character

PLACES = [
    # Cities - varied character
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
    seeds = PLACES[:100]
    run_structured_search("places", seeds, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
