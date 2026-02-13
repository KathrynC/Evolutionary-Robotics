#!/usr/bin/env python3
"""
structured_random_bible.py

Structured random search condition #3: Random Bible verses (KJV).

Selects 100 random Bible verses spanning narrative, poetry, prophecy,
and epistle, asks a local LLM to translate each verse's imagery and
emotional quality into 6 synapse weights, then runs headless simulations
with Beer-framework analytics.

Usage:
    python3 structured_random_bible.py
"""

import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from structured_random_common import run_structured_search

OUT_JSON = PROJECT / "artifacts" / "structured_random_bible.json"

# ── Seed list: Bible verses (KJV, public domain) ────────────────────────────
# Curated for diversity: action, stillness, repetition, enumeration, metaphor,
# narrative, prophecy, wisdom, poetry, apocalyptic

VERSES = [
    # Genesis - beginnings and action
    "Genesis 1:3 — And God said, Let there be light: and there was light.",
    "Genesis 1:2 — And the earth was without form, and void; and darkness was upon the face of the deep.",
    "Genesis 3:19 — In the sweat of thy face shalt thou eat bread, till thou return unto the ground.",
    "Genesis 7:17 — And the flood was forty days upon the earth; and the waters increased.",
    "Genesis 11:9 — Therefore is the name of it called Babel; because the LORD did there confound the language.",
    "Genesis 28:12 — And he dreamed, and behold a ladder set up on the earth, and the top of it reached to heaven.",
    # Exodus
    "Exodus 3:2 — And the angel of the LORD appeared unto him in a flame of fire out of the midst of a bush.",
    "Exodus 14:21 — And the LORD caused the sea to go back by a strong east wind all that night.",
    "Exodus 15:20 — And Miriam took a timbrel in her hand; and all the women went out after her with dances.",
    # Psalms - poetry and emotion
    "Psalm 23:4 — Yea, though I walk through the valley of the shadow of death, I will fear no evil.",
    "Psalm 46:10 — Be still, and know that I am God.",
    "Psalm 42:7 — Deep calleth unto deep at the noise of thy waterspouts.",
    "Psalm 137:1 — By the rivers of Babylon, there we sat down, yea, we wept.",
    "Psalm 19:1 — The heavens declare the glory of God; and the firmament sheweth his handywork.",
    "Psalm 104:26 — There go the ships: there is that leviathan, whom thou hast made to play therein.",
    "Psalm 139:14 — I am fearfully and wonderfully made: marvellous are thy works.",
    "Psalm 90:4 — For a thousand years in thy sight are but as yesterday when it is past.",
    "Psalm 18:33 — He maketh my feet like hinds' feet, and setteth me upon my high places.",
    # Proverbs
    "Proverbs 6:6 — Go to the ant, thou sluggard; consider her ways, and be wise.",
    "Proverbs 30:19 — The way of an eagle in the air; the way of a serpent upon a rock.",
    "Proverbs 25:11 — A word fitly spoken is like apples of gold in pictures of silver.",
    # Ecclesiastes
    "Ecclesiastes 1:9 — There is no new thing under the sun.",
    "Ecclesiastes 3:1 — To every thing there is a season, and a time to every purpose under the heaven.",
    "Ecclesiastes 1:6 — The wind goeth toward the south, and turneth about unto the north; it whirleth about continually.",
    "Ecclesiastes 12:6 — Or ever the silver cord be loosed, or the golden bowl be broken.",
    # Isaiah - prophecy
    "Isaiah 40:31 — But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles.",
    "Isaiah 6:3 — Holy, holy, holy, is the LORD of hosts: the whole earth is full of his glory.",
    "Isaiah 55:12 — The mountains and the hills shall break forth before you into singing, and all the trees of the field shall clap their hands.",
    "Isaiah 2:4 — They shall beat their swords into plowshares, and their spears into pruninghooks.",
    "Isaiah 40:4 — Every valley shall be exalted, and every mountain and hill shall be made low.",
    # Job
    "Job 38:4 — Where wast thou when I laid the foundations of the earth?",
    "Job 41:1 — Canst thou draw out leviathan with an hook?",
    "Job 38:31 — Canst thou bind the sweet influences of Pleiades, or loose the bands of Orion?",
    # Ezekiel - vision
    "Ezekiel 37:4 — Prophesy upon these bones, and say unto them, O ye dry bones, hear the word of the LORD.",
    "Ezekiel 1:16 — Their appearance and their work was as it were a wheel in the middle of a wheel.",
    "Ezekiel 47:5 — Afterward he measured a thousand; and it was a river that I could not pass over.",
    # Daniel
    "Daniel 3:25 — Lo, I see four men loose, walking in the midst of the fire, and they have no hurt.",
    "Daniel 5:27 — TEKEL; Thou art weighed in the balances, and art found wanting.",
    # Song of Solomon
    "Song of Solomon 2:11 — For, lo, the winter is past, the rain is over and gone.",
    "Song of Solomon 8:6 — Love is strong as death; jealousy is cruel as the grave.",
    # Gospels - narrative
    "Matthew 14:25 — And in the fourth watch of the night Jesus went unto them, walking on the sea.",
    "Matthew 7:24 — A wise man, which built his house upon a rock.",
    "Mark 4:39 — And he arose, and rebuked the wind, and said unto the sea, Peace, be still.",
    "Luke 15:4 — What man of you, having an hundred sheep, if he lose one of them, doth not leave the ninety and nine.",
    "John 1:1 — In the beginning was the Word, and the Word was with God, and the Word was God.",
    "John 11:35 — Jesus wept.",
    "Matthew 6:28 — Consider the lilies of the field, how they grow; they toil not, neither do they spin.",
    "Luke 10:30 — A certain man went down from Jerusalem to Jericho, and fell among thieves.",
    "John 3:8 — The wind bloweth where it listeth, and thou hearest the sound thereof, but canst not tell whence it cometh.",
    # Acts
    "Acts 2:2 — And suddenly there came a sound from heaven as of a rushing mighty wind.",
    "Acts 9:3 — Suddenly there shined round about him a light from heaven.",
    # Romans and Epistles
    "Romans 8:28 — All things work together for good to them that love God.",
    "1 Corinthians 13:12 — For now we see through a glass, darkly; but then face to face.",
    "1 Corinthians 15:52 — In a moment, in the twinkling of an eye, at the last trump.",
    "Hebrews 11:1 — Now faith is the substance of things hoped for, the evidence of things not seen.",
    "James 1:17 — Every good gift and every perfect gift is from above, and cometh down from the Father of lights.",
    # Revelation - apocalyptic
    "Revelation 6:8 — And I looked, and behold a pale horse: and his name that sat on him was Death.",
    "Revelation 4:6 — Before the throne there was a sea of glass like unto crystal.",
    "Revelation 1:15 — And his feet like unto fine brass, as if they burned in a furnace.",
    "Revelation 21:1 — And I saw a new heaven and a new earth: for the first heaven and the first earth were passed away.",
    "Revelation 8:1 — And when he had opened the seventh seal, there was silence in heaven about the space of half an hour.",
    # Joshua
    "Joshua 6:20 — So the people shouted when the priests blew with the trumpets, and the wall fell down flat.",
    # Judges
    "Judges 5:20 — The stars in their courses fought against Sisera.",
    # Samuel and Kings
    "1 Samuel 17:49 — And David put his hand in his bag, and took thence a stone, and slang it.",
    "1 Kings 19:12 — And after the fire a still small voice.",
    "2 Kings 2:11 — And Elijah went up by a whirlwind into heaven.",
    # Numbers
    "Numbers 22:28 — And the LORD opened the mouth of the ass, and she said unto Balaam, What have I done unto thee?",
    # Jonah
    "Jonah 1:17 — Now the LORD had prepared a great fish to swallow up Jonah.",
    # Habakkuk
    "Habakkuk 3:19 — The LORD God is my strength, and he will make my feet like hinds' feet.",
    # Micah
    "Micah 6:8 — What doth the LORD require of thee, but to do justly, and to love mercy, and to walk humbly.",
    # Amos
    "Amos 5:24 — But let judgment run down as waters, and righteousness as a mighty stream.",
    # Nahum
    "Nahum 1:3 — The LORD hath his way in the whirlwind and in the storm.",
    # Zechariah
    "Zechariah 4:6 — Not by might, nor by power, but by my spirit, saith the LORD of hosts.",
    # Lamentations
    "Lamentations 3:22 — It is of the LORD's mercies that we are not consumed, because his compassions fail not.",
    # Ruth
    "Ruth 1:16 — Whither thou goest, I will go; and where thou lodgest, I will lodge.",
    # Deuteronomy
    "Deuteronomy 32:11 — As an eagle stirreth up her nest, fluttereth over her young, spreadeth abroad her wings.",
    # Additional diversity
    "Genesis 32:24 — And Jacob was left alone; and there wrestled a man with him until the breaking of the day.",
    "Exodus 33:22 — I will put thee in a clift of the rock, and will cover thee with my hand while I pass by.",
    "Isaiah 35:6 — Then shall the lame man leap as an hart, and the tongue of the dumb sing.",
    "Jeremiah 23:29 — Is not my word like as a fire? and like a hammer that breaketh the rock in pieces?",
    "Matthew 8:26 — Then he arose, and rebuked the winds and the sea; and there was a great calm.",
    "2 Samuel 22:34 — He maketh my feet like hinds' feet: and setteth me upon my high places.",
    "Psalm 114:4 — The mountains skipped like rams, and the little hills like lambs.",
    "Psalm 29:3 — The voice of the LORD is upon the waters.",
    "Isaiah 43:2 — When thou walkest through the fire, thou shalt not be burned.",
    "Ezekiel 10:13 — As for the wheels, it was cried unto them in my hearing, O wheel.",
    "Psalm 1:3 — He shall be like a tree planted by the rivers of water.",
    "Proverbs 30:29 — There be three things which go well, yea, four are comely in going.",
    "Job 39:19 — Hast thou given the horse strength? hast thou clothed his neck with thunder?",
    "Habakkuk 2:2 — Write the vision, and make it plain upon tables, that he may run that readeth it.",
    "Isaiah 30:21 — This is the way, walk ye in it, when ye turn to the right hand, and when ye turn to the left.",
    "Psalm 18:29 — For by thee I have run through a troop; and by my God have I leaped over a wall.",
    "Psalm 77:19 — Thy way is in the sea, and thy path in the great waters, and thy footsteps are not known.",
    "Ecclesiastes 9:11 — The race is not to the swift, nor the battle to the strong.",
    "Isaiah 52:7 — How beautiful upon the mountains are the feet of him that bringeth good tidings.",
    "Genesis 1:9 — Let the waters under the heaven be gathered together unto one place, and let the dry land appear.",
    "Psalm 93:4 — The LORD on high is mightier than the noise of many waters.",
    "Revelation 10:1 — And I saw another mighty angel come down from heaven, clothed with a cloud; and a rainbow was upon his head.",
    "Job 26:7 — He stretcheth out the north over the empty place, and hangeth the earth upon nothing.",
    "Isaiah 40:22 — It is he that sitteth upon the circle of the earth.",
    "Psalm 147:4 — He telleth the number of the stars; he calleth them all by their names.",
]


def make_prompt(verse):
    return (
        f"Generate 6 synapse weights for a 3-link walking robot given the verse: "
        f'"{verse}". The weights are w03, w04, w13, w14, w23, w24, each in [-1, 1]. '
        f"Translate the imagery, action, and emotional quality of this verse into "
        f"weight magnitudes, signs, and symmetry patterns. "
        f'Return ONLY a JSON object like '
        f'{{"w03": 0.5, "w04": -0.3, "w13": 0.1, "w14": -0.7, "w23": 0.4, "w24": -0.2}} '
        f"with no other text."
    )


def main():
    random.shuffle(VERSES)
    seeds = VERSES[:100]
    run_structured_search("bible", seeds, make_prompt, OUT_JSON)


if __name__ == "__main__":
    main()
