#!/usr/bin/env python3
"""
persona_cluster_analysis.py

Load all 4 persona weight sources (celebrities, fictional characters,
politicians, mathematicians), assign categories, cluster in 6D weight
space via k-means, project to 2D via PCA, and output a curated JSON
for the persona hypergraph visualization.

Output: artifacts/persona_hypergraph_curated.json

Usage:
    python3 persona_cluster_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT / "artifacts"

# ── Data sources ─────────────────────────────────────────────────────────────

SOURCES = {
    "celebrities": ARTIFACTS / "structured_random_celebrities.json",
    "fictional": ARTIFACTS / "structured_random_archetypometrics.json",
    "politics": ARTIFACTS / "structured_random_politics.json",
    "mathematicians": ARTIFACTS / "structured_random_mathematicians.json",
}

OUT_JSON = ARTIFACTS / "persona_hypergraph_curated.json"

WEIGHT_KEYS = ["w03", "w04", "w13", "w14", "w23", "w24"]

# ── Category assignment ──────────────────────────────────────────────────────
# Celebrity domains → viz categories

CELEBRITY_CATEGORY_MAP = {
    "musician": "musician",
    "sports": "sports",
    "entertainment": "entertainment",
    "kardashian": "entertainment",
    "tech": "entertainment",
    "cultural": "thinker",
    "historical": "thinker",
    "trump_family": "politician",
    "trump_admin": "politician",
    "us_politics": "politician",
    "international": "politician",
    "controversial": "politician",
}


def extract_name_source(seed):
    """Parse 'Name [source]' format, returning (name, source)."""
    if " [" in seed and seed.endswith("]"):
        idx = seed.index(" [")
        return seed[:idx], seed[idx + 2:-1]
    return seed, ""


def assign_celebrity_category(source):
    """Map celebrity domain tag to visualization category."""
    return CELEBRITY_CATEGORY_MAP.get(source, "entertainment")


# ── Notable names (recognizable to general audience) ─────────────────────────
# These get larger dots and labels in the visualization

NOTABLE_NAMES = {
    # Politicians
    "Donald Trump", "Barack Obama", "Joe Biden", "Hillary Clinton",
    "Vladimir Putin", "Angela Merkel", "Boris Johnson", "Bernie Sanders",
    "Volodymyr Zelensky", "Xi Jinping", "Nancy Pelosi", "AOC",
    # Entertainment
    "Oprah Winfrey", "Leonardo DiCaprio", "Tom Hanks", "Keanu Reeves",
    "Will Smith", "Meryl Streep", "Tom Cruise", "Arnold Schwarzenegger",
    "Kim Kardashian", "Kylie Jenner", "Elon Musk", "Mark Zuckerberg",
    "Bill Gates", "Steve Jobs", "Jeff Bezos",
    # Musicians
    "Beyonce", "Taylor Swift", "Kanye West", "Rihanna", "Lady Gaga",
    "Drake", "Madonna", "Eminem", "Adele", "BTS",
    # Sports
    "LeBron James", "Cristiano Ronaldo", "Lionel Messi", "Serena Williams",
    "Michael Jordan", "Tom Brady", "Tiger Woods", "Usain Bolt",
    # Thinkers
    "Albert Einstein", "Charles Darwin", "William Shakespeare",
    "Stephen King", "JK Rowling", "Noam Chomsky",
    "Mahatma Gandhi", "Nelson Mandela", "Martin Luther King",
    "Napoleon Bonaparte", "Abraham Lincoln", "Cleopatra", "Winston Churchill",
    # Mathematicians
    "Euclid", "Archimedes", "Pythagoras", "Isaac Newton", "Leonhard Euler",
    "Carl Friedrich Gauss", "Alan Turing", "John von Neumann",
    "Srinivasa Ramanujan", "Emmy Noether", "Hypatia", "Ada Lovelace",
    "Kurt Gödel", "Bernhard Riemann",
    # Fictional (most recognizable)
    "Harry Potter", "Gandalf", "Darth Vader", "Batman", "Superman",
    "Spider-Man", "Sherlock Holmes", "Frodo Baggins", "Luke Skywalker",
    "Princess Leia", "Hermione Granger", "Aragorn", "Katniss Everdeen",
    "James Bond", "Lara Croft", "Mario", "Pikachu", "Sonic",
    "Mickey Mouse", "Bugs Bunny", "Homer Simpson", "SpongeBob",
    "Optimus Prime", "Wolverine", "Captain America", "Iron Man",
    "Thor", "Thanos", "The Joker", "Hannibal Lecter",
    "Elizabeth Bennet", "Jay Gatsby", "Atticus Finch", "Holden Caulfield",
    "Don Quixote", "Odysseus", "Hamlet", "Romeo",
    "Elsa", "Simba", "Shrek", "Gollum",
}


# ── K-means (numpy-only) ────────────────────────────────────────────────────

def kmeans(X, k, max_iter=300, seed=42):
    """Lloyd's algorithm, numpy-only."""
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = np.array([
            X[labels == i].mean(0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_c):
            break
        centroids = new_c
    return labels, centroids


def silhouette_score(X, labels):
    """Simplified silhouette score (numpy-only, sampled for speed)."""
    n = len(X)
    k = labels.max() + 1
    # Sample up to 500 points for speed
    if n > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 500, replace=False)
        X_s, labels_s = X[idx], labels[idx]
    else:
        X_s, labels_s = X, labels

    scores = np.zeros(len(X_s))
    for i in range(len(X_s)):
        cl = labels_s[i]
        mask_same = labels_s == cl
        mask_same[i] = False
        if mask_same.sum() == 0:
            scores[i] = 0
            continue
        a = np.mean(np.linalg.norm(X_s[mask_same] - X_s[i], axis=1))
        b_min = np.inf
        for c in range(k):
            if c == cl:
                continue
            mask_c = labels_s == c
            if mask_c.sum() == 0:
                continue
            b = np.mean(np.linalg.norm(X_s[mask_c] - X_s[i], axis=1))
            b_min = min(b_min, b)
        if b_min == np.inf:
            scores[i] = 0
        else:
            scores[i] = (b_min - a) / max(a, b_min)
    return scores.mean()


# ── PCA (numpy power iteration) ─────────────────────────────────────────────

def pca_2d(X):
    """Project to 2D via covariance eigendecomposition (numpy-only)."""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order, we want descending
    idx = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, idx[:2]]
    return X_centered @ top2


# ── Cluster naming ───────────────────────────────────────────────────────────

CLUSTER_NAME_TEMPLATES = [
    "Stable Striders", "Chaotic Explorers", "Balanced Plodders",
    "Fast Launchers", "Symmetric Walkers", "Asymmetric Drifters",
    "Torso-Driven", "BackLeg-Dominant", "FrontLeg-Led",
    "Excitatory Burst", "Inhibitory Glide", "Mixed Oscillators",
    "Phase-Locked", "Desynchronized", "High-Entropy Wanderers",
    "Dead Inert", "Power Walkers", "Gentle Amblers",
]


def name_cluster(centroid, members_data, idx):
    """Generate a descriptive cluster name from its centroid signature."""
    sign = "".join("+" if v >= 0 else "-" for v in centroid)

    # Characterize by dominant metric
    speeds = [m.get("sp", 0) for m in members_data]
    dxs = [abs(m.get("dx", 0)) for m in members_data]
    pls = [m.get("pl", 0) for m in members_data]
    mean_speed = np.mean(speeds) if speeds else 0
    mean_dx = np.mean(dxs) if dxs else 0
    mean_pl = np.mean(pls) if pls else 0

    # Pick a behavioral descriptor
    if mean_dx > 3.0:
        prefix = "Fast"
    elif mean_dx < 0.5:
        prefix = "Static"
    elif mean_pl > 0.95:
        prefix = "Phase-Locked"
    elif mean_speed > 0.5:
        prefix = "Energetic"
    else:
        prefix = "Moderate"

    # Pick a weight-structure descriptor
    pos_count = sum(1 for v in centroid if v >= 0)
    if pos_count >= 5:
        suffix = "Exciters"
    elif pos_count <= 1:
        suffix = "Inhibitors"
    elif abs(centroid[0]) + abs(centroid[1]) > abs(centroid[2]) + abs(centroid[3]):
        suffix = "Torso-Led"
    else:
        suffix = "Limb-Driven"

    return f"{prefix} {suffix}"


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    entries = []

    # 1. Load celebrities
    with open(SOURCES["celebrities"]) as f:
        celeb_data = json.load(f)
    print(f"Loaded {len(celeb_data)} celebrities")

    for d in celeb_data:
        name, source = extract_name_source(d["seed"])
        cat = assign_celebrity_category(source)
        entries.append({
            "n": name,
            "s": source,
            "c": cat,
            "w": [d["weights"][k] for k in WEIGHT_KEYS],
            "dx": d["dx"],
            "sp": d["speed"],
            "pl": d["phase_lock"],
            "ef": d["efficiency"],
            "en": d["entropy"],
        })

    # 2. Load fictional characters
    with open(SOURCES["fictional"]) as f:
        fiction_data = json.load(f)
    print(f"Loaded {len(fiction_data)} fictional characters")

    for d in fiction_data:
        name, source = extract_name_source(d["seed"])
        entries.append({
            "n": name,
            "s": source,
            "c": "fictional",
            "w": [d["weights"][k] for k in WEIGHT_KEYS],
            "dx": d["dx"],
            "sp": d["speed"],
            "pl": d["phase_lock"],
            "ef": d["efficiency"],
            "en": d["entropy"],
        })

    # 3. Load politicians
    with open(SOURCES["politics"]) as f:
        politics_data = json.load(f)
    print(f"Loaded {len(politics_data)} politicians")

    for d in politics_data:
        name, source = extract_name_source(d["seed"])
        entries.append({
            "n": name,
            "s": source,
            "c": "politician",
            "w": [d["weights"][k] for k in WEIGHT_KEYS],
            "dx": d["dx"],
            "sp": d["speed"],
            "pl": d["phase_lock"],
            "ef": d["efficiency"],
            "en": d["entropy"],
        })

    # 4. Load mathematicians
    with open(SOURCES["mathematicians"]) as f:
        math_data = json.load(f)
    print(f"Loaded {len(math_data)} mathematicians")

    for d in math_data:
        name, source = extract_name_source(d["seed"])
        entries.append({
            "n": name,
            "s": source,
            "c": "thinker",
            "w": [d["weights"][k] for k in WEIGHT_KEYS],
            "dx": d["dx"],
            "sp": d["speed"],
            "pl": d["phase_lock"],
            "ef": d["efficiency"],
            "en": d["entropy"],
        })

    print(f"\nTotal entries: {len(entries)}")

    # Category counts
    cats = {}
    for e in entries:
        cats[e["c"]] = cats.get(e["c"], 0) + 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:15s}: {n:5d}")

    # Mark notable entries
    for e in entries:
        e["notable"] = e["n"] in NOTABLE_NAMES

    notable_count = sum(1 for e in entries if e["notable"])
    print(f"\nNotable entries: {notable_count}")

    # 5. Extract weight matrix
    X = np.array([e["w"] for e in entries])
    print(f"\nWeight matrix shape: {X.shape}")

    # 6. K-means clustering — sweep k from 12 to 18, pick best silhouette
    print("\nClustering (k-means, sweeping k=12..18)...")
    best_k, best_score, best_labels, best_centroids = 12, -1, None, None
    for k in range(12, 19):
        labels, centroids = kmeans(X, k)
        score = silhouette_score(X, labels)
        sizes = [int((labels == i).sum()) for i in range(k)]
        print(f"  k={k:2d}  silhouette={score:.4f}  sizes={sorted(sizes, reverse=True)[:6]}...")
        if score > best_score:
            best_k, best_score = k, score
            best_labels, best_centroids = labels, centroids

    print(f"\nBest k={best_k} (silhouette={best_score:.4f})")

    # 7. PCA projection to 2D
    print("PCA projection to 2D...")
    pca_coords = pca_2d(X)

    # Assign PCA coords to entries
    for i, e in enumerate(entries):
        e["px"] = round(float(pca_coords[i, 0]), 4)
        e["py"] = round(float(pca_coords[i, 1]), 4)
        e["k"] = int(best_labels[i])

    # 8. Build cluster metadata
    clusters = []
    for ci in range(best_k):
        mask = best_labels == ci
        members = [entries[i] for i in range(len(entries)) if mask[i]]
        centroid = best_centroids[ci]
        sign = "".join("+" if v >= 0 else "-" for v in centroid)

        # Find famous members (notable first, then by |dx|)
        notable_members = [m for m in members if m.get("notable")]
        famous = [m["n"] for m in sorted(notable_members, key=lambda m: -abs(m["dx"]))[:3]]
        if len(famous) < 3:
            non_notable = [m for m in members if not m.get("notable")]
            famous += [m["n"] for m in sorted(non_notable, key=lambda m: -abs(m["dx"]))[:3 - len(famous)]]

        label = name_cluster(centroid, members, ci)

        clusters.append({
            "id": ci,
            "label": label,
            "sign": sign,
            "centroid": [round(float(v), 4) for v in centroid],
            "n": int(mask.sum()),
            "mean_speed": round(float(np.mean([m["sp"] for m in members])), 4),
            "mean_phase_lock": round(float(np.mean([m["pl"] for m in members])), 4),
            "mean_dx": round(float(np.mean([abs(m["dx"]) for m in members])), 4),
            "famous": famous,
        })

    # Print cluster summary
    print("\nCluster Summary:")
    for cl in sorted(clusters, key=lambda c: -c["n"]):
        print(f"  [{cl['id']:2d}] {cl['label']:25s}  n={cl['n']:4d}  "
              f"sign={cl['sign']}  |DX|={cl['mean_dx']:.2f}  "
              f"famous: {', '.join(cl['famous'][:3])}")

    # 9. Round weight values to save space
    for e in entries:
        e["w"] = [round(v, 4) for v in e["w"]]
        e["dx"] = round(e["dx"], 3)
        e["sp"] = round(e["sp"], 4)
        e["pl"] = round(e["pl"], 4)
        e["ef"] = round(e["ef"], 6)
        e["en"] = round(e["en"], 4)

    # 10. Remove the 'notable' field from output (will recompute in JS)
    for e in entries:
        del e["notable"]

    # 11. Write output
    output = {
        "clusters": clusters,
        "entries": entries,
        "notable_names": sorted(NOTABLE_NAMES),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, separators=(",", ":"))  # compact

    size_kb = OUT_JSON.stat().st_size / 1024
    print(f"\nWROTE {OUT_JSON} ({size_kb:.0f} KB)")
    print(f"  {len(entries)} entries, {len(clusters)} clusters")


if __name__ == "__main__":
    main()
