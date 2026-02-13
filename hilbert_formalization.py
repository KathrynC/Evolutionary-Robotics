#!/usr/bin/env python3
"""
hilbert_formalization.py — Part C: Hilbert Space Formalization

C1. Trajectory L² inner product: Gram matrix of 116+ zoo gaits,
    eigendecomposition, spectral basis, project structured random gaits.

C2. RKHS kernel regression: Gaussian kernel on 500 atlas points,
    predict cliffiness at LLM-generated weight points, compare to KNN.

C3. Spectral analysis: eigenvalue spectra of behavioral covariance,
    spectral gaps, submanifold thickness.

Pure computation on existing data — no simulations required.

Output: artifacts/hilbert_formalization_results.json
        artifacts/plots/hf_fig01_gram_matrix.png
        artifacts/plots/hf_fig02_spectral_basis.png
        artifacts/plots/hf_fig03_rkhs_vs_knn.png
        artifacts/plots/hf_fig04_spectral_gaps.png
        artifacts/plots/hf_fig05_trajectory_projections.png
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
BEH_KEYS = ["dx", "speed", "efficiency", "phase_lock", "entropy",
             "roll_dom", "yaw_net_rad", "dy"]
OUT_PATH = PROJECT / "artifacts" / "hilbert_formalization_results.json"
PLOT_DIR = PROJECT / "artifacts" / "plots"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder handling numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 6)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════
# C1: TRAJECTORY L² HILBERT SPACE
# ═══════════════════════════════════════════════════════════════════════════

def load_zoo_trajectories():
    """Load joint angle trajectories from all zoo gaits with 4000-step telemetry.

    Returns:
        list of dicts: [{name, j0_pos[4000], j1_pos[4000], x[4000], y[4000], z[4000]}]
    """
    telemetry_dir = PROJECT / "artifacts" / "telemetry"
    gaits = []

    # Find all gait directories that have a 4000-line telemetry file
    for gait_dir in sorted(telemetry_dir.iterdir()):
        if not gait_dir.is_dir():
            continue
        # Check for telemetry.jsonl directly in the gait directory first,
        # then in subdirectories (two different directory layouts exist)
        best_file = None
        best_lines = 0

        # Direct file (named gaits like 1_original/)
        direct = gait_dir / "telemetry.jsonl"
        if direct.exists():
            with open(direct) as f:
                n = sum(1 for _ in f)
            if n == 4000:
                best_file = direct
                best_lines = n

        # Subdirectory files (numbered gaits like 000/syn_zoo_probe_*/telemetry.jsonl)
        if best_file is None:
            for sub in gait_dir.iterdir():
                if not sub.is_dir():
                    continue
                tfile = sub / "telemetry.jsonl"
                if tfile.exists():
                    with open(tfile) as f:
                        n = sum(1 for _ in f)
                    if n == 4000 and n > best_lines:
                        best_file = tfile
                        best_lines = n

        if best_file is None:
            continue

        # Read the telemetry
        j0_pos = np.empty(4000)
        j1_pos = np.empty(4000)
        x_arr = np.empty(4000)
        y_arr = np.empty(4000)
        z_arr = np.empty(4000)

        with open(best_file) as f:
            for idx, line in enumerate(f):
                if idx >= 4000:
                    break
                rec = json.loads(line)
                # Joint positions
                joints = rec.get("joints", [])
                j0_pos[idx] = joints[0]["pos"] if len(joints) > 0 else 0.0
                j1_pos[idx] = joints[1]["pos"] if len(joints) > 1 else 0.0
                # Base position
                base = rec.get("base", {})
                x_arr[idx] = base.get("x", 0.0)
                y_arr[idx] = base.get("y", 0.0)
                z_arr[idx] = base.get("z", 0.0)

        gaits.append({
            "name": gait_dir.name,
            "j0_pos": j0_pos,
            "j1_pos": j1_pos,
            "x": x_arr,
            "y": y_arr,
            "z": z_arr,
        })

    return gaits


def compute_gram_matrix(gaits, mode="joint_angles"):
    """Compute Gram matrix of L² inner products between gait trajectories.

    The inner product is: <g1, g2> = (1/T) ∫₀ᵀ g1(t)·g2(t) dt

    For joint_angles mode, each gait is the 2D trajectory [j0(t), j1(t)].
    For position mode, each gait is the 3D trajectory [x(t), y(t), z(t)].

    Args:
        gaits: list of gait dicts from load_zoo_trajectories()
        mode: "joint_angles" or "position"

    Returns:
        G: (N, N) Gram matrix
        names: list of gait names
    """
    N = len(gaits)
    dt = 1.0 / 4000  # normalize time to [0, 1]

    # Build trajectory matrix: N x T x D
    if mode == "joint_angles":
        D = 2
        traj = np.zeros((N, 4000, D))
        for i, g in enumerate(gaits):
            traj[i, :, 0] = g["j0_pos"]
            traj[i, :, 1] = g["j1_pos"]
    else:  # position
        D = 3
        traj = np.zeros((N, 4000, D))
        for i, g in enumerate(gaits):
            traj[i, :, 0] = g["x"]
            traj[i, :, 1] = g["y"]
            traj[i, :, 2] = g["z"]

    # Center trajectories (subtract mean over time for each gait)
    traj_centered = traj - traj.mean(axis=1, keepdims=True)

    # Gram matrix: G[i,j] = (1/T) Σ_t Σ_d traj_i(t,d) * traj_j(t,d)
    # Reshape to N x (T*D) for efficient computation
    flat = traj_centered.reshape(N, -1)
    G = (flat @ flat.T) * dt

    names = [g["name"] for g in gaits]
    return G, names


def spectral_decomposition(G):
    """Eigendecompose the Gram matrix.

    Returns:
        eigenvalues: sorted descending
        eigenvectors: columns are eigenvectors (N x N)
        participation_ratio: (Σλ)² / Σ(λ²)
    """
    eigvals, eigvecs = np.linalg.eigh(G)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Clip small negative eigenvalues (numerical noise)
    eigvals = np.maximum(eigvals, 0.0)

    total = np.sum(eigvals)
    if total > 0:
        pr = total**2 / np.sum(eigvals**2)
        cumvar = np.cumsum(eigvals) / total
    else:
        pr = 0.0
        cumvar = np.zeros_like(eigvals)

    return eigvals, eigvecs, float(pr), cumvar


# ═══════════════════════════════════════════════════════════════════════════
# C2: RKHS KERNEL REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def load_atlas_data():
    """Load atlas cliffiness data (500 probe points)."""
    path = PROJECT / "artifacts" / "atlas_cliffiness.json"
    if not path.exists():
        return None, None, None
    with open(path) as f:
        atlas = json.load(f)

    probes = atlas["probe_results"]
    N = len(probes)
    W = np.zeros((N, 6))
    c_vals = np.zeros(N)
    for i, p in enumerate(probes):
        for j, wn in enumerate(WEIGHT_NAMES):
            W[i, j] = p["weights"][wn]
        c_vals[i] = p["cliffiness"]
    return W, c_vals, probes


def load_llm_weights():
    """Load all unique LLM-generated weight vectors."""
    conditions = ["verbs", "theorems", "bible", "places"]
    unique = {}

    for cond in conditions:
        path = PROJECT / "artifacts" / f"structured_random_{cond}.json"
        if not path.exists():
            continue
        with open(path) as f:
            trials = json.load(f)
        for trial in trials:
            wt = trial["weights"]
            key = tuple(wt[k] for k in WEIGHT_NAMES)
            if key not in unique:
                unique[key] = {
                    "weights": np.array([wt[k] for k in WEIGHT_NAMES]),
                    "conditions": [cond],
                    "seed": trial["seed"],
                }
            else:
                if cond not in unique[key]["conditions"]:
                    unique[key]["conditions"].append(cond)

    return list(unique.values())


def gaussian_kernel(X1, X2, sigma):
    """Compute Gaussian (RBF) kernel matrix K[i,j] = exp(-||x1_i - x2_j||² / 2σ²)."""
    # Efficient pairwise distance computation
    sq1 = np.sum(X1**2, axis=1, keepdims=True)
    sq2 = np.sum(X2**2, axis=1, keepdims=True)
    dist_sq = sq1 + sq2.T - 2 * X1 @ X2.T
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.exp(-dist_sq / (2 * sigma**2))


def rkhs_regression(W_train, y_train, W_test, sigma, ridge=1e-4):
    """Kernel ridge regression in the RKHS.

    Args:
        W_train: (N_train, 6) training weight vectors
        y_train: (N_train,) training cliffiness values
        W_test: (N_test, 6) test weight vectors
        sigma: kernel bandwidth
        ridge: regularization parameter

    Returns:
        y_pred: (N_test,) predicted cliffiness
        alpha: (N_train,) dual coefficients
    """
    K = gaussian_kernel(W_train, W_train, sigma)
    K += ridge * np.eye(len(W_train))
    alpha = np.linalg.solve(K, y_train)
    K_test = gaussian_kernel(W_test, W_train, sigma)
    y_pred = K_test @ alpha
    return y_pred, alpha


def knn_interpolation(W_train, y_train, W_test, k=5):
    """KNN interpolation (inverse-distance weighting) for comparison."""
    dist_sq = (np.sum(W_train**2, axis=1, keepdims=True).T +
               np.sum(W_test**2, axis=1, keepdims=True) -
               2 * W_test @ W_train.T)
    dist_sq = np.maximum(dist_sq, 1e-20)
    dists = np.sqrt(dist_sq)

    y_pred = np.zeros(len(W_test))
    for i in range(len(W_test)):
        idx = np.argsort(dists[i])[:k]
        d = dists[i, idx]
        w = 1.0 / d
        w /= w.sum()
        y_pred[i] = np.sum(w * y_train[idx])
    return y_pred


def cross_validate_sigma(W, y, sigmas, k=5, ridge=1e-4):
    """K-fold cross-validation for kernel bandwidth selection."""
    N = len(W)
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = N // k
    errors = {s: [] for s in sigmas}

    for fold in range(k):
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size],
                                     indices[(fold + 1) * fold_size:]])
        for sigma in sigmas:
            y_pred, _ = rkhs_regression(W[train_idx], y[train_idx],
                                         W[test_idx], sigma, ridge)
            mse = np.mean((y_pred - y[test_idx])**2)
            errors[sigma].append(mse)

    return {s: np.mean(errs) for s, errs in errors.items()}


# ═══════════════════════════════════════════════════════════════════════════
# C3: SPECTRAL ANALYSIS OF BEHAVIORAL COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════

def load_structured_random_data():
    """Load all 5 conditions of structured random data."""
    conditions = ["verbs", "theorems", "bible", "places", "baseline"]
    data = {}
    for cond in conditions:
        path = PROJECT / "artifacts" / f"structured_random_{cond}.json"
        if path.exists():
            with open(path) as f:
                data[cond] = json.load(f)
    return data


def behavioral_spectral_analysis(data):
    """Compute behavioral covariance eigenspectra per condition.

    Returns dict with per-condition spectral data.
    """
    results = {}
    for cond, trials in data.items():
        N = len(trials)
        beh = np.zeros((N, len(BEH_KEYS)))
        for i, t in enumerate(trials):
            for j, k in enumerate(BEH_KEYS):
                beh[i, j] = t.get(k, 0.0)

        # Z-score normalize
        mu = beh.mean(axis=0)
        std = beh.std(axis=0)
        std[std < 1e-12] = 1.0
        beh_z = (beh - mu) / std

        # Covariance matrix
        cov = np.cov(beh_z.T)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        eigvals = np.maximum(eigvals, 0.0)

        total = np.sum(eigvals)
        if total > 0:
            pr = total**2 / np.sum(eigvals**2)
            cumvar = np.cumsum(eigvals) / total
        else:
            pr = 0.0
            cumvar = np.zeros_like(eigvals)

        # Spectral gap: ratio of 2nd to 3rd eigenvalue
        gap_2_3 = float(eigvals[1] / eigvals[2]) if eigvals[2] > 1e-12 else float("inf")
        gap_1_2 = float(eigvals[0] / eigvals[1]) if eigvals[1] > 1e-12 else float("inf")

        results[cond] = {
            "eigenvalues": eigvals.tolist(),
            "participation_ratio": float(pr),
            "cumulative_variance": cumvar.tolist(),
            "spectral_gap_1_2": gap_1_2,
            "spectral_gap_2_3": gap_2_3,
            "n_trials": N,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def generate_figures(gram_results, rkhs_results, spectral_results):
    """Generate all Hilbert formalization figures."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Fig 1: Gram matrix heatmap
    if gram_results:
        G = np.array(gram_results["gram_matrix_joint"])
        names = gram_results["gait_names"]
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(G, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"L² Gram Matrix of {len(names)} Zoo Gaits (Joint Angle Trajectories)")
        ax.set_xlabel("Gait Index")
        ax.set_ylabel("Gait Index")
        plt.colorbar(im, ax=ax, label="⟨g_i, g_j⟩_L²")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "hf_fig01_gram_matrix.png", dpi=300)
        plt.close(fig)
        print(f"  WROTE hf_fig01_gram_matrix.png")

    # Fig 2: Spectral basis (eigenvalue spectrum)
    if gram_results:
        eigvals_j = np.array(gram_results["eigenvalues_joint"])
        eigvals_p = np.array(gram_results["eigenvalues_position"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.semilogy(np.arange(1, len(eigvals_j)+1), eigvals_j + 1e-30, "b.-")
        ax1.set_xlabel("Mode Number")
        ax1.set_ylabel("Eigenvalue")
        ax1.set_title("Joint Angle L² Spectrum")
        ax1.axhline(y=eigvals_j[0]*0.01, color="r", ls="--", alpha=0.5, label="1% threshold")
        ax1.legend()

        ax2.semilogy(np.arange(1, len(eigvals_p)+1), eigvals_p + 1e-30, "r.-")
        ax2.set_xlabel("Mode Number")
        ax2.set_ylabel("Eigenvalue")
        ax2.set_title("Position L² Spectrum")
        ax2.axhline(y=eigvals_p[0]*0.01, color="b", ls="--", alpha=0.5, label="1% threshold")
        ax2.legend()

        fig.suptitle("Gait Space Spectral Decomposition", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "hf_fig02_spectral_basis.png", dpi=300)
        plt.close(fig)
        print(f"  WROTE hf_fig02_spectral_basis.png")

    # Fig 3: RKHS vs KNN prediction
    if rkhs_results and "rkhs_pred" in rkhs_results:
        rkhs_pred = np.array(rkhs_results["rkhs_pred"])
        knn_pred = np.array(rkhs_results["knn_pred"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(knn_pred, rkhs_pred, alpha=0.6, s=30, c="steelblue")
        lims = [min(knn_pred.min(), rkhs_pred.min()),
                max(knn_pred.max(), rkhs_pred.max())]
        ax1.plot(lims, lims, "k--", alpha=0.5)
        ax1.set_xlabel("KNN Interpolated Cliffiness")
        ax1.set_ylabel("RKHS Predicted Cliffiness")
        ax1.set_title(f"RKHS vs KNN (r={rkhs_results.get('correlation', 0):.3f})")

        # Residuals
        residuals = rkhs_pred - knn_pred
        ax2.hist(residuals, bins=20, alpha=0.7, color="steelblue")
        ax2.axvline(0, color="k", ls="--")
        ax2.set_xlabel("RKHS - KNN Residual")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Residuals (mean={np.mean(residuals):.3f}, std={np.std(residuals):.3f})")

        fig.suptitle("Kernel Ridge Regression in RKHS of Cliffiness", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "hf_fig03_rkhs_vs_knn.png", dpi=300)
        plt.close(fig)
        print(f"  WROTE hf_fig03_rkhs_vs_knn.png")

    # Fig 4: Behavioral spectral gaps per condition
    if spectral_results:
        conditions = ["verbs", "theorems", "bible", "places", "baseline"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = {"verbs": "#e41a1c", "theorems": "#377eb8", "bible": "#4daf4a",
                  "places": "#984ea3", "baseline": "#aaaaaa"}

        for cond in conditions:
            if cond not in spectral_results:
                continue
            eigvals = np.array(spectral_results[cond]["eigenvalues"])
            cumvar = np.array(spectral_results[cond]["cumulative_variance"])
            ax1.semilogy(range(1, len(eigvals)+1), eigvals + 1e-30,
                         ".-", color=colors[cond], label=cond, alpha=0.8)
            ax2.plot(range(1, len(cumvar)+1), cumvar,
                     ".-", color=colors[cond], label=cond, alpha=0.8)

        ax1.set_xlabel("Eigenvalue Index")
        ax1.set_ylabel("Eigenvalue (log scale)")
        ax1.set_title("Behavioral Covariance Spectrum")
        ax1.legend()

        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Variance Explained")
        ax2.set_title("Cumulative Variance")
        ax2.axhline(0.95, color="gray", ls="--", alpha=0.5, label="95%")
        ax2.legend()

        fig.suptitle("Spectral Analysis: LLM Conditions vs Baseline", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "hf_fig04_spectral_gaps.png", dpi=300)
        plt.close(fig)
        print(f"  WROTE hf_fig04_spectral_gaps.png")

    # Fig 5: Trajectory projections onto top eigenmodes
    if gram_results and "projections" in gram_results:
        projs = gram_results["projections"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (pair, title) in zip(axes, [
            ((0, 1), "Mode 1 vs Mode 2"),
            ((0, 2), "Mode 1 vs Mode 3"),
            ((1, 2), "Mode 2 vs Mode 3"),
        ]):
            i, j = pair
            coords = np.array(projs["coordinates"])
            ax.scatter(coords[:, i], coords[:, j], alpha=0.5, s=20, c="steelblue")
            ax.set_xlabel(f"Mode {i+1}")
            ax.set_ylabel(f"Mode {j+1}")
            ax.set_title(title)

        fig.suptitle(f"Zoo Gaits in L² Spectral Basis ({len(projs['names'])} gaits)", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "hf_fig05_trajectory_projections.png", dpi=300)
        plt.close(fig)
        print(f"  WROTE hf_fig05_trajectory_projections.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    results = {}

    print("=" * 60)
    print("HILBERT SPACE FORMALIZATION")
    print("=" * 60)

    # ── C1: Trajectory L² ─────────────────────────────────────────────
    print("\n── C1: Trajectory L² Hilbert Space ──")
    gaits = load_zoo_trajectories()
    print(f"  Loaded {len(gaits)} gaits with 4000-step telemetry")

    if len(gaits) >= 10:
        print("  Computing Gram matrices...")
        G_joint, names = compute_gram_matrix(gaits, mode="joint_angles")
        G_position, _ = compute_gram_matrix(gaits, mode="position")

        print("  Eigendecomposing...")
        eigvals_j, eigvecs_j, pr_j, cumvar_j = spectral_decomposition(G_joint)
        eigvals_p, eigvecs_p, pr_p, cumvar_p = spectral_decomposition(G_position)

        # How many modes for 95% variance
        n95_j = int(np.searchsorted(cumvar_j, 0.95) + 1) if cumvar_j[-1] >= 0.95 else len(cumvar_j)
        n95_p = int(np.searchsorted(cumvar_p, 0.95) + 1) if cumvar_p[-1] >= 0.95 else len(cumvar_p)

        print(f"  Joint angle L²:")
        print(f"    Participation ratio: {pr_j:.2f}")
        print(f"    Modes for 95% variance: {n95_j}")
        print(f"    Top 5 eigenvalues: {eigvals_j[:5].tolist()}")
        print(f"  Position L²:")
        print(f"    Participation ratio: {pr_p:.2f}")
        print(f"    Modes for 95% variance: {n95_p}")
        print(f"    Top 5 eigenvalues: {eigvals_p[:5].tolist()}")

        # Project gaits onto spectral basis
        projections_j = eigvecs_j  # Each row is a gait's coordinates in eigenbasis

        gram_results = {
            "n_gaits": len(gaits),
            "gait_names": names,
            "gram_matrix_joint": G_joint.tolist(),
            "eigenvalues_joint": eigvals_j.tolist(),
            "eigenvalues_position": eigvals_p.tolist(),
            "participation_ratio_joint": pr_j,
            "participation_ratio_position": pr_p,
            "n_modes_95pct_joint": n95_j,
            "n_modes_95pct_position": n95_p,
            "cumvar_joint": cumvar_j.tolist(),
            "cumvar_position": cumvar_p.tolist(),
            "projections": {
                "names": names,
                "coordinates": projections_j[:, :10].tolist(),  # top 10 modes
            },
        }
        results["trajectory_L2"] = gram_results
    else:
        gram_results = None
        print("  WARNING: Too few gaits with telemetry for Gram matrix")

    # ── C2: RKHS Kernel Regression ────────────────────────────────────
    print("\n── C2: RKHS Kernel Regression ──")
    W_atlas, c_atlas, _ = load_atlas_data()

    if W_atlas is not None:
        print(f"  Atlas: {len(W_atlas)} points with cliffiness data")

        # Cross-validate sigma
        sigmas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        print(f"  Cross-validating kernel bandwidth (σ from {sigmas})...")
        cv_errors = cross_validate_sigma(W_atlas, c_atlas, sigmas)
        best_sigma = min(cv_errors, key=cv_errors.get)
        print(f"  Best σ = {best_sigma} (CV MSE = {cv_errors[best_sigma]:.4f})")

        # Load LLM weight vectors
        llm_entries = load_llm_weights()
        W_llm = np.array([e["weights"] for e in llm_entries])
        print(f"  LLM points: {len(W_llm)} unique weight vectors")

        # RKHS prediction at LLM points
        rkhs_pred, alpha = rkhs_regression(W_atlas, c_atlas, W_llm, best_sigma)

        # KNN prediction for comparison
        knn_pred = knn_interpolation(W_atlas, c_atlas, W_llm, k=5)

        # Correlation between RKHS and KNN
        r_corr = np.corrcoef(rkhs_pred, knn_pred)[0, 1]
        print(f"  RKHS vs KNN correlation: r={r_corr:.4f}")
        print(f"  RKHS predicted cliffiness: mean={np.mean(rkhs_pred):.4f}, "
              f"median={np.median(rkhs_pred):.4f}")
        print(f"  KNN predicted cliffiness: mean={np.mean(knn_pred):.4f}, "
              f"median={np.median(knn_pred):.4f}")

        # Compare LLM to atlas distribution
        atlas_median = np.median(c_atlas)
        n_below_rkhs = np.sum(rkhs_pred < atlas_median)
        n_below_knn = np.sum(knn_pred < atlas_median)
        print(f"  Atlas median cliffiness: {atlas_median:.4f}")
        print(f"  LLM below atlas median (RKHS): {n_below_rkhs}/{len(rkhs_pred)} "
              f"({100*n_below_rkhs/len(rkhs_pred):.0f}%)")
        print(f"  LLM below atlas median (KNN):  {n_below_knn}/{len(knn_pred)} "
              f"({100*n_below_knn/len(knn_pred):.0f}%)")

        # RKHS norm: ||c||²_H = α^T K α — measures function smoothness
        K_train = gaussian_kernel(W_atlas, W_atlas, best_sigma)
        rkhs_norm_sq = float(alpha @ K_train @ alpha)
        print(f"  RKHS norm² of cliffiness function: {rkhs_norm_sq:.2f}")

        # Per-condition RKHS predictions
        print("\n  Per-condition RKHS cliffiness:")
        for cond in ["verbs", "theorems", "bible", "places"]:
            mask = [cond in e["conditions"] for e in llm_entries]
            if any(mask):
                cond_pred = rkhs_pred[mask]
                print(f"    {cond}: mean={np.mean(cond_pred):.4f}, "
                      f"median={np.median(cond_pred):.4f}, n={len(cond_pred)}")

        rkhs_results = {
            "best_sigma": float(best_sigma),
            "cv_errors": {str(s): float(e) for s, e in cv_errors.items()},
            "rkhs_pred": rkhs_pred.tolist(),
            "knn_pred": knn_pred.tolist(),
            "correlation": float(r_corr),
            "atlas_median": float(atlas_median),
            "fraction_below_median_rkhs": float(n_below_rkhs / len(rkhs_pred)),
            "fraction_below_median_knn": float(n_below_knn / len(knn_pred)),
            "rkhs_norm_sq": rkhs_norm_sq,
            "n_llm_points": len(W_llm),
            "n_atlas_points": len(W_atlas),
        }
        results["rkhs"] = rkhs_results
    else:
        rkhs_results = None
        print("  WARNING: Atlas data not found")

    # ── C3: Spectral Analysis ─────────────────────────────────────────
    print("\n── C3: Behavioral Covariance Spectral Analysis ──")
    sr_data = load_structured_random_data()
    spectral_results = behavioral_spectral_analysis(sr_data)

    print(f"\n  Condition | PR  | Gap 1→2 | Gap 2→3 | 95% modes")
    print(f"  {'-'*52}")
    for cond in ["verbs", "theorems", "bible", "places", "baseline"]:
        if cond in spectral_results:
            s = spectral_results[cond]
            pr = s["participation_ratio"]
            g12 = s["spectral_gap_1_2"]
            g23 = s["spectral_gap_2_3"]
            cumvar = np.array(s["cumulative_variance"])
            n95 = int(np.searchsorted(cumvar, 0.95) + 1) if cumvar[-1] >= 0.95 else len(cumvar)
            print(f"  {cond:10s} | {pr:.1f} | {g12:7.1f} | {g23:7.1f} | {n95}")

    results["spectral_analysis"] = spectral_results

    # ── Figures ───────────────────────────────────────────────────────
    print("\n── Generating Figures ──")
    generate_figures(gram_results, rkhs_results, spectral_results)

    # ── Save ──────────────────────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  WROTE {OUT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HILBERT FORMALIZATION SUMMARY")
    print("=" * 60)
    if gram_results:
        print(f"  C1 Trajectory L²:")
        print(f"    {gram_results['n_gaits']} gaits, PR(joint)={gram_results['participation_ratio_joint']:.1f}, "
              f"PR(position)={gram_results['participation_ratio_position']:.1f}")
        print(f"    95% variance in {gram_results['n_modes_95pct_joint']} joint modes, "
              f"{gram_results['n_modes_95pct_position']} position modes")
    if rkhs_results:
        print(f"  C2 RKHS:")
        print(f"    σ={rkhs_results['best_sigma']}, RKHS vs KNN r={rkhs_results['correlation']:.3f}")
        print(f"    LLM points below atlas median: "
              f"{rkhs_results['fraction_below_median_rkhs']*100:.0f}% (RKHS), "
              f"{rkhs_results['fraction_below_median_knn']*100:.0f}% (KNN)")
        print(f"    ||c||²_H = {rkhs_results['rkhs_norm_sq']:.1f}")
    print(f"  C3 Spectral gaps:")
    for cond in ["verbs", "theorems", "bible", "places", "baseline"]:
        if cond in spectral_results:
            s = spectral_results[cond]
            print(f"    {cond}: PR={s['participation_ratio']:.1f}, "
                  f"gap₁₂={s['spectral_gap_1_2']:.1f}, "
                  f"gap₂₃={s['spectral_gap_2_3']:.1f}")


if __name__ == "__main__":
    main()
