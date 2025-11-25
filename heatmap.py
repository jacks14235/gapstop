"""
Create heatmap visualizations of z-scored GAPStop template-matching maps.

This script reads a `scores_*.mrc` volume produced by GAPStop, converts it
voxel-wise to z-scores, and saves several 2D heatmaps (MIP and slices).

Usage (from /home/ubuntu/gapstop):

    python heatmap.py --suffix 8

This expects files like:
    ./tm_outputs_8/scores_0_1.mrc
and writes PNGs into:
    ./plots/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cryocat import cryomap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate z-score heatmaps from GAPStop scores maps.",
    )
    parser.add_argument(
        "--suffix",
        default="8",
        help="Suffix used in tm_outputs_<suffix> and angle_list_<suffix>.txt (default: 8).",
    )
    parser.add_argument(
        "--index-i",
        type=int,
        default=0,
        help="First index in scores_i_j.mrc (default: 0).",
    )
    parser.add_argument(
        "--index-j",
        type=int,
        default=1,
        help="Second index in scores_i_j.mrc (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where PNGs will be written (default: plots).",
    )
    return parser.parse_args()


def compute_z_scores(scores: np.ndarray) -> np.ndarray:
    """Compute voxel-wise z-scores from a scores volume."""
    mu = np.mean(scores)
    sigma = np.std(scores)
    if sigma == 0:
        raise ValueError("Standard deviation of scores is zero; cannot compute z-scores.")
    z = (scores - mu) / sigma
    print(f"Score mean = {mu:.6f}, std = {sigma:.6f}")
    return z


def main() -> None:
    args = parse_args()

    base_dir = Path(".").resolve()
    tm_dir = base_dir / f"tm_outputs_{args.suffix}"
    scores_path = tm_dir / f"scores_{args.index_i}_{args.index_j}.mrc"

    if not scores_path.exists():
        raise FileNotFoundError(f"Scores volume not found: {scores_path}")

    print(f"Reading scores from {scores_path}")
    scores = cryomap.read(str(scores_path))
    print(f"Scores volume shape: {scores.shape}")

    print("Converting scores to z-scores (global mean/std)...")
    zmap = compute_z_scores(scores)

    plots_dir = base_dir / args.output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. MIP along Z
    mip_z = np.max(zmap, axis=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(mip_z, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="z-score")
    plt.title(f"Z-score MIP (max over Z)\nSuffix: {args.suffix}")
    plt.xlabel("X")
    plt.ylabel("Y")
    mip_path = plots_dir / f"zscore_mip_z_{args.suffix}.png"
    plt.savefig(mip_path, dpi=150, bbox_inches="tight")
    plt.clf()
    print(f"✓ Saved Z-score MIP (Z) to {mip_path}")

    # 2. Central Z slice
    z_center = zmap.shape[0] // 2
    plt.figure(figsize=(10, 8))
    plt.imshow(zmap[z_center, :, :], cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="z-score")
    plt.title(f"Z-score central slice (z={z_center})\nSuffix: {args.suffix}")
    plt.xlabel("X")
    plt.ylabel("Y")
    slice_path = plots_dir / f"zscore_slice_z{z_center}_{args.suffix}.png"
    plt.savefig(slice_path, dpi=150, bbox_inches="tight")
    plt.clf()
    print(f"✓ Saved Z-score central slice to {slice_path}")

    # 3. Orthogonal MIPs (Z, X, Y views)
    mip_x = np.max(zmap, axis=2)  # over X
    mip_y = np.max(zmap, axis=1)  # over Y

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(mip_z, cmap="coolwarm", interpolation="nearest")
    axes[0].set_title("XY view (MIP over Z)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im0, ax=axes[0], label="z-score")

    im1 = axes[1].imshow(mip_x, cmap="coolwarm", interpolation="nearest", aspect="auto")
    axes[1].set_title("ZY view (MIP over X)")
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("Z")
    plt.colorbar(im1, ax=axes[1], label="z-score")

    im2 = axes[2].imshow(mip_y, cmap="coolwarm", interpolation="nearest", aspect="auto")
    axes[2].set_title("ZX view (MIP over Y)")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Z")
    plt.colorbar(im2, ax=axes[2], label="z-score")

    plt.suptitle(f"Z-score MIPs (suffix: {args.suffix})", fontsize=14)
    plt.tight_layout()
    multi_path = plots_dir / f"zscore_mip_all_{args.suffix}.png"
    plt.savefig(multi_path, dpi=150, bbox_inches="tight")
    plt.clf()
    print(f"✓ Saved Z-score multi-view MIPs to {multi_path}")

    print(f"\nAll z-score heatmaps written to: {plots_dir}")


if __name__ == "__main__":
    main()


