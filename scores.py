import numpy as np
from cryocat import tmana
from cryocat import cryomap
import os

suffix = "8"

results_folder = "./tm_outputs_" + suffix
scores = cryomap.read(results_folder + "/scores_0_1.mrc")
angles = cryomap.read(results_folder + "/angles_0_1.mrc")
angle_list = "./angle_list_" + suffix + ".txt"

# More aggressive diagnostic
print("=== Detailed Score Distribution ===")
print(f"Top 10 scores: {np.sort(scores.flatten())[-10:]}")
print(f"Bottom 10 scores: {np.sort(scores.flatten())[:10]}")
print(f"\nPercentiles:")
for pct in [50, 75, 90, 95, 99]:
    val = np.percentile(scores, pct)
    print(f"  {pct}th percentile: {val:.6f}")

# Try extraction with threshold at 90th percentile
percentile_90 = np.percentile(scores, 90)
print(f"\n=== Trying extraction with 90th percentile as threshold ({percentile_90:.6f}) ===")

tmana.scores_extract_particles(
    scores_map = scores,
    angles_map = angles,
    angles_list = angle_list,
    tomo_id = 1,
    particle_diameter = 31,
    object_id=None,
    scores_threshold = percentile_90,
    sigma_threshold=None,
    cluster_size=None,
    n_particles=None,
    output_path=results_folder + "/particle_list.em",
    output_type="emmotl",
    angles_order="zxz",
    symmetry="c1",
    angles_numbering=0,
)

output_file = results_folder + "/particle_list.em"
if os.path.exists(output_file):
    print("✓ particle_list.em created successfully!")
    size = os.path.getsize(output_file)
    print(f"  File size: {size} bytes")
else:
    print("✗ Still no file created")

from cryocat import cryomotl

motl = cryomotl.Motl.load(output_file)
print("Motive list shape:", motl.df.shape)


import matplotlib.pyplot as plt

# Create the histogram
plt.hist(motl.df['score'], bins=100, edgecolor='black') # 'bins' controls the number of bars, 'edgecolor' adds borders

# Add labels and title for clarity
import os
# Ensure the plots directory exists
plots_dir = "./plots"
os.makedirs(plots_dir, exist_ok=True)
label = "score"

# Histogram of scores
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Histogram of Particle Scores")
histogram_path = os.path.join(plots_dir, f"{label}_histogram_{suffix}.png")
plt.savefig(histogram_path)
plt.clf()

# Create a histogram and get the bin edges
hist, bin_edges = np.histogram(motl.df['score'], bins=50, density=True)

# Calculate the cumulative sum of the histogram
cdf_hist = 1 - np.cumsum(hist * np.diff(bin_edges))

# Plotting the cumulative histogram
plt.plot(bin_edges[1:], cdf_hist, drawstyle='steps-post')
plt.xlabel('Score')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Histogram (Approximation of CDF)')
plt.grid(True)
cdf_path = os.path.join(plots_dir, f"{label}_cdf_{suffix}.png")
plt.savefig(cdf_path)
plt.clf()

# ===== HEATMAP VISUALIZATIONS =====
print(f"\n=== Creating heatmap visualizations ===")
print(f"Scores volume shape: {scores.shape}")

# 1. Maximum Intensity Projection (MIP) along Z-axis - shows best score at each XY position
mip_z = np.max(scores, axis=0)  # Max along Z (first axis)
plt.figure(figsize=(12, 10))
plt.imshow(mip_z, cmap='hot', interpolation='nearest')
plt.colorbar(label='Score')
plt.title(f'Template Matching Scores - Maximum Intensity Projection (Z-axis)\nSuffix: {suffix}')
plt.xlabel('X')
plt.ylabel('Y')
mip_path = os.path.join(plots_dir, f"score_mip_z_{suffix}.png")
plt.savefig(mip_path, dpi=150, bbox_inches='tight')
plt.clf()
print(f"✓ Saved MIP (Z): {mip_path}")

# 2. Central Z-slice heatmap - shows a single slice through the middle
z_center = scores.shape[0] // 2
plt.figure(figsize=(12, 10))
plt.imshow(scores[z_center, :, :], cmap='hot', interpolation='nearest')
plt.colorbar(label='Score')
plt.title(f'Template Matching Scores - Central Z-slice (z={z_center})\nSuffix: {suffix}')
plt.xlabel('X')
plt.ylabel('Y')
slice_path = os.path.join(plots_dir, f"score_slice_z{z_center}_{suffix}.png")
plt.savefig(slice_path, dpi=150, bbox_inches='tight')
plt.clf()
print(f"✓ Saved central slice: {slice_path}")

# 3. MIP along X and Y axes for different views
mip_x = np.max(scores, axis=2)  # Max along X
mip_y = np.max(scores, axis=1)  # Max along Y

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Z-projection (XY view)
im0 = axes[0].imshow(mip_z, cmap='hot', interpolation='nearest')
axes[0].set_title('XY view (MIP along Z)')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im0, ax=axes[0], label='Score')

# X-projection (ZY view)
im1 = axes[1].imshow(mip_x, cmap='hot', interpolation='nearest', aspect='auto')
axes[1].set_title('ZY view (MIP along X)')
axes[1].set_xlabel('Y')
axes[1].set_ylabel('Z')
plt.colorbar(im1, ax=axes[1], label='Score')

# Y-projection (ZX view)
im2 = axes[2].imshow(mip_y, cmap='hot', interpolation='nearest', aspect='auto')
axes[2].set_title('ZX view (MIP along Y)')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Z')
plt.colorbar(im2, ax=axes[2], label='Score')

plt.suptitle(f'Template Matching Scores - All Projections (Suffix: {suffix})', fontsize=14)
plt.tight_layout()
multi_mip_path = os.path.join(plots_dir, f"score_mip_all_{suffix}.png")
plt.savefig(multi_mip_path, dpi=150, bbox_inches='tight')
plt.clf()
print(f"✓ Saved multi-view MIP: {multi_mip_path}")

# 4. Overlay detected particles on the MIP
if motl.df.shape[0] > 0:
    plt.figure(figsize=(12, 10))
    plt.imshow(mip_z, cmap='gray', interpolation='nearest')
    
    # Extract particle positions and scores
    x_coords = motl.df['x'].values
    y_coords = motl.df['y'].values
    particle_scores = motl.df['score'].values
    
    # Plot particles as scatter points, colored by score
    scatter = plt.scatter(x_coords, y_coords, c=particle_scores, 
                         cmap='hot', s=50, alpha=0.7, edgecolors='cyan', linewidth=1)
    plt.colorbar(scatter, label='Particle Score')
    plt.title(f'Detected Particles Overlay (n={len(motl.df)})\nSuffix: {suffix}')
    plt.xlabel('X')
    plt.ylabel('Y')
    overlay_path = os.path.join(plots_dir, f"particles_overlay_{suffix}.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.clf()
    print(f"✓ Saved particle overlay: {overlay_path}")
    
print(f"\n✓ All heatmaps saved to {plots_dir}/")
