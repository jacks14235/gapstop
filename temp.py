# Add this before running gapstop
# export XLA_FLAGS='--xla_gpu_strict_conv_algorithm_picker=false'
# export TF_FORCE_GPU_ALLOW_GROWTH=true

import os

import mrcfile
mrc_path = './TS_0002.mrc'

if not os.path.exists(mrc_path):
  raise Exception('File not found')

with mrcfile.open(mrc_path) as mrc:
  data = mrc.data
  print("Pixel size", mrc.voxel_size)

print(data.shape)

import cryocat
print("cryoCAT imported successfully!")

import numpy as np
from cryocat import pana
from cryocat import cryomask
from cryocat import geom
from cryocat import wedgeutils
from cryocat import tmana
from cryocat import cryomap

import warnings
warnings.filterwarnings('ignore')

import generate_angles_stochastic

inputs_folder_path = './'
angle_list_path = './'
template_list = './Cryo-EM/inputs/template_list.csv' # is this needed??
wedge_path = './'  # not directly used in this tutorial but has to be set
results_folder = './tm_outputs_0'        # to evaluate your own results change it to the './inputs/tm_outputs/'

folder_path = './'


angles_10 = geom.generate_angles(360, 24, symmetry=13)
print(angles_10.shape)

angles = generate_angles_stochastic.generate_angles_stochastic(360, 4, 8, symmetry=13)
print(angles.shape)
np.savetxt(folder_path + '/angle_list_C13_24.txt', angles_10, fmt='%.2f', delimiter=',')


import mrcfile
import numpy as np
from scipy.ndimage import zoom

# --------------------------------------------------
# User settings
# --------------------------------------------------
template_path = "./emd_14325.mrc"      # input template
tomo_path     = "./TS_0002.mrc"  # input tomogram
out_template_path = "./emd_14325_6A.mrc"
out_tomo_path     = "./TS_0002_6A.mrc"

TARGET_PIXEL_SIZE = 6.0   # Å/pixel (recommended)
# --------------------------------------------------

def resample_volume(in_path, out_path, target_px):
    with mrcfile.open(in_path, permissive=True) as m:
        vol = np.copy(m.data)
        orig_px = float(m.voxel_size.x)

    scale = orig_px / target_px
    print(f"{in_path}: original px = {orig_px:.3f} → scale = {scale:.3f}")

    # resample (isotropic)
    vol_resampled = zoom(vol, scale, order=1)

    with mrcfile.new(out_path, overwrite=True) as out:
        out.set_data(vol_resampled.astype(np.float32))
        out.voxel_size = target_px  # update header voxel size

    print(f"Saved {out_path}, new shape = {vol_resampled.shape}")

# --------------------------------------------------
# Resample template and tomogram
# --------------------------------------------------

# resample_volume(template_path, out_template_path, TARGET_PIXEL_SIZE)
# resample_volume(tomo_path, out_tomo_path, TARGET_PIXEL_SIZE)

import mrcfile
import numpy as np

input_template = "emd_14325_6A.mrc"   # your already-downsampled 6 Å map
output_template = "emd_14325_6A_crop192.mrc"
box = 192  # target box size in voxels

with mrcfile.open(input_template, permissive=True) as m:
    data = m.data.copy()
    voxel_size = float(m.voxel_size.x)  # Å/voxel, assuming isotropic
    print("Original shape:", data.shape)

# assume Z,Y,X order
nz, ny, nx = data.shape
cz, cy, cx = np.array(data.shape) // 2  # center indices

half = box // 2

z0, z1 = cz - half, cz + half
y0, y1 = cy - half, cy + half
x0, x1 = cx - half, cx + half

# sanity check: ensure we’re inside bounds
for a0, a1, n in [(z0, z1, nz), (y0, y1, ny), (x0, x1, nx)]:
    if a0 < 0 or a1 > n:
        raise ValueError("Requested crop box is out of bounds; choose a smaller box or recenter.")

data_crop = data[z0:z1, y0:y1, x0:x1]
print("Cropped shape:", data_crop.shape)

with mrcfile.new(output_template, overwrite=True) as out:
    out.set_data(data_crop.astype(np.float32))
    out.voxel_size = voxel_size  # keep correct Å/voxel metadata

print("✅ Saved", output_template)


_ = cryomask.spherical_mask(
    mask_size=192, #resized.shape[0],
    center=None,
    gaussian=2,
    gaussian_outwards=False,
    output_name="./test_mask.em"
)


cryocat.cryomap.mrc2em("emd_14325_6A_crop192.mrc", invert=False, overwrite=True, output_name=None)


from pathlib import Path
import shutil

def distribute_template_to_tomos(base_dir, template_path, overwrite=False):
    """
    Copies the template .em file into each tomogram subfolder.

    Parameters
    ----------
    base_dir : str or Path
        Path to the folder containing tomogram subdirectories (e.g., TS_026, TS_027, etc.)
    template_path : str or Path
        Path to the EMD template file (e.g., emd_14426.em)
    overwrite : bool, default=False
        If True, overwrites existing files in the subfolders
    """
    base_dir = Path(base_dir)
    template_path = Path(template_path)

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    tomogram_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    print(f"Found {len(tomogram_dirs)} tomogram folders in {base_dir}")

    for tomo_dir in tomogram_dirs:
        dest_path = tomo_dir / template_path.name
        if dest_path.exists() and not overwrite:
            print(f"⚠️ Skipping {tomo_dir.name}: file already exists.")
            continue

        shutil.copy2(template_path, dest_path)
        print(f"✅ Copied {template_path.name} → {tomo_dir.name}/")

    print("\n🎉 Done! Template distributed to all tomogram folders.")


# Example usage
# distribute_template_to_tomos(
#     base_dir="/content",
#     template_path="/content/test_mask.em",
#     overwrite=True  # set True if you want to replace existing copies
# )


import os
import mrcfile
import pandas as pd
from cryocat import wedgeutils

# === USER CONFIGURATION ===
root_base = "./"  # Folder with all tomogram subfolders
pixel_size = 13.48
voltage = 300.0
amp_contrast = 0.07
cs = 2.7

# Template, mask, and angle list paths
template_name = "emd_14426.em"       # 80S ribosome template
mask_name = "test_mask.em"    # Cylindrical/tight ribosome mask
angle_list = "angle_list_stoch.txt"

# STAR file parameters
vol_ext = ".rec"
symmetry = "C1"
anglist_order = "zxz"
smap_name = "scores"
omap_name = "angles"
lp_rad = 16
hp_rad = 1
binning = 1
tiling = "new"

# === OUTPUT STAR FILE ===
output_star = os.path.join(root_base, "data_input.star")

# === SCAN TOMOGRAMS ===
tomograms = sorted([
    d for d in os.listdir(root_base)
    if os.path.isdir(os.path.join(root_base, d)) and d.startswith("TS_")
])

# === HELPER: Get tomogram dimensions ===
def get_tomo_dimensions(tomo_path):
    """Return tomogram dimensions (x, y, z) as a list."""
    with mrcfile.open(tomo_path, permissive=True) as mrc:
        shape = mrc.data.shape
    # MRC shape is (z, y, x)
    return [int(shape[2]), int(shape[1]), int(shape[0])]

# === CREATE WEDGE LISTS AND DATA_INPUT.STAR ===
with open(output_star, "w") as f:
    f.write("data_\n\n")
    f.write("loop_\n")
    f.write("_rootdir\n_outputdir\n_vol_ext\n_tomo_name\n_tomo_num\n_wedgelist_name\n")
    f.write("_tmpl_name\n_mask_name\n_symmetry\n_anglist_order\n_anglist_name\n")
    f.write("_smap_name\n_omap_name\n_lp_rad\n_hp_rad\n_binning\n_tiling\n\n")

    for idx, tomo in enumerate(tomograms, start=1):
        tomo_dir = os.path.join(root_base, tomo)
        output_dir = os.path.join(tomo_dir, "tm_outputs")
        os.makedirs(output_dir, exist_ok=True)

        # --- Detect tomogram file (.rec or .mrc) ---
        tomo_file = os.path.join(tomo_dir, f"{tomo}.rec")
        if not os.path.exists(tomo_file):
            tomo_file = os.path.join(tomo_dir, f"{tomo}.mrc")

        # --- Detect tilt file (.mrc.mdoc) ---
        tlt_file = os.path.join(tomo_dir, f"{tomo}.mrc.mdoc")
        if not os.path.exists(tlt_file):
            raise FileNotFoundError(f"Tilt file not found for {tomo}: expected {tomo}.mrc.mdoc")

        # --- Optional metadata ---
        ctf_file = os.path.join(tomo_dir, f"{tomo}_ctf.star")
        dose_file = os.path.join(tomo_dir, f"{tomo}_dose.txt")
        wedge_out = os.path.join(tomo_dir, "wedge_list_C13_24.star")

        # --- Detect tomogram dimensions dynamically ---
        if os.path.exists(tomo_file):
            tomo_dim = get_tomo_dimensions(tomo_file)
        else:
            print(f"Tomogram file not found for {tomo}, skipping.")
            continue

        print(f"Generating wedge list for {tomo} (ID {idx})")
        print(f"   File: {os.path.basename(tomo_file)} | Dimensions: {tomo_dim}")

        # --- Create wedge list using CryoCAT ---
        wedge_df = wedgeutils.create_wedge_list_sg(
            tomo_id=idx,
            tomo_dim=tomo_dim,
            pixel_size=pixel_size,
            tlt_file=tlt_file,
            ctf_file=tlt_file,
            ctf_file_type = 'mdoc',
            dose_file=tlt_file,
            voltage=voltage,
            amp_contrast=amp_contrast,
            cs=cs,
            output_file=wedge_out
        )

        print(f"Wedge list saved: {wedge_out}")

        # --- Append to main data_input.star file ---
        f.write(f"{tomo_dir} {output_dir} {vol_ext} {os.path.basename(tomo_file)} {idx} "
                f"wedge_list_C13_24.star {template_name} {mask_name} {symmetry} "
                f"{anglist_order} {angle_list} {smap_name} {omap_name} "
                f"{lp_rad} {hp_rad} {binning} {tiling}\n")

print(f"\nAll wedge lists generated and STAR file created:\n{output_star}")


with mrcfile.open(mrc_path) as m:
    tomo_dim = list(m.data.shape)
    pixel_size = float(m.voxel_size.x)
print(folder_path)

# Creates wedge list for single tomogram from mdoc file and CTF estimation file obtained from GCTF
_ = wedgeutils.create_wedge_list_sg(tomo_id=1,
                                tomo_dim = tomo_dim,  # get from loading the mrc file
                                pixel_size = pixel_size,
                                tlt_file = os.path.join('TS_0002.mrc.mdoc'),
                                output_file= folder_path + "/wedge_list_C13_24.star",
                                drop_nan_columns=True)

