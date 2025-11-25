"""
Utility script to prepare a microtubule template and mask for GAPStop/cryocat TM.

This mirrors the workflow in `temp.py`, but uses a microtubule template
(`./emd_6351.mrc`) as described in the high-confidence TM pipeline
of Cruz-León et al., Nat. Commun. 15, 47839 (2024)
([doi:10.1038/s41467-024-47839-8](https://doi.org/10.1038/s41467-024-47839-8)).

Steps:
1. Resample the microtubule template to TARGET_PIXEL_SIZE (default 6 Å/px).
2. Crop a cubic box around the center (default 192³ voxels).
3. Save the cropped volume as MRC and EM.
4. Create a spherical mask of the same box size for TM.
"""

import os

import mrcfile
import numpy as np
from scipy.ndimage import zoom

import cryocat
from cryocat import cryomask


# --------------------------------------------------
# User settings (edit as needed)
# --------------------------------------------------
INPUT_TEMPLATE = "./emd_6351.mrc"               # input microtubule template
RESAMPLED_TEMPLATE = "./emd_6351_6A.mrc"        # resampled template
CROPPED_TEMPLATE = "./emd_6351_6A_crop96.mrc"  # cropped, resampled template

TARGET_PIXEL_SIZE = 6.0  # Å/voxel, as recommended in the GAPStop TM paper
CROP_BOX = 96           # cubic crop size in voxels (CROP_BOX³)

MASK_OUTPUT = "./mt_mask_96.em"  # spherical mask for TM


def resample_volume(in_path: str, out_path: str, target_px: float) -> None:
    """Resample a 3D MRC volume to a target pixel size (isotropic)."""
    with mrcfile.open(in_path, permissive=True) as m:
        vol = np.copy(m.data)
        orig_px = float(m.voxel_size.x)

    scale = orig_px / target_px
    print(f"{in_path}: original px = {orig_px:.3f} Å → scale factor = {scale:.3f}")

    # isotropic resampling
    vol_resampled = zoom(vol, scale, order=1)

    with mrcfile.new(out_path, overwrite=True) as out:
        out.set_data(vol_resampled.astype(np.float32))
        out.voxel_size = target_px  # update header voxel size

    print(f"Saved resampled volume to {out_path}, new shape = {vol_resampled.shape}")


def crop_center_cube(
    in_path: str,
    out_path: str,
    box: int,
) -> None:
    """Crop a centered cubic box of size `box`³ voxels from an MRC volume."""
    with mrcfile.open(in_path, permissive=True) as m:
        data = m.data.copy()
        voxel_size = float(m.voxel_size.x)
        print(f"{in_path}: shape = {data.shape}, voxel size = {voxel_size:.3f} Å")

    # Assume (Z, Y, X) ordering
    nz, ny, nx = data.shape
    cz, cy, cx = np.array(data.shape) // 2

    half = box // 2
    z0, z1 = cz - half, cz + half
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # Sanity check: bounds
    for a0, a1, n, axis in [(z0, z1, nz, "Z"), (y0, y1, ny, "Y"), (x0, x1, nx, "X")]:
        if a0 < 0 or a1 > n:
            raise ValueError(
                f"Requested crop box is out of bounds along {axis}: "
                f"[{a0}, {a1}) vs size {n}. "
                "Choose a smaller box or recenter the template."
            )

    data_crop = data[z0:z1, y0:y1, x0:x1]
    print(f"Cropped shape: {data_crop.shape}")

    with mrcfile.new(out_path, overwrite=True) as out:
        out.set_data(data_crop.astype(np.float32))
        out.voxel_size = voxel_size

    print(f"Saved cropped volume to {out_path}")


def create_spherical_mask(mask_size: int, output_path: str) -> None:
    """Create a simple spherical mask using cryocat.cryomask."""
    print(f"Creating spherical mask of size {mask_size}³ voxels → {output_path}")
    _ = cryomask.spherical_mask(
        mask_size=mask_size,
        center=None,
        gaussian=2,
        gaussian_outwards=False,
        output_name=output_path,
    )
    print(f"Saved mask to {output_path}")


def main() -> None:
    if not os.path.exists(INPUT_TEMPLATE):
        raise FileNotFoundError(f"Template not found: {INPUT_TEMPLATE}")

    print("=== Microtubule template preparation (tubule.py) ===")

    # 1) Resample to target pixel size
    resample_volume(INPUT_TEMPLATE, RESAMPLED_TEMPLATE, TARGET_PIXEL_SIZE)

    # 2) Crop a cubic box around the center
    crop_center_cube(RESAMPLED_TEMPLATE, CROPPED_TEMPLATE, CROP_BOX)

    # 3) Convert cropped template to EM format for GAPStop/cryocat
    print("Converting cropped template to EM (.em) format...")
    cryocat.cryomap.mrc2em(
        CROPPED_TEMPLATE,
        invert=False,
        overwrite=True,
        output_name=None,  # will auto-derive name from MRC
    )
    print("✓ MRC → EM conversion complete.")

    # 4) Create a spherical mask matching the cropped box size
    create_spherical_mask(CROP_BOX, MASK_OUTPUT)

    print("\nAll done!")
    print(f"  Resampled template: {RESAMPLED_TEMPLATE}")
    print(f"  Cropped template:   {CROPPED_TEMPLATE}")
    print(f"  Mask:               {MASK_OUTPUT}")
    print(
        "You can now reference the EM template and mask in your GAPStop "
        "data_input.star for microtubule template matching."
    )


if __name__ == "__main__":
    main()


