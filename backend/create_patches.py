import os
import glob
import shutil

import numpy as np
import rasterio
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = r"..\dataset_raw\LANDSAT_FINAL_24\LANDSAT_IR_RGB_FINAL-20260904T150856Z-1-001\LANDSAT_IR_RGB_FINAL"

OUTPUT_DIR = r"..\dataset"

IR_DIR = os.path.join(OUTPUT_DIR, "infrared")
RGB_DIR = os.path.join(OUTPUT_DIR, "rgb")

PATCH_SIZE = 256

# A patch must have at least this percentage of valid pixels
MIN_VALID_PERCENT = 90.0


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(IR_DIR, exist_ok=True)
os.makedirs(RGB_DIR, exist_ok=True)


# ============================================================
# REMOVE OLD PATCHES
# ============================================================

print("=" * 70)
print("REMOVING OLD PATCHES")
print("=" * 70)

for folder in [IR_DIR, RGB_DIR]:

    for file in glob.glob(
        os.path.join(folder, "*.png")
    ):
        os.remove(file)


# ============================================================
# FIND TIFF FILES
# ============================================================

tiff_files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.tif")
    )
)

print()
print("TIFF files found:", len(tiff_files))
print()


# ============================================================
# COUNTERS
# ============================================================

total_valid_patches = 0
total_discarded_patches = 0


# ============================================================
# PROCESS EACH TIFF
# ============================================================

for scene_number, tif_path in enumerate(
    tiff_files,
    start=1
):

    filename = os.path.basename(tif_path)

    print("-" * 70)
    print(
        f"PROCESSING {scene_number}/{len(tiff_files)}: "
        f"{filename}"
    )
    print("-" * 70)


    # --------------------------------------------------------
    # OPEN TIFF
    # --------------------------------------------------------

    with rasterio.open(tif_path) as src:

        if src.count < 4:

            print(
                "ERROR: Less than 4 bands. Skipping."
            )

            continue


        # ----------------------------------------------------
        # READ FOUR BANDS
        # ----------------------------------------------------

        red = src.read(1).astype(np.float32)

        green = src.read(2).astype(np.float32)

        blue = src.read(3).astype(np.float32)

        thermal = src.read(4).astype(np.float32)


    height, width = red.shape


    # --------------------------------------------------------
    # COMMON VALID MASK
    # --------------------------------------------------------
    #
    # A pixel is valid only when ALL FOUR bands are finite.
    # --------------------------------------------------------

    valid_mask = (

        np.isfinite(red)

        & np.isfinite(green)

        & np.isfinite(blue)

        & np.isfinite(thermal)

    )


    scene_valid_percent = (
        np.mean(valid_mask) * 100
    )


    print(
        f"Scene common valid pixels: "
        f"{scene_valid_percent:.2f}%"
    )


    # --------------------------------------------------------
    # PATCH COUNTERS
    # --------------------------------------------------------

    valid_count = 0
    discarded_count = 0


    # --------------------------------------------------------
    # PATCH LOOP
    # --------------------------------------------------------

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        PATCH_SIZE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            PATCH_SIZE
        ):


            # -----------------------------------------------
            # EXTRACT PATCHES
            # -----------------------------------------------

            red_patch = red[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            green_patch = green[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            blue_patch = blue[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            thermal_patch = thermal[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]


            mask_patch = valid_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]


            # -----------------------------------------------
            # VALID PERCENTAGE
            # -----------------------------------------------

            valid_percent = (
                np.mean(mask_patch) * 100
            )


            # -----------------------------------------------
            # REJECT BAD PATCH
            # -----------------------------------------------

            if valid_percent < MIN_VALID_PERCENT:

                discarded_count += 1

                continue


            # =================================================
            # THERMAL NORMALIZATION
            # =================================================

            valid_thermal = thermal_patch[
                mask_patch
            ]


            if len(valid_thermal) == 0:

                discarded_count += 1

                continue


            t_low, t_high = np.percentile(
                valid_thermal,
                [2, 98]
            )


            if t_high <= t_low:

                discarded_count += 1

                continue


            thermal_norm = (
                thermal_patch - t_low
            ) / (
                t_high - t_low
            )


            thermal_norm = np.clip(
                thermal_norm,
                0,
                1
            )


            # =================================================
            # RGB NORMALIZATION
            # =================================================

            rgb_stack = np.stack(
                [
                    red_patch,
                    green_patch,
                    blue_patch
                ],
                axis=-1
            )


            rgb_norm = np.zeros_like(
                rgb_stack,
                dtype=np.float32
            )


            for channel in range(3):

                channel_data = rgb_stack[
                    :, :, channel
                ]

                valid_values = channel_data[
                    mask_patch
                ]


                if len(valid_values) == 0:

                    discarded_count += 1

                    break


                low, high = np.percentile(
                    valid_values,
                    [2, 98]
                )


                if high <= low:

                    discarded_count += 1

                    break


                normalized = (
                    channel_data - low
                ) / (
                    high - low
                )


                rgb_norm[
                    :, :, channel
                ] = np.clip(
                    normalized,
                    0,
                    1
                )


            else:

                # -------------------------------------------
                # REPLACE INVALID PIXELS WITH ZERO
                # -------------------------------------------

                thermal_norm[
                    ~mask_patch
                ] = 0


                rgb_norm[
                    ~mask_patch
                ] = 0


                # -------------------------------------------
                # CONVERT TO UINT8
                # -------------------------------------------

                ir_uint8 = (
                    thermal_norm * 255
                ).astype(
                    np.uint8
                )


                rgb_uint8 = (
                    rgb_norm * 255
                ).astype(
                    np.uint8
                )


                # -------------------------------------------
                # FILE NAME
                # -------------------------------------------

                base_name = os.path.splitext(
                    filename
                )[0]


                patch_name = (
                    f"{base_name}"
                    f"_patch_{valid_count + 1:04d}.png"
                )


                ir_path = os.path.join(
                    IR_DIR,
                    patch_name
                )


                rgb_path = os.path.join(
                    RGB_DIR,
                    patch_name
                )


                # -------------------------------------------
                # SAVE IR
                # -------------------------------------------

                Image.fromarray(
                    ir_uint8,
                    mode="L"
                ).save(
                    ir_path
                )


                # -------------------------------------------
                # SAVE RGB
                # -------------------------------------------

                Image.fromarray(
                    rgb_uint8,
                    mode="RGB"
                ).save(
                    rgb_path
                )


                valid_count += 1


    # ========================================================
    # SCENE SUMMARY
    # ========================================================

    total_valid_patches += valid_count

    total_discarded_patches += discarded_count


    print(
        f"Valid patches     : {valid_count}"
    )

    print(
        f"Discarded patches : {discarded_count}"
    )

    print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("PATCH GENERATION COMPLETED")
print("=" * 70)

print(
    f"Total valid patches     : "
    f"{total_valid_patches}"
)

print(
    f"Total discarded patches : "
    f"{total_discarded_patches}"
)

print()
print("IR patches:")
print(
    os.path.abspath(IR_DIR)
)

print()
print("RGB patches:")
print(
    os.path.abspath(RGB_DIR)
)

print("=" * 70)