import os
import rasterio
import numpy as np


# ============================================================
# DATASET LOCATION
# ============================================================
DATASET_RAW = r"..\dataset_raw\LANDSAT_FINAL_24\LANDSAT_IR_RGB_FINAL-20260904T150856Z-1-001\LANDSAT_IR_RGB_FINAL"
# ============================================================
# START
# ============================================================

print("=" * 70)
print("INFRARED + RGB TIFF DATASET CHECK")
print("=" * 70)


# ============================================================
# CHECK WHETHER DATASET FOLDER EXISTS
# ============================================================

if not os.path.exists(DATASET_RAW):

    print("\nERROR: Dataset folder not found.")
    print("Expected location:")
    print(os.path.abspath(DATASET_RAW))

    exit()


# ============================================================
# FIND ALL TIFF FILES
# ============================================================

tif_files = []

for root, dirs, files in os.walk(DATASET_RAW):

    for file in files:

        if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):

            full_path = os.path.join(root, file)

            tif_files.append(full_path)


tif_files.sort()


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset folder:")
print(os.path.abspath(DATASET_RAW))

print(f"\nTotal TIFF files found: {len(tif_files)}")


# ============================================================
# IF NO TIFF FILES ARE FOUND
# ============================================================

if len(tif_files) == 0:

    print("\nERROR: No TIFF files found.")
    print("\nPlease check that your TIFF files are inside:")

    print(os.path.abspath(DATASET_RAW))

    exit()


# ============================================================
# CHECK EVERY TIFF FILE
# ============================================================

for i, file_path in enumerate(tif_files, start=1):

    print("\n")
    print("-" * 70)

    print(f"FILE {i}: {os.path.basename(file_path)}")

    print("-" * 70)


    try:

        # ----------------------------------------------------
        # OPEN TIFF
        # ----------------------------------------------------

        with rasterio.open(file_path) as src:

            # ------------------------------------------------
            # BASIC INFORMATION
            # ------------------------------------------------

            print(f"Width       : {src.width}")

            print(f"Height      : {src.height}")

            print(f"Band count  : {src.count}")

            print(f"CRS         : {src.crs}")

            print(f"Data types  : {src.dtypes}")

            print(f"Descriptions: {src.descriptions}")


            # ------------------------------------------------
            # CHECK BAND COUNT
            # ------------------------------------------------

            if src.count != 4:

                print(
                    "\nWARNING: "
                    "This file does NOT contain exactly 4 bands."
                )


            # ------------------------------------------------
            # CHECK EACH BAND
            # ------------------------------------------------

            for band_number in range(1, src.count + 1):

                band = src.read(band_number).astype(np.float32)


                # --------------------------------------------
                # FIND VALID PIXELS
                # --------------------------------------------

                valid = np.isfinite(band)

                valid_pixels = band[valid]


                # --------------------------------------------
                # BAND STATISTICS
                # --------------------------------------------

                if len(valid_pixels) > 0:

                    valid_count = np.sum(valid)

                    total_count = band.size

                    valid_percentage = (
                        valid_count / total_count
                    ) * 100


                    print(f"\nBand {band_number}:")

                    print(
                        f"  Min          : "
                        f"{np.min(valid_pixels):.6f}"
                    )

                    print(
                        f"  Max          : "
                        f"{np.max(valid_pixels):.6f}"
                    )

                    print(
                        f"  Mean         : "
                        f"{np.mean(valid_pixels):.6f}"
                    )

                    print(
                        f"  Valid pixels : "
                        f"{valid_count}"
                    )

                    print(
                        f"  Valid %      : "
                        f"{valid_percentage:.2f}%"
                    )


                else:

                    print(
                        f"\nBand {band_number}: "
                        "NO VALID PIXELS"
                    )


    except Exception as e:

        print(f"\nERROR reading file:")

        print(e)


# ============================================================
# FINISHED
# ============================================================

print("\n")

print("=" * 70)

print("DATASET CHECK COMPLETED")

print("=" * 70)