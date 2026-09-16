import os
import shutil
import random
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

IR_DIR = r"..\dataset\infrared"
RGB_DIR = r"..\dataset\rgb"
OUTPUT_DIR = r"..\dataset_split"

# Reproducible split
random.seed(42)

# ============================================================
# CREATE FOLDERS
# ============================================================

splits = ["train", "val", "test"]

for split in splits:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "infrared"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "rgb"), exist_ok=True)

# ============================================================
# GET IR FILES
# ============================================================

ir_files = [
    f for f in os.listdir(IR_DIR)
    if f.lower().endswith(".png")
]

print("=" * 70)
print("DATASET SPLITTING")
print("=" * 70)

print("Total IR patches:", len(ir_files))

# ============================================================
# GROUP PATCHES BY SCENE
# ============================================================

scenes = defaultdict(list)

for filename in ir_files:

    # Example:
    # Landsat_IR_RGB_1_patch_0001.png

    if "_patch_" not in filename:
        continue

    scene_name = filename.split("_patch_")[0]

    scenes[scene_name].append(filename)

print("Total scenes:", len(scenes))

# ============================================================
# SHUFFLE SCENES
# ============================================================

scene_names = list(scenes.keys())
random.shuffle(scene_names)

total_scenes = len(scene_names)

train_end = int(total_scenes * 0.70)
val_end = train_end + int(total_scenes * 0.15)

train_scenes = scene_names[:train_end]
val_scenes = scene_names[train_end:val_end]
test_scenes = scene_names[val_end:]

print()
print("Train scenes:", len(train_scenes))
print("Validation scenes:", len(val_scenes))
print("Test scenes:", len(test_scenes))

# ============================================================
# COPY FILES
# ============================================================

def copy_scene_files(scene_list, split_name):

    count = 0

    for scene in scene_list:

        for filename in scenes[scene]:

            ir_source = os.path.join(IR_DIR, filename)
            rgb_source = os.path.join(RGB_DIR, filename)

            ir_destination = os.path.join(
                OUTPUT_DIR,
                split_name,
                "infrared",
                filename
            )

            rgb_destination = os.path.join(
                OUTPUT_DIR,
                split_name,
                "rgb",
                filename
            )

            # Check matching RGB image
            if not os.path.exists(rgb_source):
                print("WARNING: Missing RGB:", filename)
                continue

            shutil.copy2(ir_source, ir_destination)
            shutil.copy2(rgb_source, rgb_destination)

            count += 1

    return count


# ============================================================
# SPLIT
# ============================================================

train_count = copy_scene_files(train_scenes, "train")
val_count = copy_scene_files(val_scenes, "val")
test_count = copy_scene_files(test_scenes, "test")

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("SPLIT COMPLETED")
print("=" * 70)

print("Training patches   :", train_count)
print("Validation patches :", val_count)
print("Testing patches    :", test_count)
print("Total              :", train_count + val_count + test_count)

print()
print("Dataset location:")
print(os.path.abspath(OUTPUT_DIR))

print("=" * 70)