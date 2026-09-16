import os

BASE_DIR = r"..\dataset_split"

splits = ["train", "val", "test"]

print("=" * 70)
print("DATASET SPLIT VERIFICATION")
print("=" * 70)

total_ir = 0
total_rgb = 0

for split in splits:

    ir_dir = os.path.join(BASE_DIR, split, "infrared")
    rgb_dir = os.path.join(BASE_DIR, split, "rgb")

    ir_files = set(
        f for f in os.listdir(ir_dir)
        if f.lower().endswith(".png")
    )

    rgb_files = set(
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith(".png")
    )

    missing_rgb = ir_files - rgb_files
    missing_ir = rgb_files - ir_files

    print()
    print("-" * 70)
    print(split.upper())
    print("-" * 70)

    print("IR images :", len(ir_files))
    print("RGB images:", len(rgb_files))

    if missing_rgb:
        print("Missing RGB files:", len(missing_rgb))
    else:
        print("Missing RGB files: 0")

    if missing_ir:
        print("Missing IR files:", len(missing_ir))
    else:
        print("Missing IR files: 0")

    if len(ir_files) == len(rgb_files) and not missing_rgb and not missing_ir:
        print("STATUS: PASS")

    else:
        print("STATUS: FAIL")

    total_ir += len(ir_files)
    total_rgb += len(rgb_files)


print()
print("=" * 70)
print("TOTAL")
print("=" * 70)

print("Total IR :", total_ir)
print("Total RGB:", total_rgb)

if total_ir == total_rgb:
    print("OVERALL STATUS: PASS")
else:
    print("OVERALL STATUS: FAIL")

print("=" * 70)