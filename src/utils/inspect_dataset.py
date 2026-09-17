from pathlib import Path
from collections import Counter
import yaml


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "traffic-vehicle-detection"
)
YAML_PATH = DATASET_DIR / "data.yaml"


# =========================
# Load data.yaml
# =========================

with open(YAML_PATH, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Number of classes: {len(class_names)}")

for class_id, class_name in enumerate(class_names):
    print(f"{class_id}: {class_name}")


# =========================
# Count images
# =========================

splits = {
    "train": DATASET_DIR / "train",
    "valid": DATASET_DIR / "valid",
    "test": DATASET_DIR / "test",
}

print("\n" + "=" * 60)
print("IMAGE COUNTS")
print("=" * 60)

for split_name, split_path in splits.items():

    images_dir = split_path / "images"

    if not images_dir.exists():
        print(f"{split_name}: directory not found")
        continue

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    print(f"{split_name}: {len(images)} images")


# =========================
# Count labels
# =========================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

total_objects = 0

for split_name, split_path in splits.items():

    labels_dir = split_path / "labels"

    if not labels_dir.exists():
        print(f"\n{split_name}: labels directory not found")
        continue

    counter = Counter()

    label_files = list(labels_dir.glob("*.txt"))

    for label_file in label_files:

        with open(label_file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                class_id = int(parts[0])

                counter[class_id] += 1
                total_objects += 1

    print(f"\n{split_name}")

    for class_id in range(len(class_names)):

        count = counter[class_id]

        print(
            f"{class_id:2d} | "
            f"{class_names[class_id]:20s} | "
            f"{count}"
        )


# =========================
# Total objects
# =========================

print("\n" + "=" * 60)
print(f"TOTAL ANNOTATED OBJECTS: {total_objects}")
print("=" * 60)