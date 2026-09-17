from pathlib import Path
from collections import Counter
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified-traffic"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_images(images_dir):
    return {
        p.stem
        for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def validate_split(split):
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    print("\n" + "=" * 60)
    print(f"{split.upper()} SPLIT")
    print("=" * 60)

    if not images_dir.exists():
        print("Images directory not found.")
        return

    if not labels_dir.exists():
        print("Labels directory not found.")
        return

    images = get_images(images_dir)
    labels = {
        p.stem
        for p in labels_dir.glob("*.txt")
    }

    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    missing_labels = images - labels
    orphan_labels = labels - images

    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(orphan_labels)}")

    class_counter = Counter()

    invalid_lines = 0
    invalid_values = 0
    empty_labels = 0

    for label_file in labels_dir.glob("*.txt"):

        with open(label_file, "r", encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if not lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                values = [float(v) for v in parts[1:]]

            except ValueError:
                invalid_lines += 1
                continue

            if class_id < 0 or class_id > 6:
                invalid_values += 1
                continue

            if not all(0.0 <= value <= 1.0 for value in values):
                invalid_values += 1
                continue

            x, y, width, height = values

            if width <= 0 or height <= 0:
                invalid_values += 1
                continue

            class_counter[class_id] += 1

    print("\nClass distribution:")

    for class_id in range(7):
        print(
            f"{class_id}: "
            f"{class_counter[class_id]}"
        )

    print("\nValidation:")

    print(f"Invalid label lines: {invalid_lines}")
    print(f"Invalid box values: {invalid_values}")
    print(f"Empty label files: {empty_labels}")


# ============================================================
# Dataset information
# ============================================================

yaml_path = DATASET_DIR / "data.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print("=" * 60)
print("UNIFIED DATASET VALIDATION")
print("=" * 60)

print("\nClasses:")

for class_id, class_name in enumerate(data["names"]):
    print(f"{class_id}: {class_name}")


for split in ["train", "valid", "test"]:
    validate_split(split)


print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)