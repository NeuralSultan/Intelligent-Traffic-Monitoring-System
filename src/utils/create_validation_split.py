from pathlib import Path
import random
import shutil
import yaml


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "traffic-vehicle-detection"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "traffic-vehicle-detection"
)

VALIDATION_RATIO = 0.20
SEED = 42

random.seed(SEED)


# ============================================================
# Paths
# ============================================================

source_train_images = SOURCE_DIR / "train" / "images"
source_train_labels = SOURCE_DIR / "train" / "labels"

source_test_images = SOURCE_DIR / "test" / "images"
source_test_labels = SOURCE_DIR / "test" / "labels"


output_train_images = OUTPUT_DIR / "train" / "images"
output_train_labels = OUTPUT_DIR / "train" / "labels"

output_valid_images = OUTPUT_DIR / "valid" / "images"
output_valid_labels = OUTPUT_DIR / "valid" / "labels"

output_test_images = OUTPUT_DIR / "test" / "images"
output_test_labels = OUTPUT_DIR / "test" / "labels"


# ============================================================
# Create directories
# ============================================================

directories = [
    output_train_images,
    output_train_labels,
    output_valid_images,
    output_valid_labels,
    output_test_images,
    output_test_labels,
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Find training images
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

images = [
    image
    for image in source_train_images.iterdir()
    if image.is_file() and image.suffix.lower() in image_extensions
]

if not images:
    raise RuntimeError(
        f"No training images found in: {source_train_images}"
    )

print(f"Total source training images: {len(images)}")


# ============================================================
# Shuffle images
# ============================================================

random.shuffle(images)

validation_count = int(len(images) * VALIDATION_RATIO)

valid_images = images[:validation_count]
train_images = images[validation_count:]

print(f"Training images:   {len(train_images)}")
print(f"Validation images: {len(valid_images)}")


# ============================================================
# Copy train / validation
# ============================================================

def copy_image_and_label(
    image_path: Path,
    destination_images: Path,
    destination_labels: Path,
):
    label_path = source_train_labels / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"Warning: missing label for {image_path.name}")
        return

    shutil.copy2(
        image_path,
        destination_images / image_path.name
    )

    shutil.copy2(
        label_path,
        destination_labels / label_path.name
    )


for image in train_images:
    copy_image_and_label(
        image,
        output_train_images,
        output_train_labels,
    )


for image in valid_images:
    copy_image_and_label(
        image,
        output_valid_images,
        output_valid_labels,
    )


# ============================================================
# Copy test set
# ============================================================

test_images = [
    image
    for image in source_test_images.iterdir()
    if image.is_file() and image.suffix.lower() in image_extensions
]

print(f"Test images:        {len(test_images)}")

for image in test_images:

    label_path = source_test_labels / f"{image.stem}.txt"

    if not label_path.exists():
        print(f"Warning: missing test label for {image.name}")
        continue

    shutil.copy2(
        image,
        output_test_images / image.name
    )

    shutil.copy2(
        label_path,
        output_test_labels / label_path.name
    )


# ============================================================
# Create data.yaml
# ============================================================

data_yaml = {
    "path": str(OUTPUT_DIR).replace("\\", "/"),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 4,
    "names": [
        "car",
        "bus",
        "truck",
        "van",
    ],
}

yaml_path = OUTPUT_DIR / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        data_yaml,
        file,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Output directory:")
print(OUTPUT_DIR)

print("\nCreated:")
print("  train/")
print("  valid/")
print("  test/")
print("  data.yaml")