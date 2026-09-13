from pathlib import Path
import shutil
import yaml


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Intelligent Traffic Monitoring & Violation Detection System"
)

ROAD_TRAFFIC = PROJECT_ROOT / "data" / "raw" / "road-traffic"

TRAFFIC_VEHICLE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "traffic-vehicle-6classes"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified-traffic"
)


# ============================================================
# Final unified classes
# ============================================================

FINAL_CLASSES = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "traffic_light",
    "crosswalk",
]

FINAL_CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(FINAL_CLASSES)
}


# ============================================================
# Class mappings
# ============================================================

ROAD_TRAFFIC_MAPPING = {
    "vehicles": "car",
    "buses": "bus",
    "motorcycles": "motorcycle",
    "bicycles": "bicycle",
    "traffic lights": "traffic_light",
    "crosswalks": "crosswalk",
}

TRAFFIC_VEHICLE_MAPPING = {
    "Bicycle": "bicycle",
    "Bus": "bus",
    "Car": "car",
    "Motorcycle": "motorcycle",
    "Traffic-light": "traffic_light",
    "Truck": "truck",
}


# ============================================================
# Utility functions
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def load_class_names(dataset_path: Path):
    """Load class names from data.yaml."""

    yaml_path = dataset_path / "data.yaml"

    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data["names"]

    if isinstance(names, dict):
        names = [
            names[key]
            for key in sorted(names, key=lambda x: int(x))
        ]

    return names


def collect_images(images_dir: Path):
    """Return all supported images."""

    if not images_dir.exists():
        return []

    return [
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def remap_label(
    source_label_path: Path,
    destination_label_path: Path,
    source_names,
    class_mapping,
):
    """
    Convert YOLO detection OR segmentation labels
    into unified YOLO detection bounding boxes.
    """

    output_lines = []

    with open(source_label_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            parts = line.strip().split()

            if not parts:
                continue

            try:
                source_class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]

            except ValueError:
                print(
                    f"Warning: invalid numeric label: "
                    f"{source_label_path} line {line_number}"
                )
                continue

            if source_class_id < 0 or source_class_id >= len(source_names):
                print(
                    f"Warning: invalid class ID "
                    f"{source_class_id} in {source_label_path}"
                )
                continue

            source_class_name = source_names[source_class_id]

            # Ignore classes not needed by the project.
            if source_class_name not in class_mapping:
                continue

            unified_class_name = class_mapping[source_class_name]

            unified_class_id = FINAL_CLASS_TO_ID[
                unified_class_name
            ]

            # ----------------------------------------------------
            # YOLO Detection
            # class x_center y_center width height
            # ----------------------------------------------------

            if len(values) == 4:

                x_center, y_center, width, height = values

            # ----------------------------------------------------
            # YOLO Segmentation
            # class x1 y1 x2 y2 x3 y3 ...
            # ----------------------------------------------------

            elif len(values) >= 6 and len(values) % 2 == 0:

                x_coordinates = values[0::2]
                y_coordinates = values[1::2]

                x_min = min(x_coordinates)
                x_max = max(x_coordinates)

                y_min = min(y_coordinates)
                y_max = max(y_coordinates)

                width = x_max - x_min
                height = y_max - y_min

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # Skip degenerate polygons
                if width <= 0 or height <= 0:
                    continue

            else:

                print(
                    f"Warning: unsupported label format: "
                    f"{source_label_path} line {line_number}"
                )
                continue

            # ----------------------------------------------------
            # Validate normalized YOLO values
            # ----------------------------------------------------

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)

            if width <= 0 or height <= 0:
                continue

            output_lines.append(
                f"{unified_class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

    if output_lines:

        with open(
            destination_label_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "\n".join(output_lines) + "\n"
            )

        return True

    return False


def process_dataset(
    dataset_name: str,
    dataset_path: Path,
    class_mapping,
):
    """Merge one dataset into the unified dataset."""

    print("\n" + "=" * 70)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 70)

    source_names = load_class_names(dataset_path)

    print("Source classes:")
    for class_id, name in enumerate(source_names):
        print(f"  {class_id}: {name}")

    print("\nMapping:")
    for source_name, unified_name in class_mapping.items():
        print(f"  {source_name} -> {unified_name}")

    for split in ["train", "valid", "test"]:

        source_images_dir = dataset_path / split / "images"
        source_labels_dir = dataset_path / split / "labels"

        destination_images_dir = (
            OUTPUT_DIR / split / "images"
        )

        destination_labels_dir = (
            OUTPUT_DIR / split / "labels"
        )

        destination_images_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_labels_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not source_images_dir.exists():
            print(f"\n{split}: images directory not found")
            continue

        if not source_labels_dir.exists():
            print(f"\n{split}: labels directory not found")
            continue

        images = collect_images(source_images_dir)

        copied_images = 0
        copied_labels = 0
        ignored_images = 0

        for image_path in images:

            source_label_path = (
                source_labels_dir
                / f"{image_path.stem}.txt"
            )

            if not source_label_path.exists():
                print(
                    f"Warning: missing label: "
                    f"{image_path.name}"
                )
                continue

            # Prefix filename so datasets can never overwrite each other.
            new_stem = (
                f"{dataset_name}_{image_path.stem}"
            )

            destination_image_path = (
                destination_images_dir
                / f"{new_stem}{image_path.suffix.lower()}"
            )

            destination_label_path = (
                destination_labels_dir
                / f"{new_stem}.txt"
            )

            has_useful_objects = remap_label(
                source_label_path,
                destination_label_path,
                source_names,
                class_mapping,
            )

            # Ignore images that contain none of our final classes.
            if not has_useful_objects:

                if destination_label_path.exists():
                    destination_label_path.unlink()

                ignored_images += 1
                continue

            shutil.copy2(
                image_path,
                destination_image_path,
            )

            copied_images += 1
            copied_labels += 1

        print(f"\n{split}:")
        print(f"  Source images:   {len(images)}")
        print(f"  Copied images:   {copied_images}")
        print(f"  Copied labels:   {copied_labels}")
        print(f"  Ignored images:  {ignored_images}")


# ============================================================
# Prepare output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Process datasets
# ============================================================

process_dataset(
    dataset_name="road",
    dataset_path=ROAD_TRAFFIC,
    class_mapping=ROAD_TRAFFIC_MAPPING,
)

process_dataset(
    dataset_name="traffic",
    dataset_path=TRAFFIC_VEHICLE,
    class_mapping=TRAFFIC_VEHICLE_MAPPING,
)


# ============================================================
# Create unified data.yaml
# ============================================================

unified_yaml = {
    "path": str(OUTPUT_DIR).replace("\\", "/"),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(FINAL_CLASSES),
    "names": FINAL_CLASSES,
}

yaml_path = OUTPUT_DIR / "data.yaml"

with open(
    yaml_path,
    "w",
    encoding="utf-8",
) as file:

    yaml.safe_dump(
        unified_yaml,
        file,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("UNIFIED DATASET CREATED")
print("=" * 70)

print(f"Output:")
print(OUTPUT_DIR)

print("\nFinal classes:")

for class_id, class_name in enumerate(FINAL_CLASSES):
    print(f"  {class_id}: {class_name}")

print("\nDone.")