from pathlib import Path
from collections import Counter
import yaml


# ============================================================
# Dataset paths
# ============================================================

DATASETS = {
    "road_traffic": Path(
        r"D:\Intelligent Traffic Monitoring & Violation Detection System"
        r"\data\raw\road-traffic"
    ),
    "traffic_vehicle": Path(
        r"D:\Intelligent Traffic Monitoring & Violation Detection System"
        r"\data\processed\traffic-vehicle-detection"
    ),
}


# ============================================================
# Analyze one dataset
# ============================================================

def analyze_dataset(name: str, dataset_path: Path):

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    yaml_path = dataset_path / "data.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]

    print(f"Classes: {len(names)}")

    for i, class_name in enumerate(names):
        print(f"{i:2d}: {class_name}")

    print("\nCLASS DISTRIBUTION")

    total_boxes = 0

    for split in ["train", "valid", "test"]:

        labels_dir = dataset_path / split / "labels"

        if not labels_dir.exists():
            print(f"\n{split}: labels directory not found")
            continue

        counter = Counter()
        box_widths = []
        box_heights = []

        label_files = list(labels_dir.glob("*.txt"))

        for label_file in label_files:

            with open(label_file, "r", encoding="utf-8") as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])

                    counter[class_id] += 1
                    box_widths.append(width)
                    box_heights.append(height)

                    total_boxes += 1

        print(f"\n{split}:")
        print(f"  Label files: {len(label_files)}")

        split_total = sum(counter.values())
        print(f"  Total objects: {split_total}")

        for class_id in range(len(names)):
            print(
                f"    {class_id:2d} | "
                f"{names[class_id]:20s} | "
                f"{counter[class_id]}"
            )

        if box_widths:
            print(
                f"  Avg box width:  {sum(box_widths) / len(box_widths):.4f}"
            )
            print(
                f"  Avg box height: {sum(box_heights) / len(box_heights):.4f}"
            )

    print(f"\nTOTAL OBJECTS: {total_boxes}")


# ============================================================
# Run analysis
# ============================================================

for dataset_name, dataset_path in DATASETS.items():
    analyze_dataset(dataset_name, dataset_path)

print("\n" + "=" * 70)
print("LABEL ANALYSIS COMPLETED")
print("=" * 70)