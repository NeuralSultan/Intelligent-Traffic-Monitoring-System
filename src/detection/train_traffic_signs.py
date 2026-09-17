from ultralytics import YOLO


DATASET_YAML = (
    r"D:\Intelligent Traffic Monitoring & Violation Detection System"
    r"\data\raw\traffic-signs-detection\data.yaml"
)

MODEL_NAME = "yolo11n.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16

OUTPUT_DIR = (
    r"D:\Intelligent Traffic Monitoring & Violation Detection System"
    r"\outputs\training"
)


def main():

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=0,
        project=OUTPUT_DIR,
        name="traffic_signs_yolo11n",
        pretrained=True,
        patience=15,
        workers=2,
        verbose=True,
    )


if __name__ == "__main__":
    main()