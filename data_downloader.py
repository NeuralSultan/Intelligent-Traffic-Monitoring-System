from huggingface_hub import snapshot_download

# Download the dataset
snapshot_download(
    repo_id="Libre-YOLO/road-traffic",
    repo_type="dataset",
    local_dir="./road-traffic"
)
