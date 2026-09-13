from huggingface_hub import snapshot_download

# Download into local dataset directory
local_path = snapshot_download(
    repo_id="thundarstrom/traffic-vehicle-detection",
    repo_type="dataset",
    local_dir=r"D:\Intelligent Traffic Monitoring & Violation Detection System\data\raw\traffic-vehicle-detection"
)
print(f"Dataset downloaded to: {local_path}")
