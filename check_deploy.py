from huggingface_hub import hf_hub_download
import json

hf_hub_download(
    repo_id="shiv0805/VIBE",
    repo_type="dataset",
    filename="annotations/captions_val2017.json",
    local_dir="deploy_check"
)

with open("deploy_check/annotations/captions_val2017.json") as f:
    raw = json.load(f)

print(f"Images in deployed annotation: {len(raw['images'])}")
print(f"Captions in deployed annotation: {len(raw['annotations'])}")