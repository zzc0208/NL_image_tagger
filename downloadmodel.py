from huggingface_hub import snapshot_download

snapshot_download(repo_id="openbmb/MiniCPM-V-2_6-int4", local_dir="models/model")
