import os
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT']="https://hf-mirror.com"

token = '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'

path = [
        #('Salesforce/instructblip-vicuna-13b', './model_params/instructblip-flan-t5-xl', True), 
        ('openai/clip-vit-large-patch14-336', './model_params/clip-vit-large-patch14-336', True)
        ]

for (repo_id, local_dir, redownload) in path:
    if not os.path.exists(local_dir) or redownload:
        snapshot_download(repo_id, local_dir=local_dir, token=token, resume_download=True)
