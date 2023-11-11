conda env create -f environment.yml
conda activate transformer
set HF_ENDPOINT=https://hf-mirror.com
python download_model.py