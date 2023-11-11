import os

import torch
from transformers import AutoModel, AutoProcessor

models_root = 'model_params'
models_path = {
               'CLIP': 'clip-vit-large-patch14-336', 
               #'InstructBlip': 'instructblip-vicuna-13b'
               }

class Functions():
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.processors = {}
        for model_name, model_path in models_path.items():
            self.models[model_name] = AutoModel.from_pretrained(
                os.path.join(models_root, model_path), 
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.processors[model_name] = AutoProcessor.from_pretrained(
                os.path.join(models_root, model_path)
            )
    
    def __del__(self):
        for model in self.models.values():
            del model
        for processor in self.processors.values():
            del processor

    def to(self, device):
        self.device = device
        for model_name, _ in self.models.items():
            self.models[model_name] = self.models[model_name].to(device)
    
    def retrieve(self, text:str, images:list):
        model = self.models['CLIP']
        processor = self.processors['CLIP']
    
        inputs = processor(text, images, return_tensors='pt', padding=True)
        with torch.inference_mode():
            outpurs = model(**inputs.to(self.device))
        logits_per_image = outpurs.logits_per_image
        return logits_per_image.cpu()