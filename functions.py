import os
from PIL import Image

import torch
import transformers

class Functions():
    def __init__(self, 
                 device, 
                 model_params_root:str='./model_params', 
                 models_path:dict={
                     'CLIP': ('clip-vit-large-patch14-336', transformers.CLIPModel, transformers.CLIPProcessor), 
                     'InstructBlip': ('instructblip-flan-t5-xl', transformers.InstructBlipForConditionalGeneration, transformers.InstructBlipProcessor), 
                 }):
        self.device = device
        self.models = {}
        self.processors = {}
        for model_name, (model_path, model_class, processor_class) in models_path.items():
            self.models[model_name] = model_class.from_pretrained(
                os.path.join(model_params_root, model_path), 
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ).eval()
            self.processors[model_name] = processor_class.from_pretrained(
                os.path.join(model_params_root, model_path)
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
    
        inputs = processor(text=text, images=images, return_tensors='pt', padding=True)
        with torch.inference_mode():
            outputs = model(**inputs.to(self.device))
        logits_per_image = outputs.logits_per_image
        return logits_per_image.cpu()
    
    def vqa(self, texts:list[str], images:list[Image.Image]):
        model = self.models['InstructBlip']
        processor = self.processors['InstructBlip']
        
        inputs = processor(text=texts, images=images, return_tensors='pt', padding=True)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs.to(self.device), 
                do_sample=False, 
                max_length=256, 
                min_length=1, 
                length_penalty=1.0, 
                temperature=1, 
            )
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        return generated_text

if __name__ == '__main__':
    import requests
    device = 'cuda'
    url = "https://img1.baidu.com/it/u=3539595421,754041626&fm=253&fmt=auto&app=138&f=JPEG?w=889&h=500"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "Please describe this image"
    
    batch_size = 64
    func = Functions(device=device)
    output = func.vqa(texts=[prompt] * batch_size, images=[image] * batch_size)
    print(output)