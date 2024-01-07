import os
from PIL import Image

import torch
import transformers

class Functions():
    def __init__(self, 
                 device, 
                 model_params_root:str='./model_params', 
                 models_path:dict={
                     'retrieve': ('clip-vit-large-patch14-336', transformers.CLIPModel, transformers.CLIPProcessor), 
                    #'InstructBlip': ('instructblip-flan-t5-xl', transformers.InstructBlipForConditionalGeneration, transformers.InstructBlipProcessor), 
                     'vqa': ('llava-1.5-13b-hf', transformers.LlavaForConditionalGeneration, transformers.LlavaProcessor), 
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
        model = self.models['retrieve']
        processor = self.processors['retrieve']
    
        inputs = processor(text=text, images=images, return_tensors='pt', padding='longest', truncation=True).to(self.device)
        #image_inputs = processor(images=images, return_tensors='pt').to(self.device)
        #text_inputs = processor.tokenizer(text=text, return_tensors='pt', padding='longest', truncation=True).to(self.device)
        with torch.inference_mode():
            #text_features = model.get_text_features(**text_inputs)
            #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            #image_features = model.get_image_features(**image_inputs)
            #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            outputs = model(**inputs)
            #logits_per_image = image_features @ text_features.t()
        logits_per_image = outputs.logits_per_image
        return logits_per_image.cpu()
    
    def vqa(self, texts:list[str], images:list[Image.Image]):
        model = self.models['vqa']
        processor = self.processors['vqa']
        
        inputs = processor(text=texts, images=images, return_tensors='pt', padding=True)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs.to(self.device), 
                do_sample=False, 
                max_new_tokens=200, 
            )
            generated_text = processor.batch_decode(outputs[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return generated_text

if __name__ == '__main__':
    import requests
    device = 'cuda'
    url1 = "https://img1.baidu.com/it/u=3539595421,754041626&fm=253&fmt=auto&app=138&f=JPEG?w=889&h=500"
    url2 = "https://img0.baidu.com/it/u=2369081678,207047866&fm=253&fmt=auto&app=120&f=JPEG?w=648&h=405"
    image1 = Image.open(requests.get(url1, stream=True).raw).convert("RGB")
    image2 = Image.open(requests.get(url2, stream=True).raw).convert("RGB")
    images = [image1, image2, image1]
    prompt1 = "USER: <image>\n<image>Please describe this images.\nASSISTANT:"
    prompt2 = "USER: <image>Please describe this images.\nASSISTANT:"
    prompt = [prompt1, prompt2]
    
    batch_size = 64
    func = Functions(device=device)
    output = func.vqa(texts=prompt, images=images)
    print(output)