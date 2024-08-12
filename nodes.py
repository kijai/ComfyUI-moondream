import os
from PIL import Image
import torch
import numpy as np
import comfy.utils
import comfy.model_management

from .moondream import Moondream
from transformers import AutoTokenizer, AutoModelForCausalLM

script_directory = os.path.dirname(os.path.abspath(__file__))

class MoondreamQuery:
    def __init__(self):
        self.selected_model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "images": ("IMAGE", ),
            "question": ("STRING", {"multiline": True, "default": "What is this?",}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "model": (
            [   
                'moondream1',
                'moondream2',
            ], {
               "default": 'moondream2'
            }),
            
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 256}),
            },
            }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("text",)
    FUNCTION = "process"

    CATEGORY = "Moondream"

    def process(self, images, question, keep_model_loaded, model, max_new_tokens=256):
        batch_size = images.shape[0]
        device = comfy.model_management.get_torch_device()
        dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(device) else torch.float32
        
        checkpoint_path = os.path.join(script_directory, "checkpoints", model)

        if not hasattr(self, "moondream") or self.moondream is None or self.selected_model != model:
            self.selected_model = model
            model_safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_safetensors_path):
                checkpoint_path = checkpoint_path
            else:
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=f"vikhyatk/{model}", ignore_patterns=["*.jpg","*.pt","*.bin", "*0000*"],local_dir=checkpoint_path, local_dir_use_symlinks=False)
                except:
                    raise FileNotFoundError("No model found.")

            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.moondream = Moondream.from_pretrained(checkpoint_path, ignore_mismatched_sizes=True).to(device=device, dtype=dtype)
            self.moondream.eval()

        answer_dict = {}
        if batch_size > 1:
            for i in range(batch_size):
                image = Image.fromarray(np.clip(255. * images[i].cpu().numpy(),0,255).astype(np.uint8))
                image_embeds = self.moondream.encode_image(image)
                answer = self.moondream.answer_question(image_embeds, question, self.tokenizer, max_new_tokens)
                answer_dict[str(i)] = answer

            formatted_answers = ",\n".join([f'"{frame}" : "{answer}"' for frame, answer in answer_dict.items()])
            formatted_output = "{\n" + formatted_answers + "\n}"
            print(formatted_output)
            answer = formatted_output
            return formatted_output,
        else:
            image = Image.fromarray(np.clip(255. * images[0].cpu().numpy(),0,255).astype(np.uint8))
            image_embeds = self.moondream.encode_image(image)
            answer = self.moondream.answer_question(image_embeds, question, self.tokenizer, max_new_tokens)

        if not keep_model_loaded:
            self.moondream = None
            self.tokenizer = None
            comfy.model_management.soft_empty_cache()
        return answer,
    
class MoondreamQueryCaptions:
    def __init__(self):
        self.selected_model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "images": ("IMAGE", ),
            "question": ("STRING", {"multiline": True, "default": "What is this?",}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "model": (
            [   
                'moondream1',
                'moondream2',
            ], {
               "default": 'moondream2'
            }),
            
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 256}),
            },
            }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES =("text",)
    FUNCTION = "process"

    CATEGORY = "Moondream"

    def process(self, images, question, keep_model_loaded, model, max_new_tokens=256):
        batch_size = images.shape[0]
        device = comfy.model_management.get_torch_device()
        dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(device) else torch.float32
        
        checkpoint_path = os.path.join(script_directory, f"checkpoints/{model}")

        if not hasattr(self, "moondream") or self.moondream is None or self.selected_model != model:
            self.selected_model = model
            model_safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_safetensors_path):
                checkpoint_path = checkpoint_path
            else:
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=f"vikhyatk/{model}", ignore_patterns=["*.jpg","*.pt","*.bin", "*0000*"],local_dir=checkpoint_path, local_dir_use_symlinks=False)
                except:
                    raise FileNotFoundError("No model found.")

            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.moondream = Moondream.from_pretrained(checkpoint_path).to(device=device, dtype=dtype)
            self.moondream.eval()

        answer_list = []
        for i in range(batch_size):
            image = Image.fromarray(np.clip(255. * images[i].cpu().numpy(),0,255).astype(np.uint8))
            image_embeds = self.moondream.encode_image(image)
            answer = self.moondream.answer_question(image_embeds, question, self.tokenizer, max_new_tokens)
            answer_list.append(answer)
        for answer in answer_list:
            print(answer)

        if not keep_model_loaded:
            self.moondream = None
            self.tokenizer = None
            comfy.model_management.soft_empty_cache()
        return answer_list,       

        


NODE_CLASS_MAPPINGS = {
    "MoondreamQuery": MoondreamQuery,
    "MoondreamQueryCaptions": MoondreamQueryCaptions,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MoondreamQuery": "MoondreamQuery",
    "MoondreamQueryCaptions": "MoondreamQueryCaptions",
}