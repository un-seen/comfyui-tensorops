from typing import List
import numpy as np
import torchvision.transforms.functional as F
import PIL.Image
import torch
import replicate
import requests

def get_image_from_url(url: str) -> PIL.Image.Image:
    image_crop_bytes_rb = requests.get(url).content
    image_crop_rb = PIL.Image.open(io.BytesIO(image_crop_bytes_rb))
    image_crop_rb = image_crop_rb.convert("RGBA")
    return image_crop_rb

class FaceSwap:
   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face": ("IMAGE",),
                "prompt": ("STRING",),
                "image_bbox": ("BBOX",),
                "face_bbox": ("BBOX",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, image: torch.Tensor, face: torch.Tensor, prompt: str, image_bbox: torch.Tensor, face_bbox: torch.Tensor):
        image_bbox_array = image_bbox.cpu().numpy()
        face_bbox_array = face_bbox.cpu().numpy()
        print(f"Image bbox: {image_bbox_array}")
        print(f"Face bbox: {face_bbox_array}")
        image_array = image.squeeze(0).cpu().numpy() * 255.0
        image_pil = PIL.Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))
        image_pil = image_pil.crop((image_bbox[0], image_bbox[1], image_bbox[2], image_bbox[3]))
        face_array = face.squeeze(0).cpu().numpy() * 255.0
        face_pil = PIL.Image.fromarray(np.clip(face_array, 0, 255).astype(np.uint8))
        face_pil = face_pil.crop((face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]))
        
        output = replicate.run(
            "okaris/omni-zero:036947f1e1961875eef47a561293978528bf3a847e79fedb20600c9ad25d0c59",
            input={
                "seed": 42,
                "image": image_pil,
                "model": "omni-zero",
                "prompt": prompt,
                "style_image": image_pil,
                "depth_strength": 0.5,
                "guidance_scale": 3.5,
                "identity_image": face_pil,
                "image_strength": 0.15,
                "style_strength": 1,
                "negative_prompt": "blurry, out of focus, realism, photography",
                "number_of_steps": 10,
                "number_of_images": 1,
                "composition_image": image_pil,
                "identity_strength": 1,
                "composition_strength": 1
            }
        )
        
        out_image = get_image_from_url(output[0])
        out_image = F.to_tensor(out_image)
        return (out_image,)

