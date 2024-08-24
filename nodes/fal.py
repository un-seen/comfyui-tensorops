from PIL import Image
import torch
import requests
from io import BytesIO
import numpy as np
import fal_client

class FalDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": False}),
                "steps": ("INT",{"default": 2, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "external_tooling"
    FUNCTION = "load"

    def load(self, prompt: str, steps: int):
        # Fal handler
        handler = fal_client.submit(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": f"{prompt}",
                "image_size": "square_hd",
                "num_inference_steps": steps,
            },
        )
        result = handler.get()
        images = []
        for image in result['images']:
            url = image['url']
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            images.append(img)
        return (torch.cat(images, dim=0),)


class FalDifferentialDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "foreground_image": ("IMAGE", ),
                "depth_image": ("IMAGE", ),
                "foreground_prompt": ("STRING", {"multiline": False}),
                "background_prompt": ("STRING", {"multiline": False}),
                "strength": ("FLOAT",{"default": 1, "min": 0.01, "max": 3, "step": 0.01}),
                "steps": ("INT",{"default": 14, "min": 1, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "external_tooling"
    FUNCTION = "load"

    def load(self, foreground_image: torch.Tensor, depth_image: torch.Tensor, foreground_prompt: str, background_prompt: str, strength: float, steps: int):
        # Foreground Image
        foreground_image_array = foreground_image.squeeze(0).cpu().numpy() * 255.0
        foreground_image_pil = Image.fromarray(np.clip(foreground_image_array, 0, 255).astype(np.uint8))
        foreground_output = BytesIO()
        foreground_image_pil.save(foreground_output, format='PNG')
        foreground_url = fal_client.upload(foreground_output.getvalue(), "image/png")
        # Depth Image
        depth_image_array = depth_image.squeeze(0).cpu().numpy() * 255.0
        depth_image_pil = Image.fromarray(np.clip(depth_image_array, 0, 255).astype(np.uint8))
        depth_output = BytesIO()
        depth_image_pil.save(depth_output, format='PNG')
        depth_url = fal_client.upload(depth_output.getvalue(), "image/png")
        # Fal handler
        handler = fal_client.submit(
            "fal-ai/flux-differential-diffusion",
            arguments={
                "prompt": f"{foreground_prompt}, {background_prompt}, 8k, unreal engine 5, hightly detailed, intricate detailed.",
                "image_url": foreground_url,
                "change_map_image_url": depth_url,
                "strength": strength,
                "num_inference_steps": steps,
            },
        )
        result = handler.get()
        images = []
        for image in result['images']:
            url = image['url']
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            images.append(img)
        return (torch.cat(images, dim=0),)