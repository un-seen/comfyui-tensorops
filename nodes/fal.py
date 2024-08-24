from PIL import Image
import torch
import requests
from io import BytesIO
import numpy as np
import fal_client



class FalDifferentialDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "foreground_image": ("IMAGE", ),
                "depth_image": ("IMAGE", ),
                "foreground_prompt": ("STRING", {"multiline": False}),
                "background_prompt": ("STRING", {"multiline": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "external_tooling"
    FUNCTION = "load"

    def load(self, foreground_image: torch.Tensor, depth_image: torch.Tensor, foreground_prompt: str, background_prompt: str):
        # Foreground Image
        foreground_image_array = foreground_image.squeeze(0).cpu().numpy() * 255.0
        foreground_image_pil = Image.fromarray(np.clip(foreground_image_array, 0, 255).astype(np.int))
        foreground_output = BytesIO()
        foreground_image_pil.save(foreground_output, format='PNG')
        foreground_url = fal_client.upload(foreground_output, "image/png")
        # Depth Image
        depth_image_array = depth_image.squeeze(0).cpu().numpy() * 255.0
        depth_image_pil = Image.fromarray(np.clip(depth_image_array, 0, 255).astype(np.int))
        depth_output = BytesIO()
        depth_image_pil.save(depth_output, format='PNG')
        depth_url = fal_client.upload(depth_output, "image/png")
        # Fal handler
        handler = fal_client.submit(
            "fal-ai/flux-differential-diffusion",
            arguments={
                "prompt": f"{foreground_prompt}, {background_prompt}, 8k, unreal engine 5, hightly detailed, intricate detailed.",
                "image_url": foreground_url,
                "change_map_image_url": depth_url
            },
        )
        result = handler.get()
        images = []
        for image in result['images']:
            url = image['url']
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = np.array(img).astype(np.float) / 255.0
            img = torch.from_numpy(img)
            images.append(img)
        return (images,)
    