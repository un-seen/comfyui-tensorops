from typing import List
import numpy as np
import torchvision.transforms.functional as F
import PIL.Image
import torch
import replicate
import requests
import io
import random

def pil_image_to_file(image: PIL.Image.Image) -> io.BytesIO:

    # Convert the Pillow image to a file-like object
    image_file = io.BytesIO()
    image.save(image_file, format="PNG")  # Save image in PNG format or any other format you need
    image_id = int(random.random() * 1e15)
    image_file.name = f"{image_id}.png"  # Optional: Set a name if needed
    image_file.seek(0)  # Reset the file pointer to the beginning
    return image_file

def get_image_from_url(url: str) -> PIL.Image.Image:
    image_crop_bytes_rb = requests.get(url).content
    image_crop_rb = PIL.Image.open(io.BytesIO(image_crop_bytes_rb))
    image_crop_rb = image_crop_rb.convert("RGBA")
    return image_crop_rb

def resize_with_aspect_ratio(image, new_width):
    # Get original dimensions
    original_width, original_height = image.size

    # Calculate the new height to maintain the aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)

    # Resize the image with the new width and calculated height
    resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)

    return resized_image

class FaceSwap:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_bbox": ("BBOX",),
                "face_bbox": ("BBOX",),
                "steps": ("INT", {"default": 10, "min": 10, "max": 25}),
                "face_guidance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, image: torch.Tensor, face: torch.Tensor, prompt: str, image_bbox: torch.Tensor, face_bbox: torch.Tensor, steps: int, face_guidance: float):
        image_bbox_array = sorted(image_bbox, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]))[0]
        face_bbox_array = sorted(face_bbox, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]))[0]
        print(f"Image bbox: {image_bbox_array}")
        print(f"Face bbox: {face_bbox_array}")
        full_image_array = image.squeeze(0).cpu().numpy() * 255.0
        full_image_pil = PIL.Image.fromarray(np.clip(full_image_array, 0, 255).astype(np.uint8))
        image_select_box = (int(image_bbox_array[0]*0.925), 0, min(int(image_bbox_array[2]*1.075), full_image_pil.size[0]), min(int(image_bbox_array[3]*1.25), full_image_pil.size[1]))
        print(image_select_box)
        image_pil = full_image_pil.crop((image_select_box[0], image_select_box[1], image_select_box[2], image_select_box[3]))
        image_pil = resize_with_aspect_ratio(image_pil, 768)
        face_array = face.squeeze(0).cpu().numpy() * 255.0
        face_pil = PIL.Image.fromarray(np.clip(face_array, 0, 255).astype(np.uint8))
        # input_data = {
        #       "seed": 42,
        #       "image": "https://replicate.delivery/pbxt/LrKLR1Mwa8tXAbwgij5vqQA4w9pEuFNJp30yaDGn1qdSOx95/Screenshot%202024-10-25%20at%2011.52.55%E2%80%AFAM.png",
        #       "model": "omni-zero",
        #       "prompt": "A person, comic",
        #       "style_image": "https://replicate.delivery/pbxt/LrKLQXFxHsTWCasb2usAjB6pW5i2lMmWWIhg7idRkpGXcKkg/Screenshot%202024-10-25%20at%201.53.46%E2%80%AFPM.png",
        #       "depth_strength": 0.5,
        #       "guidance_scale": 3,
        #       "identity_image": "https://replicate.delivery/pbxt/LrKLRAvXO8x7LMv8JD0RBDwp00BDy2e0PPbfI36QzpzTl6zw/WhatsApp%20Image%202024-10-25%20at%2013.59.51.jpeg",
        #       "image_strength": 0.15,
        #       "style_strength": 1,
        #       "negative_prompt": "blurry, out of focus",
        #       "number_of_steps": 10,
        #       "number_of_images": 1,
        #       "composition_image": "https://replicate.delivery/pbxt/LrKLQYhbCVjI9MvjgvtqBwB4c0iZrLFUKAkDG7n41kU0q1RJ/Screenshot%202024-10-25%20at%2011.52.55%E2%80%AFAM.png",
        #       "identity_strength": 1,
        #       "composition_strength": 1
        # }
        output = replicate.run(
            "okaris/omni-zero:036947f1e1961875eef47a561293978528bf3a847e79fedb20600c9ad25d0c59",
            input={
                "seed": 42,
                "image": pil_image_to_file(image_pil),
                "model": "omni-zero",
                "prompt": prompt,
                "style_image": pil_image_to_file(image_pil),
                "depth_strength": 0.5,
                "guidance_scale": face_guidance,
                "identity_image": pil_image_to_file(face_pil),
                "image_strength": 0.10,
                "style_strength": 1,
                "negative_prompt": "blurry, out of focus, realism, photography",
                "number_of_images": 1,
                "composition_image": pil_image_to_file(image_pil),
                "identity_strength": 1,
                "number_of_steps": steps,
                "composition_strength": 1
            }
        )
        print(output)
        out_image = get_image_from_url(output[0])
        out_image = out_image.resize((image_select_box[2] - image_select_box[0], image_select_box[3] - image_select_box[1]))
        full_image_pil.paste(out_image, (image_select_box[0], image_select_box[1]))
        out_image = F.to_tensor(full_image_pil).permute(1, 2, 0).unsqueeze(0)
        out_image = torch.cat([out_image], dim=0)
        return (out_image, torch.cat([F.to_tensor(image_pil).permute(1, 2, 0).unsqueeze(0)], dim=0), torch.cat([F.to_tensor(face_pil).permute(1, 2, 0).unsqueeze(0)], dim=0))