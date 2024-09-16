import PIL
from typing import List
import numpy as np
import torchvision.transforms.functional as F
import torch

def multiply_grayscale_images(image1, image2):
    # Convert the images to NumPy arrays
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    # Perform element-wise multiplication (ensure to use np.float32 to avoid overflow)
    multiplied_image = image1_np.astype(np.float32) * image2_np.astype(np.float32)

    # Normalize the result to the range 0-255 (if needed)
    multiplied_image = np.clip(multiplied_image, 0, 255)

    # Convert back to uint8 (8-bit grayscale image)
    multiplied_image = multiplied_image.astype(np.uint8)

    # Convert back to an image and save the result
    result_image = PIL.Image.fromarray(multiplied_image)
    return result_image

def create_color_masks(image: PIL.Image.Image):
    # Load the image
    image = image.convert("RGB")
    image_np = np.array(image)  # Convert to numpy array (Height x Width x 3)
    # Find unique colors in the image
    unique_colors = np.unique(image_np.reshape(-1, 3), axis=0)
    output = []
    # Create masks for each color
    for color in unique_colors:
        if sum(color) == 0:
            continue
        mask = np.all(image_np == color, axis=-1)
        color_str = '_'.join(map(str, color))  # Create a string representation of the color
        output.append((color_str, mask))
    # Skip Background Mask Image
    background_area = 0.0
    background_mask_index = -1
    for idx, (color_str, mask) in enumerate(output):
        area = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        if area > background_area:
            background_area = area
            background_mask_index = idx
    # Final Elements
    elements = []
    for idx, (color_str, mask) in enumerate(output):
        if idx == background_mask_index:
            print(background_mask_index)
            continue
        mask_image = PIL.Image.fromarray(mask.astype(np.uint8) * 255)
        elements.append((color_str, mask_image))
    # Final Background
    final_background_mask_image = PIL.Image.new("L", (image.size[0], image.size[1]), 255)
    draw = PIL.ImageDraw.Draw(final_background_mask_image)
    for idx, (color_str, mask_image) in enumerate(elements):
        final_background_mask_image = multiply_grayscale_images(final_background_mask_image, PIL.ImageOps.invert(mask_image))

    return final_background_mask_image, elements


def create_text_masks(polygons, width, height):
    # Loop over each polygon in the list
    text_masks = []
    for i, polygon_coords in enumerate(polygons):
        # Create a new grayscale image (L mode) with a black background (0)
        mask = PIL.Image.new('L', (width, height), 0)

        # Create a drawing object
        draw = PIL.ImageDraw.Draw(mask)

        # Convert the list of polygon coordinates into a format ImageDraw can use (list of tuples)
        polygon_points = [(polygon_coords[j], polygon_coords[j + 1]) for j in range(0, len(polygon_coords), 2)]

        # Draw the polygon with white (255) fill
        draw.polygon(polygon_points, fill=255)
        text_masks.append(mask)
    return text_masks

class GetLayerMask:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "json_data": ("JSON",),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "JSON")

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, image: torch.Tensor, json_data: str):
        # Create PIL.Image
        image = image.permute(0, 3, 1, 2)
        image_pil = F.to_pil_image(image[0])
        # Create bg and elements
        bg, elements = create_color_masks(image_pil)
        # Create Text Masks
        print("items", json_data)
        items = [item for item in json_data]
        text_polygon_list = []
        text_label_list = []
        text_masks = []

        for item in items:
            text_polygon_list.append(item["polygon"])
            text_label_list.append(item["label"])

        for mask_image in create_text_masks(text_polygon_list, bg.size[0], bg.size[1]):
            img = np.array(mask_image).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            text_masks.append(img)

        output = []
        bg = np.array(bg).astype(np.float32) / 255.0
        bg = torch.from_numpy(bg)[None,]
        output.append(bg)
        for _, mask_image in elements:
            img = np.array(mask_image).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            output.append(img)
        return (torch.cat(output, dim=0), torch.cat(text_masks, dim=0), text_label_list)
