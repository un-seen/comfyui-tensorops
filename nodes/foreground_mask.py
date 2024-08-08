from typing import List
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter
import torchvision.transforms.functional as F
import json

def calculate_bounding_box(points) -> List[float]:
    """
    Calculate the bounding box for a polygon.

    Args:
    flat_points (list of int): Flat list of x, y coordinates defining the polygon points.

    Returns:
    tuple: (min_x, min_y, max_x, max_y) defining the bounding box.
    """
    if not points or len(points) % 2 != 0:
        raise ValueError("The list of points must be non-empty and have an even number of elements")

    x_coords = points[0::2]
    y_coords = points[1::2]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    return [min_x, min_y, max_x, max_y]


def find_mode_color(image: Image.Image):
    """
    Identify the most frequent (mode) color in a PIL image.
    
    Parameters:
    image_path (str): The path to the input image.
    
    Returns:
    tuple: The mode color in the image as an (R, G, B) tuple.
    """
    # Convert image to RGB mode if it's not already
    image = image.convert('RGB')
    
    # Get the list of pixels
    pixels = list(image.getdata())
    
    # Use Counter to count the frequency of each color
    counter = Counter(pixels)
    
    # Find the most common color
    mode_color = counter.most_common(1)[0][0]
    
    return mode_color

def separate_foreground_background(image):
    """
    Separate the Pillow image into foreground and background using the mode color and distance clustering.
    
    Parameters:
    image_path (str): The path to the input image.
    output_foreground (str): The path to save the foreground image.
    output_background (str): The path to save the background image.
    
    Returns:
    None
    """
    # Convert image to RGBA mode to handle transparency
    image = image.convert('RGBA')
    pixels = np.array(image)
    
    # Calculate the Euclidean distance of each pixel to the mode color
    background_color = find_mode_color(image)
    print("Background color:", background_color)
    mode_color_array = np.array(background_color)
    distances = np.linalg.norm(pixels[:, :, :3] - mode_color_array, axis=2)
    
    # Determine the threshold distance for clustering
    threshold_distance = np.mean(distances)
    
    print("Threshold distance:", threshold_distance)
    # Create masks for foreground and background
    foreground_mask = distances > threshold_distance
    background_mask = distances <= threshold_distance
    
    # Create empty arrays for the new images
    foreground_image = np.zeros_like(pixels)
    background_image = np.zeros_like(pixels)
    
    # Copy the pixels to the new images based on the masks
    foreground_image[foreground_mask] = pixels[foreground_mask]
    background_image[background_mask] = pixels[background_mask]
    
    # Find the fg color
    fg_color = find_mode_color(Image.fromarray(foreground_image, 'RGBA'))
    
    # Set foreground pixels with alpha == 255 to black
    alpha_channel = foreground_image[:, :, 3] == 255
    foreground_image[alpha_channel, :3] = [255, 255, 255]
    foreground_image[:, :, 3] = 255

    # Convert back to PIL images
    foreground_image = Image.fromarray(foreground_image, 'RGBA')
    background_image = Image.fromarray(background_image, 'RGBA')
    
    # Invert Foreground As White
    # foreground_image = ImageOps.invert(foreground_image.convert("RGB"))
    
    return foreground_image, fg_color

def crop_polygon(image, points):
    """
    Create a white mask on a black image of size width x height using a list of polygon points.

    Args:
    points (list of tuples): List of (x, y) tuples defining the polygon points.
    width (int): Width of the image.
    height (int): Height of the image.

    Returns:
    Image: Pillow Image object with the polygon mask.
    """
    x_min, y_min, x_max, y_max = calculate_bounding_box(points)
    image_crop = image.crop((x_min, y_min, x_max, y_max))
    return image_crop
    
def mask_polygon(image, points):
    """
    Crop a polygon from a Pillow image.

    Args:
    image (PIL.Image): The input image.
    flat_points (list of int): Flat list of x, y coordinates defining the polygon points.

    Returns:
    PIL.Image: Cropped image of the polygon.
    """
    if not points or len(points) % 2 != 0:
        raise ValueError("The list of points must be non-empty and have an even number of elements")

    # Create a mask
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    new_box = (np.array(points) * 1.0).tolist()
    draw.polygon(new_box, fill="white")

    # Apply the mask to the image
    masked_image = Image.composite(image.convert("RGBA"), mask.convert("RGBA"), mask)
    return masked_image


import torch

class ForegroundMask:
   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "json_data": ("JSON",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, image: torch.Tensor, json_data: str):
        print("items", json_data)
        items = [item for item in json_data]
        image = image.permute(0, 3, 1, 2)
        image_pil = F.to_pil_image(image[0])
        full_image = Image.new("RGBA", image_pil.size, (0, 0, 0, 255))
        for item in items:
            points = item["polygon"]
            print("polygon", points)
            masked_image = mask_polygon(image_pil, points)
            masked_image_crop = crop_polygon(image_pil, points)
            fg_image, fg_color = separate_foreground_background(masked_image_crop)
            x_min, y_min, x_max, y_max = calculate_bounding_box(points)
            full_image.paste(fg_image, (int(x_min), int(y_min)))
        out_image = F.to_tensor(full_image)
        return (out_image,)

