from typing import List
import numpy as np
import torchvision.transforms.functional as F
import PIL.Image
from typing import List, Tuple
import torch
from scipy.ndimage import label, find_objects


def find_bounding_boxes(image: PIL.Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Find the smallest non-overlapping bounding boxes that contain all white values in a grayscale image.

    Parameters:
    image_path (str): Path to the grayscale image.

    Returns:
    list: A list of tuples representing the bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    # Load image and convert to numpy array
    image = image.convert("L")
    image_array = np.array(image)

    # Create a binary mask of white pixels
    binary_mask = image_array > 0

    # Label connected components
    labeled_array, num_features = label(binary_mask)

    # Find bounding boxes for each labeled region
    bounding_boxes = []
    slices = find_objects(labeled_array)

    for slice_tuple in slices:
        if slice_tuple is not None:
            y_min, y_max = slice_tuple[0].start, slice_tuple[0].stop
            x_min, x_max = slice_tuple[1].start, slice_tuple[1].stop
            bounding_boxes.append((x_min, y_min, x_max, y_max))

    return bounding_boxes

def select_element(
    src_image: PIL.Image.Image,
    mask_image: PIL.Image.Image,    
) -> List[PIL.Image.Image]:
    """
    Select an element from an element image and place it on a background image using a mask.

    Parameters:
    mask_image (PIL.Image.Image): A binary mask image.
    background_image (PIL.Image.Image): A background image.
    element_image (PIL.Image.Image): An element image.

    Returns:
    PIL.Image.Image: The composite image with the element placed on the background.
    """
    mask_image = mask_image.convert("L")
    data = []
    area_min_threshold = 1000
    for bbox in find_bounding_boxes(mask_image):
        x_min, y_min, x_max, y_max = bbox

        # Crop the bounding box area from the mask and RGBA images
        mask_crop = mask_image.crop((x_min, y_min, x_max, y_max))
        rgba_crop = src_image.crop((x_min, y_min, x_max, y_max))

        # Apply the mask to the alpha channel of the RGBA image
        r, g, b, a = rgba_crop.split()
        a = PIL.Image.composite(a, mask_crop, mask_crop)
        masked_image = PIL.Image.merge("RGBA", (r, g, b, a))

        if masked_image.size[0] * masked_image.size[1] > area_min_threshold:
            data.append(masked_image)

    return data


class SeparateMask:
   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, image: torch.Tensor, mask: torch.Tensor):
        items = [item for item in select_element(image, mask)]
        out_image = F.to_tensor(items)
        return (out_image,)

