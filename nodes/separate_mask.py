from typing import List
import numpy as np
import torchvision.transforms.functional as F
import PIL.Image
import PIL.ImageChops
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


def get_padded_image(src_image: PIL.Image.Image, desired_width: int, desired_height: int) -> PIL.Image.Image:
    # Get the dimensions of the original masked image
    original_width, original_height = src_image.size

    # Create a new image with the desired dimensions and a transparent background
    padded_image = PIL.Image.new("RGBA", (desired_width, desired_height), (0, 0, 0, 0))

    # Calculate the position to paste the masked image (center it in the new padded image)
    x_offset = (desired_width - original_width) // 2
    y_offset = (desired_height - original_height) // 2

    # Paste the original masked image onto the padded image
    padded_image.paste(src_image, (x_offset, y_offset), src_image)
    return padded_image

def select_element(
    src_image: PIL.Image.Image,
    mask_image: PIL.Image.Image,
    bboxes
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
    cropped_rgbs = []
    area_min_threshold = 1000
    fixed_width = max(box[2]-box[1] for box in bboxes)
    fixed_height = max(box[3]-box[1] for box in bboxes)
    fixed_width = int(fixed_width)
    fixed_height = int(fixed_height)
    for bbox in bboxes:
        print(bbox)
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        # Crop the bounding box area from the mask and RGBA images
        mask_crop = mask_image.crop((x_min, y_min, x_max, y_max))
        rgba_crop = src_image.crop((x_min, y_min, x_max, y_max))

        # Apply the mask to the alpha channel of the RGBA image
        r, g, b, a = rgba_crop.split()
        a = PIL.Image.composite(a, mask_crop, mask_crop)
        masked_image = PIL.Image.merge("RGBA", (r, g, b, a))

        if masked_image.size[0] * masked_image.size[1] > area_min_threshold:
            masked_output = PIL.Image.new("RGBA", src_image.size, (255, 255, 255, 255))
            masked_output.paste(masked_image, (x_min, y_min))
            masked_output = masked_output.split()[3]
            data.append(masked_output)
            fixed_size_image = get_padded_image(PIL.Image.merge("RGBA", (r, g, b, PIL.ImageChops.invert(a))), fixed_width, fixed_height)
            cropped_rgbs.append(fixed_size_image)
    return data, cropped_rgbs


class SeparateMask:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "bboxes": ("BBOX",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, image: torch.Tensor, mask: torch.Tensor, bboxes):
        img_array = image.squeeze(0).cpu().numpy() * 255.0
        mask_array = mask.squeeze(0).cpu().numpy() * 255.0
        mask_pil = PIL.Image.fromarray(np.clip(mask_array, 0, 255).astype(np.uint8))
        img_pil = PIL.Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8)).convert("RGBA")
        masks, images = select_element(img_pil, mask_pil, bboxes)
        masks_items = [F.to_tensor(item) for item in masks]
        images_items = [F.to_tensor(item).permute(1, 2, 0).unsqueeze(0) for item in images]
        out_mask = torch.cat(masks_items, dim=0)
        out_image = torch.cat(images_items, dim=0)
        print(out_image.shape)
        return (out_mask, out_image,)
