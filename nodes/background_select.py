import torch


def get_background_mask(tensor: torch.Tensor):
    """
    Function to identify the background mask from a batch of masks in a PyTorch tensor.

    Args:
        tensor (torch.Tensor): A tensor of shape (B, H, W, 1) where B is the batch size, H is the height, W is the width.

    Returns:
        List of masks as torch.Tensor and the background mask as torch.Tensor.
    """
    B, H, W = tensor.shape

    # Compute areas of each mask
    areas = tensor.sum(dim=(1, 2))  # Shape: (B,)
    
    # Find the mask with the largest area
    largest_idx = torch.argmax(areas)
    background_mask = tensor[largest_idx]

    # Identify if the largest mask touches the borders
    border_touched = (
        torch.any(background_mask[0, :]) or
        torch.any(background_mask[-1, :]) or
        torch.any(background_mask[:, 0]) or
        torch.any(background_mask[:, -1])
    )

    # If the largest mask doesn't touch the border, search for another one
    if not border_touched:
        for i in range(B):
            if i != largest_idx:
                mask = tensor[i]
                border_touched = (
                    torch.any(mask[0, :]) or
                    torch.any(mask[-1, :]) or
                    torch.any(mask[:, 0]) or
                    torch.any(mask[:, -1])
                )
                if border_touched:
                    background_mask = mask
                    break
    
    # Reshape the masks to match the original tensor shape
    return background_mask

class BackgroundSelect:
   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, mask: torch.Tensor):
        # TODO loop through all masks
        # identify the background mask
        # return the background mask
        background_mask = get_background_mask(mask)
        return (background_mask,)

