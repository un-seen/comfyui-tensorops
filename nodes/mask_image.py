import torch

class MaskImage:
   
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
        mask = mask.unsqueeze(-1)
        new_image = image * mask
        print("MaskImage")
        print("ImageShape", image.shape)
        print("MaskShape", mask.shape)
        print("NewImageShape", new_image.shape)
        return (new_image,)

