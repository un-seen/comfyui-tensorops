import torch

class ChannelSelector:
   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = "tensorops"

    def main(self, image, channel):
        # Select the specified channel and add a new dimension at position 0
        mask = image[channel].unsqueeze(0)

        return (mask,)

