from .config import REDIS_URL
import redis
import numpy as np
import base64
from PIL import Image
from io import BytesIO

def pil_image_to_base64(image):
    # Create a BytesIO buffer to save the image
    buffered = BytesIO()

    # Save the image in the buffer in PNG format (you can also use JPEG or others)
    image.save(buffered, format="PNG")

    # Get the byte content of the image
    img_bytes = buffered.getvalue()

    # Encode the image bytes to base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64

class SendImageToRedis:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"key": ("STRING", {"multiline": False}), "images": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "tensorops"

    def send_images(self, key, images):
        connection = redis.Redis.from_url(REDIS_URL)
        connection.delete(key)
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            connection.xadd(key, pil_image_to_base64(image))
        connection.close()
        return ()