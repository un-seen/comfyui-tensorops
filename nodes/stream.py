from __future__ import annotations
from PIL import Image
import numpy as np
from server import PromptServer, BinaryEventTypes

class SendImageOnWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"event": ("STRING", {"multiline": False}), "images": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "tensorops"

    def send_images(self, event, images):
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                ["PNG", image, event],
                server.client_id,
            )
        return ()
    
    
class SendJsonOnWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"event": ("STRING", {"multiline": False}), "json": ("JSON",)}}

    RETURN_TYPES = ()
    FUNCTION = "send_json"
    OUTPUT_NODE = True
    CATEGORY = "tensorops"

    def send_json(self, event, json):
        server = PromptServer.instance
        server.send_sync(
            event,
            json,
            server.client_id,
        )
        return ()