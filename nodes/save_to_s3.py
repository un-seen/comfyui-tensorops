from .config import DATALAKE_AWS_ACCESS_KEY_ID, DATALAKE_AWS_ENDPOINT_URL, DATALAKE_AWS_SECRET_ACCESS_KEY, BUCKET
import boto3
from botocore.client import Config
import torch
import PIL
import PIL.Image
import io
import logging
import numpy as np

logger = logging.getLogger(__name__)

S3_RESOURCE = None

def init_s3():
    global S3_RESOURCE
    if not S3_RESOURCE:
        S3_ENDPOINT_URL = DATALAKE_AWS_ENDPOINT_URL
        AWS_ACCESS_KEY_ID = DATALAKE_AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = DATALAKE_AWS_SECRET_ACCESS_KEY
        # Initialize the S3 client
        S3_RESOURCE = boto3.resource(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )

def store_image(key: str, image: PIL.Image.Image):
    global S3_RESOURCE
    init_s3()
    try:
        s3_bucket = S3_RESOURCE.Bucket(BUCKET)
        file_content = io.BytesIO()
        image.save(file_content, format="webp")
        file_content.seek(0)
        logger.info(f"StoreImage: {key} in {BUCKET}")
        s3_bucket.put_object(Key=key, Body=file_content)
    except Exception as e:
        logger.error(f"StoreContentError {key}: {e}")

class SaveImageToS3:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "database": ("STRING", {"multiline": False}),
                "key": ("STRING", {"multiline": False}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "database_ops"

    def main(self, database: str, key: str, image: torch.Tensor):
        B = image.shape[0]
        for i in range(B):
            im = image[i]
            img_array = im.squeeze(0).cpu().numpy() * 255.0
            img_pil = PIL.Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            if i > 0:
                final_key = f"{database}/{key}-{i}.webp"
            else:
                final_key = f"{database}/{key}.webp"
            print("final_key", final_key)
            store_image(final_key, img_pil)
        return ()