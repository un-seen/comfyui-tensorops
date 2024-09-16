from .channel_select import ChannelSelector
from .mask_image import MaskImage
from .save_surreal import SaveJsonToSurreal, SaveTextToSurreal
from .fetch_surreal import FetchJsonFromSurreal
from .foreground_mask import ForegroundMask
from .save_to_s3 import SaveImageToS3
from .redis import SaveToRedis, FetchFromRedis
from .fal import FalDifferentialDiffusion, FalDiffusion
from .background_select import BackgroundSelect
from .layer_mask import GetLayerMask

NODE_CLASS_MAPPINGS = {
    "ChannelSelector": ChannelSelector,
    "MaskImage": MaskImage,
    "SaveImageToS3": SaveImageToS3,
    "SaveJsonToSurreal": SaveJsonToSurreal,
    "SaveTextToSurreal": SaveTextToSurreal,
    "FetchJsonFromSurreal": FetchJsonFromSurreal,
    "ForegroundMask": ForegroundMask,
    "SaveToRedis": SaveToRedis,
    "FetchFromRedis": FetchFromRedis,
    "FalDifferentialDiffusion": FalDifferentialDiffusion,
    "FalDiffusion": FalDiffusion,
    "BackgroundSelect": BackgroundSelect,
    "GetLayerMask": GetLayerMask,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChannelSelector":"ChannelSelector",
    "MaskImage": "MaskImage",
    "SaveImageToS3": "SaveImageToS3",
    "SaveJsonToSurreal": "SaveJsonToSurreal",
    "SaveTextToSurreal": "SaveTextToSurreal",
    "FetchJsonFromSurreal": "FetchJsonFromSurreal",
    "ForegroundMask": "ForegroundMask",
    "SaveToRedis": "SaveToRedis",
    "FetchFromRedis": "FetchFromRedis",
    "FalDifferentialDiffusion": "FalDifferentialDiffusion",
    "FalDiffusion": "FalDiffusion",
    "BackgroundSelect": "BackgroundSelect",
    "GetLayerMask": "GetLayerMask",
}
