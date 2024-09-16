
from .config import REDIS_URL
import redis
import json

class SaveToRedis:
   
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False}),
                "data": ("JSON",)
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "database_ops"
    
    def main(self, key: str, data: dict):
        connection = redis.Redis.from_url(REDIS_URL)
        connection.set(key, json.dumps(data))
        connection.close()
        return ()

class FetchFromRedis:
   
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False})
            },
        }

    RETURN_TYPES = ("JSON",)

    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "database_ops"
    
    def main(self, key: str):
        connection = redis.Redis.from_url(REDIS_URL)
        data = connection.get(key)
        if data is None:
            return {}
        else:
            data = json.loads(data)
            return [data]
        