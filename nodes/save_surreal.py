from .surreal import surreal_connect

SURREAL_TABLE = "processor"

class SaveJsonToSurreal:
   
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "database": ("STRING", {"multiline": False}),
                "json": ("JSON",),
                "id": ("STRING", {"multiline": False}),
                "key": ("STRING", {"multiline": False})
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "database_ops"
    
    def main(self, database: str, id: str, key: str, json: str):
        connection = surreal_connect(database)
        query = f"UPDATE {SURREAL_TABLE}:`{id}` CONTENT {{{key}: {json}}};"
        connection.query(query)
        return ()

class SaveTextToSurreal:
   
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "database": ("STRING", {"multiline": False}),
                "text":  ("STRING",{"forceInput": True}),
                "id": ("STRING", {"multiline": False}),
                "key": ("STRING", {"multiline": False})
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "database_ops"

    def main(self, database: str, id: str, key: str, text: str):
        connection = surreal_connect(database)
        query = f"UPDATE {SURREAL_TABLE}:`{id}` CONTENT {{{key}: '{text}'}};"
        connection.query(query)
        return ()