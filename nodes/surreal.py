from surrealist import Surreal
from .config import SURREAL_URL, SURREAL_NAMESPACE, SURREAL_USER, SURREAL_PASSWORD

def surreal_connect(database: str):
    surreal_client = Surreal(SURREAL_URL, namespace=SURREAL_NAMESPACE, database=database, credentials=(SURREAL_USER, SURREAL_PASSWORD), use_http=True, timeout=10)
    surreal_connection =  surreal_client.connect()
    return surreal_connection