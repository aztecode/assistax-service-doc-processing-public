"""
Descarga de PDFs desde Azure Blob Storage.
Alineado con assistax-fn: retorna solo bytes, sin hash ni timeout.
"""
import logging

from azure.storage.blob import BlobServiceClient

from settings import settings

logger = logging.getLogger(__name__)


def download_pdf_bytes(blob_path: str) -> bytes:
    """
    Descarga el PDF desde Azure Blob Storage.

    Args:
        blob_path: Ruta del blob dentro del contenedor.

    Returns:
        Contenido del PDF como bytes.

    Raises:
        ValueError: Si blob_path está vacío.
        Exception: Errores de Azure (ResourceNotFoundError, etc.).
    """
    blob_name = blob_path.strip()
    if not blob_name:
        raise ValueError("blob_path debe ser no vacío")

    logger.debug("Descargando blob: container=%s, path=%s", settings.AZURE_BLOB_CONTAINER, blob_name[:80])

    client = BlobServiceClient.from_connection_string(
        settings.AZURE_STORAGE_CONNECTION_STRING
    )
    container_client = client.get_container_client(settings.AZURE_BLOB_CONTAINER)
    blob_client = container_client.get_blob_client(blob_name)
    download_stream = blob_client.download_blob()
    data = download_stream.readall()

    logger.debug(
        "Blob descargado: %d KB (%.2f MB) — container=%s, blob=%s",
        round(len(data) / 1024),
        len(data) / (1024 * 1024),
        settings.AZURE_BLOB_CONTAINER,
        blob_name[:80],
    )
    return data
