"""
Excepciones custom del pipeline.
Permite diferenciar errores para respuestas HTTP y logging adecuado.
"""


class BlobNotFoundError(Exception):
    """El blob no existe en Azure Storage."""
    pass


class BlobDownloadTimeoutError(Exception):
    """Timeout al descargar el blob."""
    pass


class PDFExtractionError(Exception):
    """Error al extraer contenido del PDF (archivo corrupto o inválido)."""
    pass


class EmbeddingError(Exception):
    """Error al generar embeddings tras agotar reintentos (ej. throttling de Azure OpenAI)."""
    pass
