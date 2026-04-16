"""
Generación de embeddings para chunks vía Azure OpenAI.
Retry con backoff exponencial, semáforo para evitar throttling.
"""
import logging
import threading
import time
from typing import Callable

from openai import AzureOpenAI
from openai import RateLimitError

from pipeline.exceptions import EmbeddingError
from pipeline.legal_chunker import Chunk
from settings import settings

logger = logging.getLogger(__name__)

# Semáforo: máximo N llamadas concurrentes a OpenAI (OPENAI_MAX_CONCURRENT)
_openai_semaphore = threading.Semaphore(settings.OPENAI_MAX_CONCURRENT)

# 5 intentos con backoff exponencial (segundos)
RETRY_DELAYS = [1, 2, 4, 8, 16]


def embed_chunks(
    chunks: list[Chunk],
    batch_size: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """
    Genera embeddings para todos los chunks vía Azure OpenAI.

    Args:
        chunks: Lista de chunks a embeber.
        batch_size: Tamaño de lote por request (máx 2048 para Azure).
        progress_callback: Opcional, recibe (processed, total) tras cada batch.

    Returns:
        Lista de vectores (1536 dims) en mismo orden que chunks.

    Raises:
        EmbeddingError: Si tras 5 reintentos el batch falla.
    """
    if not chunks:
        return []

    client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )
    results: list[list[float] | None] = [None] * len(chunks)

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        # Evitar strings vacíos (Azure rechaza input vacío)
        texts = [c.text.strip() if c.text.strip() else " " for c in batch]

        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                with _openai_semaphore:
                    response = client.embeddings.create(
                        model=settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
                        input=texts,
                    )
                # Reordenar por índice por si Azure responde fuera de orden
                for item in response.data:
                    # index es 0-based dentro del batch
                    rel_idx = getattr(item, "index", response.data.index(item))
                    results[batch_start + rel_idx] = item.embedding
                break
            except RateLimitError as e:
                retry_after = getattr(e, "retry_after", None)
                wait_time = float(retry_after) if retry_after is not None else delay
                if attempt < len(RETRY_DELAYS) - 1:
                    logger.warning(
                        "embed.rate_limit: reintentando batch %d (intento %d/%d, espera %.1fs)",
                        batch_start,
                        attempt + 1,
                        len(RETRY_DELAYS),
                        wait_time,
                    )
                time.sleep(wait_time)
                if attempt == len(RETRY_DELAYS) - 1:
                    raise EmbeddingError(
                        f"Batch {batch_start} falló tras {len(RETRY_DELAYS)} intentos por rate limit"
                    ) from e
            except Exception as e:
                if attempt < len(RETRY_DELAYS) - 1:
                    logger.warning(
                        "embed.error: reintentando batch %d (intento %d/%d): %s",
                        batch_start,
                        attempt + 1,
                        len(RETRY_DELAYS),
                        e,
                    )
                if attempt == len(RETRY_DELAYS) - 1:
                    raise EmbeddingError(
                        f"Batch {batch_start} falló tras {len(RETRY_DELAYS)} intentos: {e}"
                    ) from e
                time.sleep(delay)

        if progress_callback:
            processed = min(batch_start + batch_size, len(chunks))
            progress_callback(processed, len(chunks))

    if any(r is None for r in results):
        raise EmbeddingError("Algunos embeddings no se generaron correctamente")
    return results
