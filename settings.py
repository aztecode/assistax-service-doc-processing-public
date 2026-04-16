# settings.py — Validación fail-fast de variables de entorno
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Requeridas — la app no arranca si faltan
    DATABASE_URL: str
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: str
    FUNCTIONS_API_KEY: str

    # Opcionales con defaults ajustados al volumen real (máx 30 docs)
    AZURE_BLOB_CONTAINER: str = "laws"
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = "gpt-4o-mini"
    PDF_WORKER_THREADS: int = 2  # 2 pipelines simultáneos máximo
    DB_POOL_MIN_CONN: int = 1
    DB_POOL_MAX_CONN: int = 5  # 5 conexiones cubre holgado para 2 workers
    OPENAI_MAX_CONCURRENT: int = 2  # Evita throttling en Azure OpenAI
    ENABLE_LLM_DOC_TYPE: bool = True
    ENABLE_LLM_HEADING_CLASSIFIER: bool = True
    ENABLE_LLM_GENERIC_HEADING_REFINE: bool = False
    # When True, run LLM on every generic chunk without articleRef (higher API cost).
    LLM_GENERIC_HEADING_REFINE_ALL: bool = False
    ENABLE_LLM_BOXED_NOTE_ARBITER: bool = True
    # Phase 3 — block classifier v2 (heuristic + LLM hybrid)
    ENABLE_BLOCK_LLM_CLASSIFIER_V2: bool = True
    BLOCK_LLM_MAX_CONCURRENT: int = 2
    BLOCK_LLM_BATCH_SIZE: int = 8
    ENABLE_PDF_TEXT_NORMALIZATION: bool = True
    # Collapse single blank lines between prose lines (DOF bold/regular block splits)
    ENABLE_DECRETO_PROSE_BLANK_COLLAPSE: bool = True
    # PDF extraction tuning (see pipeline/pdf_extractor.py, scripts/pdf_corpus_table_audit.py)
    RELAXED_VISUAL_FRAME_DETECTION: bool = False
    RELAX_PROSE_TABLE_FILTER: bool = False
    CLEANUP_INTERVAL_MINUTES: int = 10
    WEBSITES_PORT: int = 8000  # Azure Web App inyecta esta variable
    LOG_FORMAT: str = "json"
    ENVIRONMENT: str = "production"
    # Phase 7 — pipeline v2 integration flags
    ENABLE_LAYOUT_V2: bool = False
    LAYOUT_V2_SHADOW_MODE: bool = False
    LAYOUT_V2_MIN_QUALITY_SCORE: float = 0.85
    ALLOW_LOW_QUALITY_INDEX_IN_NON_PROD: bool = True

    class Config:
        env_file = ".env"


# Falla al arrancar si falta cualquier variable requerida
settings = Settings()
