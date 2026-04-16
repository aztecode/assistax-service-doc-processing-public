# models.py — Contrato tipado para ProcessPdfRequest
from typing import Optional

from pydantic import BaseModel, field_validator
import uuid


class ProcessPdfRequest(BaseModel):
    runId: str
    blobPath: str
    documentTitle: str
    categoryId: str
    publishDate: Optional[str] = None  # Siempre presente en el body, puede ser null
    # Omitido o null → usar settings.RELAX_PROSE_TABLE_FILTER / RELAXED_VISUAL_FRAME_DETECTION
    relaxProseTableFilter: Optional[bool] = None
    relaxedVisualFrameDetection: Optional[bool] = None
    # Omitido o null → usar settings.ENABLE_LLM_GENERIC_HEADING_REFINE
    enableLlmGenericHeadingRefine: Optional[bool] = None

    @field_validator("runId", "categoryId")
    @classmethod
    def must_be_valid_uuid(cls, v: str):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError(f"'{v}' no es un UUID válido")
        return v

    @field_validator("blobPath", "documentTitle")
    @classmethod
    def must_not_be_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("El campo no puede estar vacío")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "runId": "550e8400-e29b-41d4-a716-446655440000",
                "blobPath": "fiscal/2024/ley-federal-trabajo.pdf",
                "documentTitle": "Ley Federal del Trabajo",
                "categoryId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "publishDate": "2024-01-15",
                "relaxProseTableFilter": None,
                "relaxedVisualFrameDetection": None,
                "enableLlmGenericHeadingRefine": None,
            }
        }
    }


class ProcessPdfResponse(BaseModel):
    runId: str
    status: str = "processing"


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str]
    version: str
    environment: str
