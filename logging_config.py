"""
Configuración de logging estructurado con structlog.
LOG_FORMAT=json en producción, LOG_FORMAT=console en desarrollo.
Nunca loguear: connection strings, API keys, contenido de documentos.
"""
import structlog

from settings import settings


def configure_structlog() -> None:
    """Configura structlog según LOG_FORMAT (json o console)."""
    if settings.LOG_FORMAT == "json":
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
        )
