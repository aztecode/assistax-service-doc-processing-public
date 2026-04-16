"""
Job de cleanup: marca como failed los index_runs en processing por más de 1 hora.
APScheduler embebido, suficiente para una sola instancia.
"""
from pipeline.db_writer import get_db_conn
from settings import settings

_scheduler = None


def _get_scheduler():
    """Importación lazy para evitar dependencia circular."""
    from apscheduler.schedulers.background import BackgroundScheduler
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
    return _scheduler


def cleanup_stuck_runs() -> None:
    """
    Marca como failed los index_runs en processing por más de 1 hora.
    Silencioso si la tabla index_runs no existe (BD recién creada).
    """
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE index_runs
                    SET status = 'failed',
                        "endedAt" = NOW(),
                        error = 'Timeout: pipeline superó 1 hora sin completar'
                    WHERE status = 'processing'
                      AND "startedAt" < NOW() - INTERVAL '1 hour'
                    RETURNING id
                """)
                fixed = cursor.rowcount
                conn.commit()
        if fixed > 0:
            import structlog
            structlog.get_logger().warning("cleanup.stuck_runs_fixed", count=fixed)
    except Exception as e:
        # Tabla inexistente o schema no migrado: no es crítico en Fase 0
        import structlog
        structlog.get_logger().debug("cleanup.stuck_runs_skipped", reason=str(e))


def start_cleanup_job() -> None:
    """Inicia el job de cleanup periódico."""
    scheduler = _get_scheduler()
    scheduler.add_job(
        cleanup_stuck_runs,
        "interval",
        minutes=settings.CLEANUP_INTERVAL_MINUTES,
    )
    if not scheduler.running:
        scheduler.start()


def stop_cleanup_job() -> None:
    """Detiene el scheduler de cleanup."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
