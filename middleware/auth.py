# middleware/auth.py — Validación x-functions-key (obligatoria)
from fastapi import Request
from fastapi.responses import JSONResponse

from settings import settings


async def verify_api_key(request: Request, call_next):
    # /health es público para Azure Web App health probes
    if request.url.path == "/health":
        return await call_next(request)

    key = request.headers.get("x-functions-key")
    if not key or key != settings.FUNCTIONS_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)
