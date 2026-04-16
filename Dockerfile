# Multi-stage: imagen final sin devdeps
# Compatible con Azure Web App
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY . .

EXPOSE 8000

# Azure Web App inyecta WEBSITES_PORT; uvicorn lo lee via settings
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${WEBSITES_PORT:-8000} --workers 1"]
