# ShopSense AI - FastAPI backend (prediction + SHAP explanations)
#
# Model artifacts are NOT in git. Put them in backend/models/ before building
# (train them with the ml pipeline, or download them from your artifact store).
# The build fails if any artifact does not match the SHA-256 in model_manifest.json.
#
#   docker build -t shopsense-api ./backend
#   docker run -p 8000:8000 -e CORS_ORIGINS='["https://your-frontend.example"]' shopsense-api

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY app ./app
COPY ml ./ml
COPY reports ./reports
COPY models ./models

# Integrity gate: refuse to build an image with unexpected/tampered model pickles.
RUN python -m ml.integrity

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker per container: each worker holds its own copy of the model (~250 MB).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
