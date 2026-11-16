# ShopSense AI - FastAPI backend (prediction + SHAP explanations + Website Intelligence)
#
# Model artifacts are NOT in git. Either put them in backend/models/ before building, or pass
# MODEL_ARTIFACTS_URL (e.g. a GitHub Release download URL) and they are downloaded at build time.
# Each file is accepted only if its SHA-256 matches the committed model_manifest.json, and the
# build fails if any artifact is missing or different.
#
#   docker build -t shopsense-api --build-arg MODEL_ARTIFACTS_URL=https://github.com/<user>/shopsense-ai/releases/download/models-v1 ./backend
#   docker run -p 8000:8000 -e CORS_ORIGINS='["https://your-frontend.example"]' shopsense-api

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    # Proxies whose X-Forwarded-For is believed (private ranges used by platform load balancers).
    # Never "*": Uvicorn would then trust the client-supplied entry and rate limits could be bypassed.
    FORWARDED_ALLOW_IPS="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10,127.0.0.1,::1"

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY app ./app
COPY ml ./ml
COPY reports ./reports
COPY models ./models

# Optional verified download, then the integrity gate: refuse to build an image with
# missing, unexpected or tampered model pickles.
ARG MODEL_ARTIFACTS_URL=""
RUN if [ -n "$MODEL_ARTIFACTS_URL" ]; then python -m ml.fetch_artifacts "$MODEL_ARTIFACTS_URL"; fi \
    && python -m ml.integrity

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health\", timeout=4)"

# One worker per container: each worker holds its own copy of the model (~250 MB).
# PORT is set by platforms such as Render; 8000 otherwise.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers"]
