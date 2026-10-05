# DeepfakeGuard API - container recipe
# Build:  docker build -t deepfakeguard-api .
# Run:    docker run --rm -p 8000:8000 deepfakeguard-api
# Then open http://127.0.0.1:8000/docs

# 1. Start from a small official Linux image that already has Python 3.12
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 2. Install libraries first (Docker caches this step, so code changes rebuild fast)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 3. Copy only what the API needs: code + the V2 model (never .env or the dataset)
COPY api/ api/
COPY src/ src/
COPY models/image_classifier_v2/ models/image_classifier_v2/

# 4. Run as a normal user, not the all-powerful "root" user
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# 5. Let Docker check the API is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 6. Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
