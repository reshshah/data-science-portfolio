# Build:  docker build -t cip-serving .
# Run:    docker run -p 8000:8000 cip-serving
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer between code changes
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY serving/ ./serving/
COPY registry/ ./registry/

# Train a demo model at build time if the registry shipped empty.
RUN python -c "import pathlib,sys; sys.exit(0 if any(pathlib.Path('registry').glob('model_v*')) else 1)" \
    || python -m serving.make_demo_model

EXPOSE 8000

# Container-level health check — the same endpoint k8s would probe
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
