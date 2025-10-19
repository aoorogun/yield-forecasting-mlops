FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YF_MODEL_DIR=/opt/app/artifacts

RUN useradd --create-home --uid 10001 appuser
WORKDIR /opt/app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY configs ./configs
COPY artifacts ./artifacts
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health/live')"
CMD ["uvicorn", "yield_forecasting.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

