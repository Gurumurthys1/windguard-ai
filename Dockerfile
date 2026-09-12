FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow / OpenCV-adjacent image libs
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libglib2.0-0 libsm6 libxext6 libxrender1 \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/rtdetr-wtbd-final/ ./models/rtdetr-wtbd-final/

ENV WINDGUARD_MODEL_DIR=models/rtdetr-wtbd-final
ENV WINDGUARD_SCORE_THRESHOLD=0.5

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
