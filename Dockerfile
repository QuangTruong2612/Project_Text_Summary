# Sử dụng base image slim
FROM python:3.12.3-slim
RUN apt-get update && apt-get install -y \
    awscli \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]