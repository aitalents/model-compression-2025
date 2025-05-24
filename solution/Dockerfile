# ---------- build (скачиваем модели) ----------
FROM huggingface/transformers-pytorch-gpu:4.35.0 AS builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir gdown

WORKDIR /build
RUN gdown --id 1u4hyVg2ZHXc93Ka9-GcDjqXrRfDGGQBT --output models.zip && \
    unzip models.zip && rm models.zip

# ---------- runtime ----------
FROM huggingface/transformers-pytorch-gpu:4.35.0
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /src

RUN apt-get update && \
    apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# копируем model-директории из builder-слоя
COPY --from=builder /build /models
ENV PYTHONPATH="${PYTHONPATH}:/src" \
    TRANSFORMERS_CACHE="/src/.cache/" \
    MODELS_DIR="/models"

COPY . /src

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
