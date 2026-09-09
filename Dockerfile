FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kuala_Lumpur

WORKDIR /app/tds

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    ffmpeg \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/tds/requirements.txt

# This service runs on a CPU-only host (actual GPU inference happens on the
# separate RunPod tds_runner service) - installing torch/torchvision from the
# default PyPI index pulls in several GB of unused NVIDIA CUDA runtime
# libraries per image, across 6 worker containers built from this same
# Dockerfile. Installing the CPU-only build first means requirements.txt's
# other torch-dependent packages (ultralytics, transformers, torchreid, etc.)
# reuse it instead of each pulling their own CUDA-enabled copy.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/tds/requirements.txt

COPY . /app/tds

EXPOSE 8010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
