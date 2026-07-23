FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3-pip git \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.8 -m pip install --upgrade pip --no-cache-dir && \
    python3.8 -m pip install --require-hashes --no-cache-dir -r requirements.txt

COPY . .
CMD ["bash"]
