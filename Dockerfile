# Dockerfile — MSML/MSAI 605 Face Verification
#
# CPU-only image for portability
#
# Build:
#   docker build -t msml605-verifier .
#
# Run tests:
#   docker run --rm msml605-verifier python -m pytest tests/test_milestone3.py -v
#
# Run load test:
#   docker run --rm msml605-verifier python scripts/load_test.py --config configs/milestone3.yaml --workers 2 --n-pairs 50
#
# Run CLI (with local files mounted):
#   docker run --rm -v "%cd%":/app/data msml605-verifier python scripts/verify.py --config configs/milestone3.yaml --image-a data/face1.jpg --image-b data/face2.jpg

FROM python:3.11-slim

# System dependencies needed by keras-facenet and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Pre-download the keras-facenet model weights so the container works offline.
RUN python -c "\
from keras_facenet import FaceNet; \
print('Downloading FaceNet (VGGFace2) weights ...'); \
FaceNet(); \
print('Weights cached successfully.')"

# Default command — show help
CMD ["python", "scripts/verify.py", "--help"]
