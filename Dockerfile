# Dockerfile
# Builds the container for the Gradio mental health diagnosis app

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for faiss and torch
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ config/
COPY internal/ internal/
COPY cmd/gradio/ cmd/gradio/
COPY data/ data/

# Expose Gradio default port
EXPOSE 7860

CMD ["python", "cmd/gradio/main.py"]