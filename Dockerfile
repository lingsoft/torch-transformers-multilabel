# Simple development container
# Usage:
# docker build -t reglab-dev .
# docker run -it --rm -v $(pwd):/app reglab-dev bash
# Note. Tokenizer is downloaded from the web (requires root user).

FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
