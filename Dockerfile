# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and other files
COPY src/ ./src/
COPY main.py .
COPY README.md .
COPY LICENSE .

# Create directory for documents
RUN mkdir -p data/documents samples

# Copy sample data
COPY samples/ ./samples/

# Set command
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
