FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs .cache rag_storage

# Expose port for GUI
EXPOSE 7860

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Default command (can be overridden)
CMD ["python", "rag_gui.py"]
