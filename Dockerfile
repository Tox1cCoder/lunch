# Use official Python slim image for smaller footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Ho_Chi_Minh

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY bot.py .
COPY config.py .
COPY nlp_parser.py .
COPY sheets_manager.py .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash botuser
USER botuser

# Run the bot
CMD ["python", "bot.py"]
