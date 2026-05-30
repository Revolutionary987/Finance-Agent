# Use a lightweight stable Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies needed for PDF processing and greenlet/psycopg compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Download the spaCy model directly into the container image
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application files
COPY . .

# Hugging Face Spaces runs on port 7860 by default
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]