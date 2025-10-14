# Use an official PyTorch image as the base image
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies first (these rarely change)
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files (these change less frequently than source code)
COPY pyproject.toml poetry.lock ./
COPY wheels/ ./wheels/

# Install Poetry and Python dependencies
# This layer will be cached unless pyproject.toml or poetry.lock change
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    # Install the wheel files manually
    pip install wheels/*.whl

# Copy source code last (these change most frequently)
# Changes here won't invalidate the dependency installation cache
COPY trellis/ ./trellis/
COPY extensions/ ./extensions/
COPY assets/ ./assets/
COPY app.py ./

# Expose the port Streamlit will use
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
