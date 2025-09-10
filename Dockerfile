# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.11-slim AS build

WORKDIR /work

# Create virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Build wheel without dependencies
RUN pip wheel --no-deps -w dist .
