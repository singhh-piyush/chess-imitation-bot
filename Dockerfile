# Stage 1: Build React Frontend
FROM node:18-alpine as build
WORKDIR /app/frontend
COPY chess-frontend/package*.json ./
RUN npm install
COPY chess-frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Runtime
FROM python:3.9-slim

# Install system dependencies (Stockfish)
RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend Code
COPY main.py .

COPY piyush_clone.pth .
# Copy Built Frontend from Stage 1
COPY --from=build /app/frontend/dist ./chess-frontend/dist

# Expose Hugging Face Port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
