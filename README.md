---
title: My Chess Bot
emoji: ♟️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# My Chess Bot

A Neural Network powered chess bot with Stockfish safeguard, capable of mimicking human play styles and avoiding blunders.

## Deployment

This space receives the Docker build which:
1. Builds the React Frontend (Vite)
2. Starts the Python FastAPI Backend
3. Serves the Frontend from the Backend

## Local Development

1. **Frontend**: `cd chess-frontend && npm install && npm run dev`
2. **Backend**: `python main.py`

## Features

- **Neural Network**: Trained on 3,000+ games.
- **Safeguard**: 2-Pass Stockfish verification (Depth 15/22).
- **Interactive UI**: React-based board with real-time analysis logs.
- **Adaptive**: Fallback to Stockfish if NN is unsure.
