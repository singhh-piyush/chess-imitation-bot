# Chess Bot v2: Neural Chess Clone

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Stack](https://img.shields.io/badge/stack-React%20%7C%20FastAPI%20%7C%20PyTorch-blue)
![Engine](https://img.shields.io/badge/engine-Stockfish%2017-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> **A chess bot that copies how a real person plays.**

---

## Project Overview

Most chess engines try to find the strongest move every time.

This project does something different.

It learns how **one specific person** plays chess by studying their past games. Instead of asking “what is the best move?”, it asks:

> “What move would this person most likely play here?”

The result is a chess bot that feels human rather than perfect.

---

## Live Demo

* **Running application (Hugging Face):**
  [https://huggingface.co/spaces/singhh-piyush/chess-bot-backend](https://huggingface.co/spaces/singhh-piyush/chess-bot-backend)

---

## How the Engine Works

The engine makes decisions in three clear steps.

---

### 1. Neural Network (Move Prediction)

* A custom neural network looks at the board position
* It predicts the **top 5 moves** the player would usually make
* These predictions come from patterns learned from past games

The network does not understand chess rules or strategy. It only learns from examples.

---

### 2. Stockfish Check (Blunder Protection)

* Each predicted move is checked by the Stockfish engine
* If a move makes the position worse by more than **150 centipawns (about 1.5 pawns)**, it is rejected

This prevents:

* Losing a queen for free
* Missing obvious captures
* One-move blunders

---

### 3. Fallback Move

* If all predicted human moves are unsafe
* Stockfish selects a normal engine move instead

This keeps the game playable even in bad positions.

---

## Transparency

Unlike most chess bots, this project shows its decision process:

* Which moves the neural network suggested
* How Stockfish evaluated each move
* Which moves were blocked and why

Nothing is hidden.

---

## Training Data

The model was trained on:

* **3,400+ personal games**
* **Around 170,000 board positions**

This allows the model to learn:

* Opening preferences
* Common habits
* Typical mistakes

---

## Technology Stack

| Part         | Technology                 | Purpose             |
| ------------ | -------------------------- | ------------------- |
| Frontend     | React, Vite, TailwindCSS   | User interface      |
| Backend      | Python, FastAPI            | API and game logic  |
| Neural Model | PyTorch                    | Move prediction     |
| Validation   | Stockfish 17, python-chess | Blunder checking    |
| Hosting      | Linux (Debian)             | Runtime environment |

---

## Quick Start (Local Setup)

> **Recommended system:** Linux (Arch, Ubuntu, Debian)

Windows is not officially supported. Use WSL if needed.

---

### 1. Clone the Repository

```bash
git clone https://github.com/singhh-piyush/chess-imitation-bot.git
cd chess-imitation-bot
```

---

### 2. Backend Setup

#### Install Stockfish

Arch Linux:

```bash
sudo pacman -S stockfish
```

Ubuntu / Debian:

```bash
sudo apt-get install stockfish
```

#### Install Python Dependencies and Run

```bash
cd chess-bot-backend
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at:

```
http://localhost:8000
```

---

### 3. Frontend Setup

```bash
cd chess-frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

### 4. Model File

The trained model file is not included because it is large.

You can:

* Train your own model using `train_model.py`
* Download the pre-trained model from Hugging Face

Place the file here:

```
chess-bot-backend/piyush_clone.pth
```

---

### 5. Verify Everything Works

* Open `http://localhost:8000/` → should return a JSON response
* `POST /predict` should return a chess move
* Frontend should connect to the backend
* Stockfish must be found by the system

If something fails, check the `STOCKFISH_PATH` in `main.py` first.

---

## Contributing

You are welcome to fork this project.

To adapt it:

* Train the model on your own games
* Replace the `.pth` file
* Update the backend if the model changes

Submit a pull request with clear notes.

---

## License

This project is released under the **MIT License**.
