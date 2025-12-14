# ♟️ The Imitation Engine: Neural Chess Clone (v2.0)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Tech Stack](https://img.shields.io/badge/stack-React%20%7C%20FastAPI%20%7C%20PyTorch-blue)
![Model](https://img.shields.io/badge/model-CNN%20%2B%20Stockfish%2016-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> **"I trained a neural network to play exactly like me—blunders included."**

A hybrid chess engine that combines deep learning (behavioral cloning) with algorithmic safety (Stockfish). Unlike standard engines that calculate the *best* move, this bot predicts the *most human* move based on my personal games, while using a real-time "Safeguard Protocol" to prevent obvious throws.

![Gameplay Hero](screenshots/chessbot3.jpg)

## Live Status
* **Backend API (Hugging Face):** [🟢 Online](https://huggingface.co/spaces/singhh-piyush/chess-bot-backend)
* **Frontend:** *(Add your Vercel Link Here)*

---

## The Interface

### 1. The Decision Matrix
Most bots hide their logic. We visualize it.
The **Decision Matrix** (left panel) shows the internal conflict between the **Neural Network** (my instincts) and the **Stockfish Safeguard** (the logic). You can see exactly which "human" moves were considered and which were vetoed for being blunders.

### 2. Training Data & Scale
The model was trained on a curated dataset of over **3,400 games** and **170,000 positions**, extracting my specific opening repertoire and mid-game tendencies.

![Stats Splash Screen](screenshots/chessbot1.png)

### 3. Configurable "Humanity"
You can adjust the **Intervention Budget** in the settings.
* **Strict Mode:** Stockfish intervenes often (High Elo, less human).
* **Loose Mode:** Stockfish stays silent, letting the Neural Net make "human" mistakes.

![Settings Panel](screenshots/chessbot2.png)

---

## Architecture: The "Hybrid" Brain
This project uses a **Two-Stage Decision Process** to balance intuition with safety.

```mermaid
graph TD
    User((User)) -->|FEN String| Frontend[React Frontend]
    Frontend -->|POST /predict| API[FastAPI Backend]
    
    subgraph "Backend Logic (Hugging Face Docker)"
        API -->|Input| Brain[PyTorch CNN Model]
        API -->|Input| Safety[Stockfish 16 Engine]
        
        Brain -->|Generates Candidates| Decision{Decision Matrix}
        
        Decision -->|Verify Move 1| Safety
        Decision -->|Verify Move 2| Safety
        
        Safety -->|Blunder?| Veto[❌ VETO]
        Safety -->|Safe?| Approved[✅ PLAY]
        
        Veto -->|Next Candidate| Decision
        Approved -->|Final Move| Response
    end
    
    Response -->|JSON| Frontend
