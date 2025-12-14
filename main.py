import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import chess
import chess.engine
import os
import contextlib
import json
import time
import shutil

# --- Configuration ---
MODEL_PATH = "piyush_clone.pth"
# Check system path first (Docker), then local fallback
STOCKFISH_PATH = shutil.which("stockfish") or "stockfish/stockfish-ubuntu-x86-64-avx2"
STOCKFISH_DEPTH = 15     # Standard
PANIC_DEPTH = 22         # Deep check
SAFETY_THRESHOLD_CP = 150 # Veto threshold
SUSPICIOUS_THRESHOLD_CP = 50 # Trigger deep check

# --- Model Architecture (Must match training) ---
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        # Input: 12 x 8 x 8
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Flatten: 128 * 8 * 8 = 8192
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(8192, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 4096) # Output: 0-4095 classes

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Helpers ---
def board_to_tensor(board):
    """Convert board to 12x8x8 boolean array (float32 for model)."""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                layer += 6
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[layer, rank, file] = 1.0
    return tensor

def decode_move(move_idx):
    """Convert integer index back to UCI move."""
    from_sq = move_idx // 64
    to_sq = move_idx % 64
    return chess.Move(from_sq, to_sq)

def is_move_safe(board, move, current_eval):
    """
    Evaluates if a move is safe using 2-pass verification.
    Returns True if safe, False if vetoed.
    """
    if not engine:
        return True

    # Perspective check
    is_white_turn = board.turn == chess.WHITE
    
    # --- Pass 1: Standard Check (Depth 15) ---
    board.push(move)
    try:
        info_1 = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
        score_after_1 = info_1["score"].white().score(mate_score=10000)
    except:
        board.pop()
        return True # Assume safe on error
    board.pop()
    
    # Calculate Drop
    if is_white_turn:
        drop = current_eval - score_after_1
    else:
        drop = score_after_1 - current_eval
        
    # Check Thresholds
    if drop > SAFETY_THRESHOLD_CP:
        return False # Blunder in Pass 1
        
    if drop < SUSPICIOUS_THRESHOLD_CP:
        return True # Safe in Pass 1
        
    # --- Pass 2: Panic Check (Depth 22) ---
    # Drop is between 50 and 150 -> Suspicious
    board.push(move)
    try:
        info_2 = engine.analyse(board, chess.engine.Limit(depth=PANIC_DEPTH))
        score_after_2 = info_2["score"].white().score(mate_score=10000)
    except:
        board.pop()
        return True
    board.pop()
    
    if is_white_turn:
        drop_2 = current_eval - score_after_2
    else:
        drop_2 = score_after_2 - current_eval
        
    if drop_2 > SAFETY_THRESHOLD_CP:
        return False # Blunder confirmed in Pass 2
        
    return True

# --- Global State ---
model = None
engine = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, engine
    print("Initializing Backend...")
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = ChessModel()
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Warning: Model file {MODEL_PATH} not found.")
        model = None

    # Init Stockfish
    if os.path.exists(STOCKFISH_PATH):
        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            print("Stockfish engine started.")
        except Exception as e:
            print(f"Error starting Stockfish: {e}")
            engine = None
    else:
        print(f"Warning: Stockfish not found at {STOCKFISH_PATH}")
        engine = None
        
    yield
    
    # Shutdown
    if engine:
        engine.quit()
        print("Stockfish engine stopped.")

# --- App ---
app = FastAPI(lifespan=lifespan)

# CORS
origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FenRequest(BaseModel):
    fen: str

def get_stockfish_score(board):
    """Get White-centric score of the current position."""
    if not engine:
        return 0
    try:
        # Upgraded to depth=STOCKFISH_DEPTH (15)
        info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
        score = info["score"].white().score(mate_score=10000)
        return score
    except:
        return 0

@app.post("/predict")
async def predict(request: FenRequest):
    global model, engine
    
    thinking_log = []
    start_time = time.time()
    
    board = chess.Board(request.fen)
    is_white_turn = board.turn == chess.WHITE
    
    candidates = []
    
    thinking_log.append("Neural Network: Scanning position...")
    
    # 1. Neural Network Prediction
    if model:
        try:
            tensor = board_to_tensor(board)
            tensor = torch.from_numpy(tensor).unsqueeze(0) # Batch size 1
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                
            # Get Top 20
            top_probs, top_indices = torch.topk(probs, 20)
            top_probs = top_probs[0].numpy()
            top_indices = top_indices[0].numpy()
            
            for i in range(len(top_indices)):
                move = decode_move(top_indices[i])
                
                # Check Legality
                if move in board.legal_moves:
                    if chess.square_rank(move.to_square) in [0, 7] and board.piece_at(move.from_square).piece_type == chess.PAWN:
                         move.promotion = chess.QUEEN
                         
                    if move in board.legal_moves:
                        san_move = board.san(move)
                        candidates.append({
                            "move": move.uci(),
                            "san": san_move,
                            "confidence": float(top_probs[i]),
                            "object": move
                        })
            
            # Initialize status
            for c in candidates:
                c['status'] = 'IGNORED'

            thinking_log.append(f"Neural Network: Found {len(candidates)} legal candidate moves.")
                        
        except Exception as e:
            thinking_log.append(f"Neural Network Error: {e}")
            candidates = []

    # 2. Safety Check (Stockfish)
    safe_move = None
    fallback = False
    
    if candidates and engine:
        thinking_log.append(f"Safety Check: Verifying top moves (Adaptive Depth)...")
        
        # Calculate Current Eval first
        try:
             info_curr = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
             current_eval = info_curr["score"].white().score(mate_score=10000)
        except:
             current_eval = 0

        valid_candidates = []
        
        # Check up to 5 candidates
        for cand in candidates[:5]:
            move_obj = cand['object']
            
            # Use new safeguard logic
            is_safe = is_move_safe(board, move_obj, current_eval)
            
            if is_safe:
                cand['status'] = 'ANALYZED'
                valid_candidates.append(cand)
                # If we found a safe move that is high confidence, we can stop or keep looking?
                # Usually we want the highest confidence safe move.
            else:
                cand['status'] = 'VETOED'
                # thinking_log.append(f"Move {cand['move']}: VETOED (Blunder detected)")

        if valid_candidates:
             safe_move = valid_candidates[0]
             safe_move['status'] = 'SELECTED'
             thinking_log.append(f"Selected Best Safe Move: {safe_move['move']}")
        else:
            thinking_log.append("All Neural candidates rejected by Safety Check.")
            fallback = True
            
    elif candidates and not engine:
        thinking_log.append("Stockfish engine not available. Using purely Neural Network move.")
        safe_move = candidates[0]
        safe_move['status'] = 'SELECTED'
        
    else:
        fallback = True

    # 3. Fallback
    if fallback or not safe_move:
        if engine:
            try:
                thinking_log.append(f"Fallback: Calculating best move with Stockfish (Depth {STOCKFISH_DEPTH})...")
                # Upgraded fallback to depth-based
                result = engine.play(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
                best_move = result.move
                safe_move = {
                    "move": best_move.uci(),
                    "confidence": 1.0,
                    "is_fallback": True,
                    "status": "SELECTED" 
                }
                thinking_log.append(f"Fallback Move Found: {best_move.uci()}")
                
                # Auto-Learning Log
                try:
                    with open("hard_positions.jsonl", "a") as f:
                        f.write(json.dumps({"fen": request.fen}) + "\n")
                except:
                    pass
                    
            except Exception as e:
                thinking_log.append(f"Fallback Error: {e}")
                # Emergency random
                import random
                m = random.choice(list(board.legal_moves))
                safe_move = { "move": m.uci(), "confidence": 0.0, "is_fallback": True, "status": "SELECTED" }
        else:
            # Random fallback
            import random
            moves = list(board.legal_moves)
            if moves:
                m = random.choice(moves)
                safe_move = { "move": m.uci(), "confidence": 0.0, "is_fallback": True, "status": "SELECTED" }
            else:
                return {"move": None, "game_over": True}

    elapsed = time.time() - start_time
    thinking_log.append(f"Thinking Complete in {elapsed:.2f}s")
    
    # Enrich candidates return
    final_candidates = []
    for c in candidates[:5]:
        final_candidates.append({
            "move": c["move"], 
            "san": c["san"], 
            "confidence": c["confidence"],
            "status": c.get("status", "IGNORED")
        })

    return {
        "move": safe_move["move"],
        "confidence": safe_move["confidence"],
        "is_fallback": safe_move.get("is_fallback", False),
        "candidates": final_candidates,
        "thinking_log": thinking_log 
    }

class MistakeRequest(BaseModel):
    fen: str

@app.post("/report_mistake")
async def report_mistake(request: MistakeRequest):
    mistakes_file = "mistakes.json"
    mistakes = []
    
    if os.path.exists(mistakes_file):
        try:
            with open(mistakes_file, "r") as f:
                mistakes = json.load(f)
        except:
            mistakes = []
            
    if request.fen not in mistakes:
        mistakes.append(request.fen)
        with open(mistakes_file, "w") as f:
            json.dump(mistakes, f)
            
    return {"status": "success", "message": "Mistake reported"}


# --- SERVE FRONTEND (Production/Docker) ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve React App if built
if os.path.exists("chess-frontend/dist"):
    app.mount("/assets", StaticFiles(directory="chess-frontend/dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Allow API calls to pass through
        if full_path.startswith("predict") or full_path.startswith("report_mistake"):
             return {"error": "Not Found"} # API routes should be handled by their handlers
        
        # Serve index.html for any other route (SPA)
        file_path = f"chess-frontend/dist/{full_path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
             return FileResponse(file_path)
             
        return FileResponse("chess-frontend/dist/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
