import requests
import chess
import chess.pgn
import chess.engine
import numpy as np # type: ignore
import json
import os
import io
import time
import multiprocessing
from tqdm import tqdm # type: ignore

# --- Configuration ---
USERNAME = 'piyushhsingh'
STOCKFISH_PATH = '/usr/bin/stockfish'
CP_LOSS_THRESHOLD = 150
STOCKFISH_TIME_LIMIT = 0.05 # Reduced for speed
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_FILE = os.path.join(OUTPUT_DIR, 'inputs.npz')
TARGETS_FILE = os.path.join(OUTPUT_DIR, 'targets.npz')
PROCESSED_LOG = os.path.join(OUTPUT_DIR, 'processed_urls.json')

# Global variable for worker process
engine = None

def worker_init():
    """Initialize the Stockfish engine for the worker process."""
    global engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Error initializing engine in worker: {e}")

def get_archives(username):
    """Fetch list of monthly archive URLs from Chess.com API."""
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    headers = {'User-Agent': f'ChessDataMiner/1.0 (username: {username})'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get('archives', [])
    except requests.RequestException as e:
        print(f"Error fetching archives: {e}")
        return []

def process_archive_download(url):
    """Download PGN data for a specific monthly archive."""
    if not url.endswith('/pgn'):
        url += '/pgn'
        
    headers = {'User-Agent': f'ChessDataMiner/1.0 (username: {USERNAME})'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching archive {url}: {e}")
        return None

def board_to_tensor(board):
    """Convert board to 12x8x8 boolean array."""
    tensor = np.zeros((12, 8, 8), dtype=bool)
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
            tensor[layer, rank, file] = True
    return tensor

def encode_move(move):
    """Convert move to integer index: from_sq * 64 + to_sq."""
    return move.from_square * 64 + move.to_square

def mirror_board_and_move(board_tensor, move_from, move_to):
    """Mirror the board tensor and move horizontally."""
    new_tensor = np.flip(board_tensor, axis=2)
    def flip_h(sq):
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        return chess.square(7 - f, r)
    new_from = flip_h(move_from)
    new_to = flip_h(move_to)
    return new_tensor, encode_move(chess.Move(new_from, new_to))

def process_single_game(pgn_text):
    """
    Worker function to process a single game PGN.
    Returns a list of (tensor, move_idx) tuples.
    """
    global engine
    if engine is None:
        return []
    
    results = []
    pgn_io = io.StringIO(pgn_text)
    
    try:
        game = chess.pgn.read_game(pgn_io)
    except Exception:
        return results
        
    if game is None:
        return results

    if game.headers.get("Variant", "Standard") != "Standard":
        return results

    board = game.board()
    
    white = game.headers.get("White", "").lower()
    username_lower = USERNAME.lower()
    if white == username_lower:
        our_color = chess.WHITE
    elif game.headers.get("Black", "").lower() == username_lower:
        our_color = chess.BLACK
    else:
        return results # Not our game

    node = game
    while not node.is_end():
        next_node = node.next()
        if not next_node:
            break
            
        move = next_node.move
        
        if board.turn == our_color:
            try:
                # Pre-move analysis
                info_pre = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
                score_pre = info_pre["score"].white().score(mate_score=10000)
                
                if score_pre is not None:
                     board.push(move)
                     info_post = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
                     score_post = info_post["score"].white().score(mate_score=10000)
                     board.pop()
                     
                     if score_post is not None:
                        loss = (score_pre - score_post) if our_color == chess.WHITE else (score_post - score_pre)
                        
                        if loss <= CP_LOSS_THRESHOLD:
                            tensor = board_to_tensor(board)
                            move_idx = encode_move(move)
                            results.append((tensor, move_idx))
                            
                            aug_tensor, aug_move_idx = mirror_board_and_move(tensor, move.from_square, move.to_square)
                            results.append((aug_tensor, aug_move_idx))
            except Exception:
                pass # Skip problematic positions

        board.push(move)
        node = next_node
        
    return results

def split_pgn_text(pgn_full_text):
    """
    Splits a large string of multiple PGN games into a list of single PGN strings.
    This is a heuristic split on '[Event "'.
    """
    # Simply using a regex or split might be fragile if [Event " appears in comments.
    # But for Chess.com PGN downloads, it's usually clean.
    # A safer way might be to iterate line by line, but splitting on '\n[Event "' is reasonably safe.
    # We add the newline back.
    
    games = []
    current_game = []
    
    for line in pgn_full_text.splitlines():
        if line.startswith('[Event "'):
            if current_game:
                games.append("\n".join(current_game))
                current_game = []
        current_game.append(line)
        
    if current_game:
        games.append("\n".join(current_game))
        
    return games

def main():
    # 0. Load State
    if os.path.exists(INPUTS_FILE) and os.path.exists(TARGETS_FILE):
        print("Loading existing data...")
        try:
            with np.load(INPUTS_FILE) as data:
                all_inputs = list(data['arr_0'])
            with np.load(TARGETS_FILE) as data:
                all_targets = list(data['arr_0'])
        except Exception as e:
            print(f"Error loading existing npz: {e}, starting fresh.")
            all_inputs = []
            all_targets = []
    else:
        all_inputs = []
        all_targets = []

    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r') as f:
            processed_urls = set(json.load(f))
    else:
        processed_urls = set()
        
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Stockfish not found at {STOCKFISH_PATH}. Please check path.")
        return

    # 1. Get Archives
    archives = get_archives(USERNAME)
    print(f"Found {len(archives)} monthly archives.")
    
    # Use 90% of cores or max-1
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting stats mining with {num_processes} worker processes...")

    try:
        with multiprocessing.Pool(processes=num_processes, initializer=worker_init) as pool:
            
            for url in tqdm(archives, desc="Archives"):
                if url in processed_urls:
                    continue

                pgn_text = process_archive_download(url)
                if not pgn_text:
                    continue
                
                # Split games
                game_texts = split_pgn_text(pgn_text)
                if not game_texts:
                    continue
                    
                # Process in parallel
                # chunksize=1 is fine, or bigger for fewer IPC calls
                results_nested = list(tqdm(pool.imap(process_single_game, game_texts, chunksize=5), 
                                           total=len(game_texts), 
                                           desc="Processing Games", 
                                           leave=False))
                
                # Flatten results
                for game_results in results_nested:
                    for tensor, move_idx in game_results:
                        all_inputs.append(tensor)
                        all_targets.append(move_idx)
                
                # Incremental Save
                processed_urls.add(url)
                
                temp_inputs = os.path.join(OUTPUT_DIR, 'inputs_temp.npz')
                temp_targets = os.path.join(OUTPUT_DIR, 'targets_temp.npz')
                
                np.savez_compressed(temp_inputs, np.array(all_inputs, dtype=bool))
                np.savez_compressed(temp_targets, np.array(all_targets, dtype=np.int16))
                
                os.replace(temp_inputs, INPUTS_FILE)
                os.replace(temp_targets, TARGETS_FILE)
                
                with open(PROCESSED_LOG, 'w') as f:
                    json.dump(list(processed_urls), f)
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    finally:
        print(f"Final dataset size: {len(all_inputs)} samples.")
        if len(all_inputs) > 0:
            np.savez_compressed(INPUTS_FILE, np.array(all_inputs, dtype=bool))
            np.savez_compressed(TARGETS_FILE, np.array(all_targets, dtype=np.int16))
            with open(PROCESSED_LOG, 'w') as f:
                json.dump(list(processed_urls), f)
            print(f"Saved to {INPUTS_FILE} and {TARGETS_FILE}")

if __name__ == "__main__":
    main()
