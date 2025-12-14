import { useState, useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import axios from 'axios';
import SplashScreen from './SplashScreen';

// --- HELPERS ---

const getPieceImg = (type, color) => {
  const names = {
    p: 'pawn',
    n: 'knight',
    b: 'bishop',
    r: 'rook',
    q: 'queen',
    k: 'king'
  };
  return `/TakenPiecesSVG/${names[type]}-${color}.svg`;
};

const getCapturedPieces = (game) => {
  const board = game.board();
  const currentCounts = {
    w: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 },
    b: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 }
  };

  board.forEach(row => {
    row.forEach(square => {
      if (square) {
        currentCounts[square.color][square.type]++;
      }
    });
  });

  const captured = { w: [], b: [] };
  const STARTING_PIECES = { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 };

  ['q', 'r', 'b', 'n', 'p'].forEach(type => {
    const wMissing = STARTING_PIECES[type] - currentCounts['w'][type];
    for (let i = 0; i < wMissing; i++) captured['b'].push({ type, color: 'w' });

    const bMissing = STARTING_PIECES[type] - currentCounts['b'][type];
    for (let i = 0; i < bMissing; i++) captured['w'].push({ type, color: 'b' });
  });

  return captured;
};

// --- COMPONENTS ---

const Modal = ({ children, onClose }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm" onClick={onClose}>
    <div className="bg-slate-900 border border-slate-700 p-8 rounded-xl shadow-[0_0_50px_rgba(30,41,59,0.5)] max-w-md w-full text-center relative overflow-hidden" onClick={e => e.stopPropagation()}>
      {/* Decorative top line removed as requested */}
      {children}
    </div>
  </div>
);

const InfoModal = ({ onClose }) => {
  const [isClosing, setIsClosing] = useState(false);
  const handleClose = () => {
    setIsClosing(true);
    setTimeout(onClose, 300); // Wait for animation
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-[2px] pointer-events-auto" onClick={handleClose}>
      <div className={`h-full w-full max-w-md bg-slate-900 border-l border-slate-700 p-8 shadow-2xl relative overflow-y-auto ${isClosing ? 'animate-slide-out-right' : 'animate-slide-in-right'}`} onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6 border-b border-slate-800 pb-4">
          <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider">About the Bot</h3>
          {/* Close X */}
          <button onClick={handleClose} className="text-slate-500 hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-4 text-sm text-slate-300 leading-relaxed font-sans text-left">
          <p>
            Most chess engines (like Stockfish) are calculated to play perfect chess. <strong className="text-slate-200">My bot is different.</strong> It is an Imitation Model based on Behavioral Cloning.
          </p>
          <p>
            I trained a Convolutional Neural Network (CNN) on <strong className="text-slate-200">3,437</strong> of my own games (over 171,000 board positions). Instead of calculating the best move, it predicts the move I would play.
          </p>
          <p>
            But pure imitation models blindly copy human errors. To fix this, I created a Hybrid Inference Pipeline.
          </p>

          <div>
            <h4 className="text-blue-400 font-bold mb-2">How it works ?</h4>
            <p className="mb-2">
              The Neural Net analyzes the board and suggests a move based on my style. Before playing, a background engine (Stockfish) instantly evaluates the move.
            </p>
            <p>
              <strong className="text-slate-200">Engine Safeguard:</strong> If the suggested move is a blunder (dropping the win probability significantly), the engine vetoes it and forces the Neural Net to choose its second-best "human" option.
            </p>
          </div>

          <p>
            The Result is a bot that plays with the personality of myself but has a safety net to prevent trivial losses.
          </p>

          <div>
            <h4 className="text-blue-400 font-bold mb-2">Limitations</h4>
            <p className="mb-2">
              The Stockfish safeguard prevents blunders but doesn't suggest winning moves when the neural net falters. People often resign or play passively in lost positions, which the imitation model replicates.
            </p>
            <p>
              The safeguard is configured only for blunder-prevention and does not steer the bot towards a checkmate.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}; // Close InfoModal

const SettingsModal = ({ showGlow, setShowGlow, showAnalysis, setShowAnalysis, showHistory, setShowHistory, onClose }) => {
  const [isClosing, setIsClosing] = useState(false);
  const handleClose = () => {
    setIsClosing(true);
    setTimeout(onClose, 300);
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-[2px] pointer-events-auto" onClick={handleClose}>
      <div className={`h-full w-full max-w-md bg-slate-900 border-l border-slate-700 p-8 shadow-2xl relative overflow-y-auto ${isClosing ? 'animate-slide-out-right' : 'animate-slide-in-right'}`} onClick={e => e.stopPropagation()}>

        <div className="flex items-center justify-between mb-6 border-b border-slate-800 pb-4">
          <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider">Settings</h3>
          {/* Close X */}
          <button onClick={handleClose} className="text-slate-500 hover:text-white transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>


        {/* Visuals */}
        <div className="mb-8">
          <div className="text-[10px] text-slate-400 uppercase mb-3 font-bold tracking-widest">Visuals</div>
          <div className="flex items-center justify-between mb-4">
            <span className="text-slate-200 text-sm font-medium">Brain Glow</span>
            <button
              onClick={() => setShowGlow(!showGlow)}
              className={`w-12 h-6 rounded-full flex items-center transition-colors p-1 ${showGlow ? 'bg-blue-600 justify-end' : 'bg-slate-700 justify-start'}`}
            >
              <div className="w-4 h-4 rounded-full bg-white shadow-sm" />
            </button>
          </div>

          <div className="flex items-center justify-between mb-4">
            <span className="text-slate-200 text-sm font-medium">Show Analysis Panel</span>
            <button
              onClick={() => setShowAnalysis(!showAnalysis)}
              className={`w-12 h-6 rounded-full flex items-center transition-colors p-1 ${showAnalysis ? 'bg-blue-600 justify-end' : 'bg-slate-700 justify-start'}`}
            >
              <div className="w-4 h-4 rounded-full bg-white shadow-sm" />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-200 text-sm font-medium">Show Move History</span>
            <button
              onClick={() => setShowHistory(!showHistory)}
              className={`w-12 h-6 rounded-full flex items-center transition-colors p-1 ${showHistory ? 'bg-blue-600 justify-end' : 'bg-slate-700 justify-start'}`}
            >
              <div className="w-4 h-4 rounded-full bg-white shadow-sm" />
            </button>
          </div>

        </div>

        {/* COMING SOON SECTION */}
        <div className="mb-2 opacity-60 pointer-events-none relative">
          <div className="text-[10px] text-slate-500 uppercase mb-4 font-bold tracking-widest flex items-center gap-2">
            Coming Soon
            <span className="text-[9px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-400">Dev Preview</span>
          </div>

          {/* Feature A: Safeguard Toggle */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm font-medium">Enable Engine Safeguard</span>
              <div className="w-12 h-6 rounded-full bg-blue-900/40 flex items-center justify-end p-1 border border-blue-500/30">
                <div className="w-4 h-4 rounded-full bg-blue-500/50 shadow-sm" />
              </div>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              When enabled, Stockfish will double-check the Neural Net's moves to prevent blunders.
            </p>
          </div>

          {/* Feature B: Intervention Budget */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm font-medium">Intervention Budget</span>
              <div className="flex items-center gap-2">
                <div className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-500 font-mono w-12 text-center">8</div>
                <div className="w-4 h-4 border border-slate-600 rounded bg-slate-800 flex items-center justify-center">
                  {/* Unchecked */}
                </div>
                <span className="text-[10px] text-slate-500 uppercase">Unlimited</span>
              </div>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Limits how many times Stockfish can help the Neural Net. Set to 0 to force raw imitation style.
            </p>
          </div>

          {/* Feature C: Theme */}
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm font-medium">App Theme</span>
              <div className="flex bg-slate-800/50 border border-slate-700/50 rounded-lg p-1 gap-1">
                <button className="px-3 py-1 rounded bg-slate-700/50 text-slate-400 text-xs font-medium shadow-sm cursor-not-allowed">Dark</button>
                <button className="px-3 py-1 rounded text-slate-600 text-xs font-medium cursor-not-allowed">Light</button>
              </div>
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Switch between Dark and Light modes.
            </p>
          </div>

          {/* Overlay to block interaction more explicitly if needed, but pointer-events-none does it on parent */}
        </div>

      </div>
    </div>
  );
}; // Close SettingsModal

const Button = ({ onClick, children, variant = 'primary', className = '' }) => {
  const baseStyle = "px-6 py-3 rounded-xl font-bold uppercase tracking-wider transition-colors duration-200";
  const variants = {
    primary: "bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_15px_rgba(37,99,235,0.4)]",
    secondary: "bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600"
  };
  return (
    <button onClick={onClick} className={`${baseStyle} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
};

// --- HELPERS (Continued) ---

const SanRenderer = ({ san, color }) => {
  const firstChar = san.charAt(0);
  const isPiece = ['N', 'B', 'R', 'Q', 'K'].includes(firstChar);

  // Map SAN char to internal piece type char
  const pieceTypeMap = { 'N': 'n', 'B': 'b', 'R': 'r', 'Q': 'q', 'K': 'k' };

  if (isPiece) {
    const pieceType = pieceTypeMap[firstChar];
    return (
      <span className="inline-flex items-center gap-1">
        <img
          src={getPieceImg(pieceType, color)}
          alt={firstChar}
          className="w-3.5 h-3.5 select-none"
        />
        <span>{san.slice(1)}</span>
      </span>
    );
  }

  return <span>{san}</span>;
};


// --- MAIN APP ---

export default function App() {
  const [game, setGame] = useState(new Chess());
  const [moveHistory, setMoveHistory] = useState([]); // Persistent history
  const [gameOverReason, setGameOverReason] = useState(''); // New state for specific reason
  const [isThinking, setIsThinking] = useState(false);
  const [playerSide, setPlayerSide] = useState('white');
  const [gameStatus, setGameStatus] = useState('SPLASH'); // Initial state is Splash
  const [winner, setWinner] = useState(null);

  // Settings
  const [showGlow, setShowGlow] = useState(true);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [showHistory, setShowHistory] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isInfoOpen, setIsInfoOpen] = useState(false);

  // Bot Stats
  const [botStats, setBotStats] = useState({
    confidence: 0,
    candidates: [],
    lastMove: '',
    bgLog: []
  });

  const [thinkingLog, setThinkingLog] = useState([]);
  const thinkingInterval = useRef(null);
  const moveListRef = useRef(null);

  useEffect(() => {
    if (moveListRef.current) {
      moveListRef.current.scrollTop = moveListRef.current.scrollHeight;
    }
  }, [moveHistory.length]);

  const captured = getCapturedPieces(game);

  const playerCaptured = playerSide === 'white' ? captured['w'] : captured['b'];
  const botCaptured = playerSide === 'white' ? captured['b'] : captured['w'];

  // --- LOGIC ---

  const checkGameOver = (gameInstance) => {
    if (gameInstance.isGameOver()) {
      // Delay showing the modal so it's not a "bomb in face"
      setTimeout(() => {
        setGameStatus('GAME_OVER');
        if (gameInstance.isCheckmate()) {
          setWinner(gameInstance.turn() === 'w' ? 'black' : 'white');
          setGameOverReason('Checkmate');
        } else if (gameInstance.isDraw()) {
          setWinner('draw');
          if (gameInstance.isStalemate()) setGameOverReason('Stalemate');
          else if (gameInstance.isThreefoldRepetition()) setGameOverReason('Repetition');
          else if (gameInstance.isInsufficientMaterial()) setGameOverReason('Insufficient Material');
          else setGameOverReason('Draw');
        }
      }, 1000); // 1 second delay
      return true;
    }
    return false;
  };

  const safeGameMutate = (modify) => {
    setGame((g) => {
      const update = new Chess(g.fen());
      modify(update);
      return update;
    });
  };

  const startThinkingAnimation = () => {
    setThinkingLog(["Neural Net initialized.", "Scanning patterns..."]);
    let stage = 0;
    const stages = [
      "Depth 15 search...",
      "Evaluating tactics...",
      "Safety verification...",
      "Positional analysis...",
      "Finalizing move..."
    ];
    if (thinkingInterval.current) clearInterval(thinkingInterval.current);
    thinkingInterval.current = setInterval(() => {
      if (stage < stages.length) {
        setThinkingLog(prev => [...prev.slice(-5), stages[stage]]);
        stage++;
      }
    }, 800);
  };

  const stopThinkingAnimation = (finalLogs) => {
    if (thinkingInterval.current) clearInterval(thinkingInterval.current);
    if (finalLogs && finalLogs.length > 0) {
      setThinkingLog(finalLogs.slice(-6));
    } else {
      setThinkingLog(prev => [...prev, "Execution complete."]);
    }
  };

  const API_BASE = import.meta.env.PROD ? '' : 'http://localhost:8000';

  const makeBotMove = async (currentFen) => {
    setIsThinking(true);
    startThinkingAnimation();

    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        fen: currentFen,
      });

      const { move, confidence, candidates, thinking_log } = response.data;

      setBotStats({
        confidence,
        candidates: candidates || [],
        lastMove: move,
        bgLog: thinking_log || []
      });

      stopThinkingAnimation(thinking_log);

      safeGameMutate((game) => {
        const from = move.substring(0, 2);
        const to = move.substring(2, 4);
        const promotion = move.length > 4 ? move.substring(4) : 'q';
        const result = game.move({ from, to, promotion });
        if (result) {
          setMoveHistory(prev => [...prev, result.san]);
        }
        checkGameOver(game);
      });

    } catch (err) {
      console.error("Bot Error:", err);
      setIsThinking(false);
    } finally {
      setIsThinking(false);
    }
  };

  const onDrop = (sourceSquare, targetSquare) => {
    if (gameStatus !== 'PLAYING') return false;
    if (isThinking) return false;
    if (game.turn() !== playerSide[0]) return false;

    const gameCopy = new Chess(game.fen());
    try {
      const move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });
      if (move === null) return false;

      setGame(gameCopy);
      setMoveHistory(prev => [...prev, move.san]);

      if (!checkGameOver(gameCopy)) {
        makeBotMove(gameCopy.fen());
      }
      return true;
    } catch (e) {
      return false;
    }
  };

  const startGame = (side) => {
    setPlayerSide(side);
    setGame(new Chess());
    setMoveHistory([]);
    setGameStatus('PLAYING');
    setWinner(null);
    setThinkingLog([]);
  };

  useEffect(() => {
    if (gameStatus === 'PLAYING' && playerSide === 'black' && game.turn() === 'w' && !isThinking) {
      makeBotMove(game.fen());
    }
  }, [gameStatus, playerSide, game]); // eslint-disable-line react-hooks/exhaustive-deps


  // --- RENDER ---



  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex flex-col overflow-hidden relative selection:bg-blue-500/30">

      {/* HEADER - Absolute Top */}
      <header className="absolute top-0 w-full h-16 bg-transparent flex items-center justify-between px-8 z-40 pointer-events-none">
        <h1 className="text-xl font-bold tracking-widest text-slate-100/50 pointer-events-auto">
          Chess Bot v2
        </h1>

        <div className="flex items-center gap-2 pointer-events-auto">
          {/* Info Button */}
          <button
            onClick={() => { setIsInfoOpen(!isInfoOpen); setIsSettingsOpen(false); }}
            className={`p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors ${isInfoOpen ? 'bg-slate-800 text-white' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>

          {/* Settings Button */}
          <button
            onClick={() => { setIsSettingsOpen(!isSettingsOpen); setIsInfoOpen(false); }}
            className={`p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors ${isSettingsOpen ? 'bg-slate-800 text-white' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>

        {isSettingsOpen && (
          <SettingsModal
            showGlow={showGlow}
            setShowGlow={setShowGlow}
            showAnalysis={showAnalysis}
            setShowAnalysis={setShowAnalysis}
            showHistory={showHistory}
            setShowHistory={setShowHistory}
            onClose={() => setIsSettingsOpen(false)}
          />
        )}

        {isInfoOpen && (
          <InfoModal onClose={() => setIsInfoOpen(false)} />
        )}
      </header>

      {/* --- SPLASH SCREEN --- */}
      {gameStatus === 'SPLASH' && (
        <SplashScreen onStart={startGame} />
      )}

      {/* --- MAIN CENTERED CONTENT --- */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="flex flex-row gap-16 items-center max-w-7xl w-full justify-center">

          {/* --- LEFT PANEL: STATS --- */}
          {/* --- LEFT PANEL: STATS --- */}
          {showAnalysis && (
            <aside className="w-[340px] bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col gap-6 shadow-2xl h-[550px] relative overflow-hidden backdrop-blur-sm">

              <div className="relative z-10 space-y-6 h-full flex flex-col font-sans">
                <div className="space-y-1">
                  <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Analysis Engine</div>
                  <div className="h-px w-full bg-slate-800" />
                </div>

                {/* Confidence */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-400 uppercase font-bold tracking-wider">
                    <span>Confidence</span>
                    <span className="text-blue-400">
                      {(botStats.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-600 transition-all duration-700 ease-out"
                      style={{ width: `${Math.max(5, botStats.confidence * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Decision Matrix */}
                <div className="flex-1 flex flex-col overflow-hidden">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-3 pr-2">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Decision Matrix</span>
                    {isThinking && <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse shadow-[0_0_8px_cyan]" />}
                  </div>

                  <div className="flex-1 overflow-y-auto pr-1 custom-scrollbar flex flex-col gap-2">
                    {/* Candidates List with Badges */}
                    {botStats.candidates.map((c, i) => {
                      let badgeColor = 'bg-slate-700 text-slate-300';
                      let badgeText = c.status || 'IGNORED';

                      if (c.status === 'SELECTED') {
                        badgeColor = 'bg-green-900/40 text-green-400 border border-green-800/50';
                      } else if (c.status === 'VETOED') {
                        badgeColor = 'bg-red-900/20 text-red-400 border border-red-800/50';
                        badgeText += ' (Blunder)';
                      } else {
                        badgeColor = 'bg-slate-800 text-slate-500 border border-slate-700';
                      }

                      return (
                        <div key={i} className="bg-slate-800/30 p-2 rounded-lg border border-slate-800/50 flex flex-col gap-1 relative group hover:bg-slate-800/60 transition-colors h-16">
                          <div className="flex justify-between items-center z-10">
                            <div className="flex items-center gap-2">
                              <span className="text-slate-600 text-[10px] font-mono w-3">{i + 1}.</span>
                              <span className="text-base text-white font-bold tracking-tight">{c.san || c.move}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`text-[10px] font-bold ${c.status === 'SELECTED' ? 'text-blue-400' : 'text-slate-500'}`}>
                                {(c.confidence * 100).toFixed(1)}%
                              </span>
                              <div className={`text-[9px] px-2 py-0.5 rounded uppercase font-bold tracking-wide ${badgeColor}`}>
                                {badgeText}
                              </div>
                            </div>
                          </div>

                          {/* Mini Prob Bar */}
                          <div className="w-full h-1 bg-slate-700/50 rounded-full overflow-hidden flex">
                            <div className="h-full bg-blue-500/60" style={{ width: `${c.confidence * 100}%` }} />
                          </div>
                        </div>
                      );
                    })}

                    {!isThinking && botStats.candidates.length === 0 && (
                      <div className="text-center text-slate-600 text-xs py-10 italic">
                        Waiting for turn...
                      </div>
                    )}

                    {isThinking && botStats.candidates.length === 0 && (
                      <div className="space-y-3 animate-pulse">
                        <div className="h-14 bg-slate-800/50 rounded-lg" />
                        <div className="h-14 bg-slate-800/50 rounded-lg" />
                        <div className="h-14 bg-slate-800/50 rounded-lg" />
                      </div>
                    )}
                  </div>
                </div>

              </div>

            </aside>
          )}

          {/* --- CENTER AREA: BOARD + CAPTURED BARS --- */}
          <div className="flex flex-col items-center gap-6">

            {/* TOP BAR: Bot's Taken Pieces */}
            <div className="w-full max-w-[550px] flex items-center justify-start gap-4 h-8">
              <div className="flex items-center gap-1 justify-start">
                {botCaptured.map((piece, i) => (
                  <img
                    key={i}
                    src={getPieceImg(piece.type, piece.color)}
                    alt={piece.type}
                    className={`w-6 h-6 transition-transform hover:scale-110 ${piece.color === 'b' ? 'drop-shadow-[0_0_2px_rgba(255,255,255,0.8)] opacity-100' : 'opacity-80 drop-shadow-lg'}`}
                  />
                ))}
              </div>
            </div>

            {/* BOARD */}
            {/* BOARD */}
            <div className="relative z-10 w-[550px] h-[550px]">
              {/* Gemini Glow Animation */}
              <div className={`gemini-glow-effect ${isThinking && showGlow ? 'active' : ''}`} />
              <div className={`gemini-glow-effect secondary ${isThinking && showGlow ? 'active' : ''}`} />

              <div className={`w-full h-full rounded-lg overflow-hidden border border-slate-800 bg-slate-900 transition-all duration-1000 ${isThinking && showGlow ? '' : 'shadow-2xl'}`}>
                <Chessboard
                  id="MainBoard"
                  position={game.fen()}
                  onPieceDrop={onDrop}
                  boardOrientation={playerSide}
                  customDarkSquareStyle={{ backgroundColor: '#1e293b' }}
                  customLightSquareStyle={{ backgroundColor: '#475569' }}
                  animationDuration={300}
                />
              </div>
            </div>

            {/* BOTTOM BAR: Player's Taken Pieces */}
            <div className="w-full max-w-[550px] flex items-center justify-start gap-4 h-8">
              <div className="flex items-center gap-1 justify-start">
                {playerCaptured.map((piece, i) => (
                  <img
                    key={i}
                    src={getPieceImg(piece.type, piece.color)}
                    alt={piece.type}
                    className={`w-6 h-6 transition-transform hover:scale-110 ${piece.color === 'b' ? 'drop-shadow-[0_0_2px_rgba(255,255,255,0.8)] opacity-100' : 'opacity-80 drop-shadow-lg'}`}
                  />
                ))}
              </div>
            </div>

          </div>

          {/* --- RIGHT PANEL: MOVE LIST --- */}
          {/* Note: Height 550px matches Board */}
          {showHistory && (
            <aside className="w-[340px] bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col gap-6 shadow-2xl h-[550px] relative overflow-hidden backdrop-blur-sm">
              <div className="space-y-1 relative z-10">
                <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Move History</div>
                <div className="h-px w-full bg-slate-800" />
              </div>

              <div ref={moveListRef} className="flex-1 overflow-y-auto pr-1 space-y-0.5 relative z-10 custom-scrollbar">
                {(() => {
                  const history = moveHistory; // Render from moveHistory state
                  const moves = [];
                  for (let i = 0; i < history.length; i += 2) {
                    moves.push({
                      num: Math.floor(i / 2) + 1,
                      white: history[i],
                      black: history[i + 1] || '',
                      whiteIndex: i,
                      blackIndex: i + 1
                    });
                  }
                  if (moves.length === 0) return <div className="text-slate-700 text-xs italic text-center mt-10">No moves yet</div>;

                  return moves.map((m, i) => (
                    <div key={i} className="flex text-sm py-1 border-b border-slate-800/20 items-center hover:bg-white/5 transition-colors px-1">
                      <span className="text-slate-600 w-8 text-xs font-mono select-none">{m.num}.</span>

                      {/* White Move */}
                      <span className={`w-20 transition-all duration-300 flex items-center ${m.whiteIndex === history.length - 1 ? 'text-white font-bold opacity-100' : 'text-slate-400 opacity-50'}`}>
                        <SanRenderer san={m.white} color="w" />
                      </span>

                      {/* Black Move */}
                      <span className={`w-20 transition-all duration-300 flex items-center ${m.blackIndex === history.length - 1 ? 'text-white font-bold opacity-100' : 'text-slate-400 opacity-50'}`}>
                        {m.black && <SanRenderer san={m.black} color="b" />}
                      </span>
                    </div>
                  ));
                })()}
              </div>
            </aside>
          )}

        </div>
      </div>

      {/* --- MODALS --- */}
      {gameStatus === 'GAME_OVER' && (
        <Modal onClose={() => setGameStatus('SPLASH')}>
          <div className="animate-scale-in text-center">
            <h2 className="text-3xl font-bold mb-6 text-white tracking-widest font-display">
              {winner === playerSide ? 'VICTORY' : winner === 'draw' ? 'DRAW' : 'DEFEAT'}
            </h2>

            <p className="text-slate-500 mb-8 font-mono text-xs uppercase tracking-widest">
              Game Over by {gameOverReason}
            </p>

            <div className="flex flex-col gap-3">
              <Button onClick={() => setGameStatus('SPLASH')} variant="primary" className="w-full">
                Play Again
              </Button>
              <button
                onClick={() => setGameStatus('REVIEW')}
                className="text-slate-500 hover:text-white text-xs uppercase tracking-wider font-bold py-2 transition-colors"
              >
                View Board
              </button>
            </div>
          </div>
        </Modal>
      )}

      {/* Play Again Floating Button (For Review Mode) */}
      {gameStatus === 'REVIEW' && (
        <div className="fixed bottom-8 right-8 z-50 animate-scale-in">
          <Button onClick={() => setGameStatus('SPLASH')} variant="primary" className="shadow-2xl">
            New Game
          </Button>
        </div>
      )}

    </div>
  );
}
