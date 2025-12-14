import React, { useState, useEffect, useRef } from 'react';

// Help component for counting numbers
const CountUp = ({ end, duration = 1500, suffix = '', decimals = 0 }) => {
    const [count, setCount] = useState(0);
    const startTime = useRef(null);

    useEffect(() => {
        let animationFrame;

        const animate = (timestamp) => {
            if (!startTime.current) startTime.current = timestamp;
            const progress = timestamp - startTime.current;

            // Easing function (easeOutExpo) for a technical feel
            const easeOutExpo = (x) => (x === 1 ? 1 : 1 - Math.pow(2, -10 * x));

            const percentage = Math.min(progress / duration, 1);
            const currentVal = easeOutExpo(percentage) * end;

            setCount(currentVal);

            if (progress < duration) {
                animationFrame = requestAnimationFrame(animate);
            } else {
                setCount(end);
            }
        };

        animationFrame = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(animationFrame);
    }, [end, duration]);

    // Format with commas
    const formatted = count.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });

    return <span>{formatted}{suffix}</span>;
};

const SplashScreen = ({ onStart }) => {
    const [isExiting, setIsExiting] = useState(false);

    const handleStart = (side) => {
        setIsExiting(true);
        setTimeout(() => {
            onStart(side);
        }, 500); // Matches the CSS duration
    };

    return (
        <div className={`fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/90 backdrop-blur-sm transition-opacity duration-500 ${isExiting ? 'opacity-0' : 'opacity-100'}`}>

            {/* Main Card */}
            <div className={`relative bg-slate-900 border border-slate-700 rounded-2xl shadow-[0_0_80px_rgba(2,6,23,0.8)] max-w-2xl w-full mx-4 overflow-hidden transform transition-all duration-500 ${isExiting ? 'scale-95' : 'scale-100'}`}>

                <div className="p-8 md:p-10 flex flex-col items-center text-center">

                    {/* Header section */}
                    <div className="mb-8 space-y-2">
                        <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
                            Chess Bot <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">v2</span>
                        </h1>
                        <p className="text-slate-400 text-lg">Chess Player Imitation Model</p>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full mb-8">
                        {/* Stat 1 */}
                        <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                            <div className="text-2xl md:text-3xl font-bold text-white font-mono">
                                <CountUp end={3437} />
                            </div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider mt-1">Games Analyzed</div>
                        </div>

                        {/* Stat 2 */}
                        <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                            <div className="text-2xl md:text-3xl font-bold text-white font-mono">
                                <CountUp end={171104} />
                            </div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider mt-1">Total Samples</div>
                        </div>

                        {/* Stat 3 */}
                        <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 relative overflow-hidden">
                            {/* Subtle glow for accuracy */}
                            <div className="absolute inset-0 bg-blue-500/5 pointer-events-none" />
                            <div className="text-2xl md:text-3xl font-bold text-cyan-400 font-mono">
                                <CountUp end={27.28} decimals={2} suffix="%" />
                            </div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider mt-1">Val Accuracy</div>
                            <div className="text-[10px] text-slate-600 mt-1">@ Epoch 6</div>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-4 w-full max-w-md">
                        <button
                            onClick={() => handleStart('white')}
                            className="flex-1 bg-slate-100 hover:bg-white text-slate-900 font-bold py-4 rounded-xl transition-all hover:scale-[1.02] shadow-[0_0_20px_rgba(255,255,255,0.2)] active:scale-95"
                        >
                            Play as White
                        </button>
                        <button
                            onClick={() => handleStart('black')}
                            className="flex-1 bg-slate-800 hover:bg-slate-700 text-white font-bold py-4 rounded-xl border border-slate-600 transition-all hover:scale-[1.02] active:scale-95"
                        >
                            Play as Black
                        </button>
                    </div>

                </div>
            </div>
        </div>
    );
};

export default SplashScreen;
