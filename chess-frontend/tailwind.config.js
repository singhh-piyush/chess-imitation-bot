/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                cyber: {
                    950: '#020617', // Deepest background
                    900: '#0f172a', // Panel background
                    800: '#1e293b', // Lighter panel
                },
                neon: {
                    cyan: '#06b6d4',
                    blue: '#3b82f6',
                    green: '#22c55e',
                    magenta: '#d946ef',
                    red: '#ef4444',
                }
            },
            fontFamily: {
                mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', "Liberation Mono", "Courier New", 'monospace'],
                display: ['Orbitron', 'ui-sans-serif', 'system-ui'],
                sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            },
            keyframes: {
                'wavy-pulse': {
                    '0%, 100%': { transform: 'scaleY(1)', opacity: '0.4' },
                    '50%': { transform: 'scaleY(1.05)', opacity: '0.6' },
                },
                'slide-in-right': {
                    '0%': { transform: 'translateX(100%)', opacity: '0' },
                    '100%': { transform: 'translateX(0)', opacity: '1' },
                },
                'slide-out-right': {
                    '0%': { transform: 'translateX(0)', opacity: '1' },
                    '100%': { transform: 'translateX(100%)', opacity: '0' },
                },
                'scale-in': {
                    '0%': { transform: 'scale(0.95)', opacity: '0' },
                    '100%': { transform: 'scale(1)', opacity: '1' },
                }
            },
            animation: {
                'wavy-pulse': 'wavy-pulse 3s ease-in-out infinite',
                'slide-in-right': 'slide-in-right 0.3s ease-out forwards',
                'slide-out-right': 'slide-out-right 0.3s ease-in forwards',
                'scale-in': 'scale-in 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards', // Smooth pop-up
            }
        },
    },
    plugins: [],
}
