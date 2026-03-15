import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#e8ebf3',
          100: '#c5cce0',
          200: '#9eaacb',
          300: '#7789b6',
          400: '#5970a6',
          500: '#3b5896',
          600: '#33508e',
          700: '#294583',
          800: '#1f3b79',
          900: '#1a2b5f',
          950: '#0e1a3d',
        },
        orange: {
          50: '#fff3e0',
          100: '#ffe0b2',
          200: '#ffcc80',
          300: '#ffb74d',
          400: '#ffa726',
          500: '#ff7d00',
          600: '#f57c00',
          700: '#ef6c00',
          800: '#e65100',
          900: '#bf360c',
        },
        court: {
          wood: '#c4813d',
          line: '#ffffff',
          dark: '#8b5e34',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      animation: {
        'pulse-live': 'pulse-live 2s ease-in-out infinite',
        'fade-in': 'fade-in 0.3s ease-out',
        'slide-up': 'slide-up 0.3s ease-out',
      },
      keyframes: {
        'pulse-live': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.4' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config
