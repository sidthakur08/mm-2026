import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api/ncaa-live-scores': {
        target: 'https://ncaa-api.henrygd.me',
        changeOrigin: true,
        rewrite: () => {
          const today = new Date()
          const year = today.getFullYear()
          const month = String(today.getMonth() + 1).padStart(2, '0')
          const day = String(today.getDate()).padStart(2, '0')
          return `/scoreboard/basketball-men/d1/${year}/${month}/${day}`
        },
      },
    },
  },
})
