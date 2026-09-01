import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  base: '/gpt/',
  plugins: [vue()],
  server: {
    port: 5000,
    host: '0.0.0.0',
    allowedHosts: true,
    proxy: {
      '/gpt/api': {
        target: 'http://localhost:8021',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/gpt\/api/, ''),
        timeout: 180000,
        proxyTimeout: 180000,
      },
      '/api': {
        target: 'http://localhost:8021',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        timeout: 180000,
        proxyTimeout: 180000,
      }
    }
  }
})
