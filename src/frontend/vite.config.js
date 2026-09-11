import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Vite configuration for SignalScope frontend.
 *
 * Dev server proxy:
 *   All requests to /api/* are forwarded to the Flask backend
 *   running on http://localhost:5000 during development.
 *   This avoids CORS issues when running both servers locally.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
