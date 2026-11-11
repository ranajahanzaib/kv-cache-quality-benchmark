import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (/\/node_modules\/(react|react-dom)\//.test(id))
            return "react-vendor";
        },
      },
    },
  },
});
