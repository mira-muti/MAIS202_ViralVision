# ViralVision Frontend

A modern, Instagram-inspired React frontend for predicting video engagement.

## Features

- 🎨 Instagram-inspired gradient design
- 🌙 Dark mode first
- ✨ Glassmorphism effects
- 🎭 Smooth animations with Framer Motion
- 📱 Fully responsive
- ⚡ Built with Vite + React + TypeScript

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Backend API

The frontend expects a backend API at `http://localhost:8000/api/predict` that accepts:

- `POST /api/predict`
  - FormData with: `video` (File), `title` (string), `hashtags` (string), `niche` (string)
  - Returns: `{ label: "High" | "Low", score: number, top_features: [string, number][] }`

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/         # Page components
│   ├── api/           # API integration
│   └── App.tsx        # Main app with routing
├── public/            # Static assets
└── index.html         # HTML entry point
```

