# Frontend Setup Guide

## Quick Start

### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 3. Start Backend API Server

In a separate terminal:

```bash
# Install Flask dependencies (if not already installed)
pip install flask flask-cors

# Start the API server
python api_server.py
```

The API will be available at `http://localhost:8000`

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── GradientButton.tsx
│   │   ├── GlassCard.tsx
│   │   ├── ProgressCircle.tsx
│   │   ├── UploadBox.tsx
│   │   ├── FeatureList.tsx
│   │   ├── Navbar.tsx
│   │   ├── Footer.tsx
│   │   └── AnimatedBackground.tsx
│   ├── pages/              # Page components
│   │   ├── Landing.tsx
│   │   ├── Upload.tsx
│   │   ├── Results.tsx
│   │   └── Dashboard.tsx
│   ├── api/                # API integration
│   │   └── predict.ts
│   ├── App.tsx             # Main app with routing
│   ├── main.tsx            # Entry point
│   └── index.css           # Global styles
├── public/                 # Static assets
├── index.html
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## Features

✅ Instagram-inspired gradient design  
✅ Dark mode first  
✅ Glassmorphism effects  
✅ Smooth animations with Framer Motion  
✅ Fully responsive (mobile → desktop)  
✅ TypeScript for type safety  
✅ TailwindCSS for styling  

## Pages

- **Landing** (`/`) - Marketing page with hero section and features
- **Upload** (`/upload`) - Video upload form with drag-and-drop
- **Results** (`/results`) - Prediction results with animations
- **Dashboard** (`/dashboard`) - Admin dashboard with stats and history

## API Integration

The frontend expects a backend API at `/api/predict` that:

- Accepts: `POST /api/predict`
  - FormData with: `video` (File), `title` (string), `hashtags` (string), `niche` (string)
- Returns: 
  ```json
  {
    "label": "High" | "Low",
    "score": 87.5,
    "top_features": [["feature_name", 0.25], ...]
  }
  ```

## Build for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

## Customization

### Colors

Edit `tailwind.config.js` to modify the Instagram gradient colors:

```js
colors: {
  instaBlue: "#405DE6",
  instaPurple: "#5851DB",
  // ... etc
}
```

### Components

All components are in `src/components/` and can be easily customized.

## Troubleshooting

**Port already in use?**
- Change the port in `vite.config.ts` or use `npm run dev -- --port 3001`

**API connection issues?**
- Ensure the backend server is running on port 8000
- Check CORS settings in `api_server.py`

**Build errors?**
- Run `npm install` again
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`

