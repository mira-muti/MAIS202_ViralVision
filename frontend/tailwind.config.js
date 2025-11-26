/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ViralVision brand colors
        purple: "#8A2BE2",
        magenta: "#FF2EC4",
        violet: "#4A00E0",
        deepBlack: "#0D0D0F",
        softWhite: "rgba(255,255,255,0.9)",
      },
      backgroundImage: {
        "insta-gradient": "linear-gradient(135deg, #8A2BE2, #FF2EC4, #4A00E0)",
        "gradient-primary": "linear-gradient(135deg, #8A2BE2, #FF2EC4, #4A00E0)",
        "gradient-soft": "linear-gradient(135deg, #8A2BE2 0%, #FF2EC4 50%, #4A00E0 100%)",
        "gradient-radial": "radial-gradient(circle at 30% 20%, #8A2BE2 0%, #FF2EC4 50%, #4A00E0 100%)",
      },
      backdropBlur: {
        xs: "2px",
      },
      boxShadow: {
        "glow": "0 0 20px rgba(138, 43, 226, 0.4)",
        "glow-lg": "0 0 40px rgba(138, 43, 226, 0.6)",
        "glow-purple": "0 0 20px rgba(138, 43, 226, 0.4)",
        "glow-magenta": "0 0 20px rgba(255, 46, 196, 0.4)",
      },
      animation: {
        "gradient": "gradient 8s linear infinite",
        "float": "float 6s ease-in-out infinite",
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
      },
      keyframes: {
        gradient: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" },
        },
        "pulse-glow": {
          "0%, 100%": { boxShadow: "0 0 20px rgba(131, 58, 180, 0.3)" },
          "50%": { boxShadow: "0 0 40px rgba(131, 58, 180, 0.6)" },
        },
      },
    },
  },
  plugins: [],
}

