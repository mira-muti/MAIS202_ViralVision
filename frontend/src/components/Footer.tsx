import { Github, Instagram, Mail } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-white/10 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div>
            <p className="text-gradient font-bold text-lg mb-2">ViralVision</p>
            <p className="text-sm text-gray-400">
              Predict video engagement with AI-powered insights
            </p>
          </div>
          <div className="flex items-center gap-6">
            <a
              href="https://github.com/mira-muti/MAIS202_ViralVision"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-purple transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="https://www.instagram.com/mira.almuti/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-purple transition-colors"
            >
              <Instagram className="w-5 h-5" />
            </a>
            <a
              href="mailto:workwithmira1@gmail.com"
              className="text-gray-400 hover:text-purple transition-colors"
            >
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-white/10 text-center text-sm text-gray-500">
          <p>&copy; 2025 ViralVision. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}

