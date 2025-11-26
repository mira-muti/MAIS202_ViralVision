import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Send, Video, Upload } from 'lucide-react'
import GradientButton from '../components/GradientButton'
import GlassCard from '../components/GlassCard'
import UploadBox from '../components/UploadBox'
import AnimatedBackground from '../components/AnimatedBackground'
import { predictVideo } from '../api/predict'

const niches = [
  { label: 'Music', value: 'music' },
  { label: 'GRWM', value: 'GRWM' },
]

export default function Analyze() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [hashtags, setHashtags] = useState('')
  const [niche, setNiche] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [videoPreview, setVideoPreview] = useState<string | null>(null)

  // Cleanup video preview URL on unmount
  useEffect(() => {
    return () => {
      if (videoPreview) {
        URL.revokeObjectURL(videoPreview)
      }
    }
  }, [videoPreview])

  const handleFileSelect = (selectedFile: File) => {
    // Cleanup previous preview URL
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview)
    }
    
    setFile(selectedFile)
    setError(null)
    
    // Create preview URL
    const url = URL.createObjectURL(selectedFile)
    setVideoPreview(url)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!file) {
      setError('Please select a video file')
      return
    }
    if (!title.trim()) {
      setError('Please enter a title')
      return
    }
    if (!niche) {
      setError('Please select a niche')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const result = await predictVideo(file, title, hashtags, niche)
      navigate('/results', { state: { result, niche: niche as 'music' | 'GRWM' } })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to predict video')
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen pt-20 pb-20 px-4 sm:px-6 lg:px-8">
      <AnimatedBackground />
      
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Analyze Your <span className="text-gradient">Video</span>
          </h1>
          <p className="text-gray-400">
            Upload your content and get AI-powered insights
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Form */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <GlassCard variant="gradient-border">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Video File
                  </label>
                  <UploadBox onFileSelect={handleFileSelect} />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Title / Caption
                  </label>
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="Enter your video title..."
                    className="w-full px-4 py-3 rounded-xl glass border border-white/20 focus:border-purple focus:outline-none transition-colors bg-[#0D0D0F]/50"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Hashtags
                  </label>
                  <input
                    type="text"
                    value={hashtags}
                    onChange={(e) => setHashtags(e.target.value)}
                    placeholder="#viral #trending #music"
                    className="w-full px-4 py-3 rounded-xl glass border border-white/20 focus:border-purple focus:outline-none transition-colors bg-[#0D0D0F]/50"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Separate hashtags with spaces
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Select Niche
                  </label>
                  <select
                    value={niche}
                    onChange={(e) => setNiche(e.target.value)}
                    className="w-full px-4 py-3 rounded-xl glass border border-white/20 focus:border-purple focus:outline-none transition-colors bg-[#0D0D0F]/50"
                  >
                    <option value="">Choose your niche...</option>
                    {niches.map((n) => (
                      <option key={n.value} value={n.value} className="bg-[#0D0D0F]">
                        {n.label}
                      </option>
                    ))}
                  </select>
                </div>

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400"
                  >
                    {error}
                  </motion.div>
                )}

                <GradientButton
                  type="submit"
                  size="lg"
                  isLoading={isLoading}
                  className="w-full"
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Video'}
                  {!isLoading && <Send className="w-5 h-5" />}
                </GradientButton>
              </form>
            </GlassCard>
          </motion.div>

          {/* Video Preview */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <GlassCard variant="gradient-border">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Video className="w-5 h-5 text-purple" />
                Video Preview
              </h3>
              {videoPreview ? (
                <div className="relative rounded-xl overflow-hidden bg-black/20">
                  <video
                    ref={videoRef}
                    src={videoPreview}
                    controls
                    className="w-full h-auto max-h-[400px]"
                  />
                </div>
              ) : (
                <div className="aspect-video rounded-xl glass flex items-center justify-center border-2 border-dashed border-white/20">
                  <div className="text-center">
                    <Upload className="w-12 h-12 text-gray-600 mx-auto mb-2" />
                    <p className="text-sm text-gray-500">Upload a video to see preview</p>
                  </div>
                </div>
              )}
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

