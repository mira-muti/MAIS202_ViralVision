import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { History as HistoryIcon, Video, Calendar, TrendingUp } from 'lucide-react'
import GlassCard from '../components/GlassCard'
import AnimatedBackground from '../components/AnimatedBackground'
import GradientButton from '../components/GradientButton'
import { useNavigate } from 'react-router-dom'

interface PredictionLog {
  timestamp: string
  date: string
  video_name: string
  title: string
  niche: string
  predicted_label: string
  prob_high: number
  prob_low: number
}

export default function History() {
  const navigate = useNavigate()
  const [predictions, setPredictions] = useState<PredictionLog[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadHistory()
  }, [])

  const loadHistory = async () => {
    try {
      const response = await fetch('/api/history')
      if (response.ok) {
        const data = await response.json()
        setPredictions(data)
      }
    } catch (error) {
      console.error('Failed to load history:', error)
    } finally {
      setLoading(false)
    }
  }

  const getLabelColor = (label: string) => {
    return label === 'High' 
      ? 'bg-green-500/20 text-green-400 border-green-500/30'
      : 'bg-orange-500/20 text-orange-400 border-orange-500/30'
  }

  return (
    <div className="min-h-screen pt-20 pb-20 px-4 sm:px-6 lg:px-8">
      <AnimatedBackground />
      
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold mb-2">
                Prediction <span className="text-gradient">History</span>
              </h1>
              <p className="text-gray-400">View all your past predictions</p>
            </div>
            <GradientButton onClick={() => navigate('/analyze')}>
              New Prediction
            </GradientButton>
          </div>
        </motion.div>

        {loading ? (
          <GlassCard className="text-center py-12">
            <div className="animate-spin w-8 h-8 border-4 border-deepPurple border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-gray-400">Loading history...</p>
          </GlassCard>
        ) : predictions.length === 0 ? (
          <GlassCard variant="gradient-border" className="text-center py-12">
            <HistoryIcon className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No predictions yet</h3>
            <p className="text-gray-400 mb-6">Start analyzing videos to see your history here</p>
            <GradientButton onClick={() => navigate('/analyze')}>
              Analyze Your First Video
            </GradientButton>
          </GlassCard>
        ) : (
          <GlassCard variant="gradient-border" className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-4 px-6 font-semibold text-gray-300">Date</th>
                    <th className="text-left py-4 px-6 font-semibold text-gray-300">Video Name</th>
                    <th className="text-left py-4 px-6 font-semibold text-gray-300">Title</th>
                    <th className="text-left py-4 px-6 font-semibold text-gray-300">Niche</th>
                    <th className="text-left py-4 px-6 font-semibold text-gray-300">Engagement</th>
                    <th className="text-left py-4 px-6 font-semibold text-gray-300">High %</th>
                    <th className="text-left py-4 px-6 font-semibold text-gray-300">Low %</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((prediction, index) => (
                    <motion.tr
                      key={prediction.timestamp}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="border-b border-white/5 hover:bg-white/5 transition-colors group"
                    >
                      <td className="py-4 px-6">
                        <div className="flex items-center gap-2">
                          <Calendar className="w-4 h-4 text-gray-500" />
                          <span className="text-sm text-gray-400">{prediction.date}</span>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center gap-2">
                          <Video className="w-4 h-4 text-gray-500" />
                          <span className="text-sm font-medium truncate max-w-[200px]">
                            {prediction.video_name}
                          </span>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <span className="text-sm text-gray-300 truncate max-w-[250px] block">
                          {prediction.title || 'No title'}
                        </span>
                      </td>
                      <td className="py-4 px-6">
                        <span className="text-sm text-gray-400 capitalize">
                          {prediction.niche}
                        </span>
                      </td>
                      <td className="py-4 px-6">
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-medium border ${getLabelColor(prediction.predicted_label)}`}
                        >
                          {prediction.predicted_label}
                        </span>
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-green-400" />
                          <span className="text-sm font-semibold text-green-400">
                            {(prediction.prob_high * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <span className="text-sm font-semibold text-orange-400">
                          {(prediction.prob_low * 100).toFixed(1)}%
                        </span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        )}
      </div>
    </div>
  )
}

