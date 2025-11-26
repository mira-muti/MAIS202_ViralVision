import { useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { CheckCircle, XCircle, RotateCcw, TrendingUp, Sparkles, Lightbulb, Music, Sparkle, Zap, AlertCircle } from 'lucide-react'
import GradientButton from '../components/GradientButton'
import GlassCard from '../components/GlassCard'
import ProgressCircle from '../components/ProgressCircle'
import AnimatedBackground from '../components/AnimatedBackground'

interface Feature {
  feature: string
  importance: number
}

interface PredictionResult {
  label: 'High' | 'Low'
  prob_high: number
  prob_low: number
  top_positive_features: Feature[]
  top_negative_features: Feature[]
  recommendations: string[]
  raw_feature_importances: Record<string, number>
}

export default function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const result = location.state?.result as PredictionResult | undefined
  const niche = location.state?.niche as 'music' | 'GRWM' | undefined

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">No prediction results found</p>
          <GradientButton onClick={() => navigate('/analyze')}>
            Analyze a Video
          </GradientButton>
        </div>
      </div>
    )
  }

  const isHigh = result.label === 'High'
  const successPercent = result.prob_high * 100  // This is the probability of HIGH engagement (success)
  const isMusic = niche === 'music'
  
  // Filter out niche-related features from negative features (they're not actionable)
  const filteredNegativeFeatures = result.top_negative_features.filter(
    f => !f.feature.toLowerCase().includes('niche')
  )

  // Niche-specific feature labels
  const getFeatureLabel = (featureName: string) => {
    const featureMap: Record<string, string> = {
      'caption_length': 'Caption Strength',
      'hashtag_count': 'Hashtag Strategy',
      'fft_max_freq': isMusic ? 'Audio Brightness' : 'Audio Vibe',
      'fft_max_amp': isMusic ? 'Audio Energy' : 'Audio Presence',
      'engagement_ratio': 'Engagement Potential',
    }
    return featureMap[featureName] || featureName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  // Niche-specific positive insights
  const getPositiveInsight = (feature: Feature) => {
    const name = feature.feature
    if (isMusic) {
      if (name.includes('fft_max_freq')) return 'Audio brightness is amazing 🎧'
      if (name.includes('fft_max_amp')) return 'Strong audio energy detected 🔥'
      if (name.includes('caption')) return 'Caption strength contributes well ✍️'
      if (name.includes('hashtag')) return 'Hashtag strategy is working 💯'
    } else {
      if (name.includes('fft_max_freq')) return 'Audio vibe matches aesthetic 💅'
      if (name.includes('fft_max_amp')) return 'Audio presence is strong 🎵'
      if (name.includes('caption')) return 'Caption tone is on point ✨'
      if (name.includes('hashtag')) return 'Hashtag diversity is good 📱'
    }
    return 'This feature contributes positively'
  }

  // Niche-specific improvement suggestions
  const getImprovementSuggestion = (feature: Feature) => {
    const name = feature.feature
    if (isMusic) {
      if (name.includes('caption')) return 'Try adding trending keywords or storytelling'
      if (name.includes('hashtag')) return 'Optimize hashtag selection for music niche'
      if (name.includes('fft_max_freq')) return 'Consider brighter, higher-energy audio'
      if (name.includes('fft_max_amp')) return 'Audio clarity could be improved'
    } else {
      if (name.includes('caption')) return 'Adjust caption length and tone'
      if (name.includes('hashtag')) return 'Focus hashtags on lifestyle/aesthetic tags'
      if (name.includes('fft_max_freq')) return 'Audio vibe could match aesthetic better'
      if (name.includes('fft_max_amp')) return 'Audio presence needs enhancement'
    }
    return 'Consider optimizing this feature'
  }

  return (
    <div className="min-h-screen pt-20 pb-20 px-4 sm:px-6 lg:px-8">
      <AnimatedBackground />
      
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            {isMusic ? (
              <Music className="w-8 h-8 text-purple" />
            ) : (
              <Sparkle className="w-8 h-8 text-magenta" />
            )}
            <h1 className="text-4xl md:text-5xl font-bold">
              Your <span className="text-gradient">Results</span>
            </h1>
          </div>
          <p className="text-gray-400">AI-powered insights for {isMusic ? 'Music' : 'GRWM'} creators</p>
        </motion.div>

        {/* Engagement Prediction */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <GlassCard variant="gradient-border" className="text-center">
            <div className="flex flex-col items-center justify-center mb-6">
              <ProgressCircle 
                value={successPercent} 
                size={240}
                label="Success Probability"
              />
            </div>
            
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mb-6"
            >
              <div className="flex items-center justify-center gap-3 mb-4">
                {isHigh ? (
                  <>
                    <CheckCircle className="w-8 h-8 text-green-400" />
                    <span className="text-3xl font-bold text-green-400">
                      High Success Potential
                    </span>
                  </>
                ) : (
                  <>
                    <XCircle className="w-8 h-8 text-orange-400" />
                    <span className="text-3xl font-bold text-orange-400">
                      Low Success Potential
                    </span>
                  </>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto mb-4">
                <div className="glass rounded-xl p-4">
                  <p className="text-sm text-gray-400 mb-1">Success Probability</p>
                  <p className="text-2xl font-bold text-green-400">
                    {(result.prob_high * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500 mt-1">Chance of high engagement</p>
                </div>
                <div className="glass rounded-xl p-4">
                  <p className="text-sm text-gray-400 mb-1">Low Engagement Risk</p>
                  <p className="text-2xl font-bold text-orange-400">
                    {(result.prob_low * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500 mt-1">Chance of low engagement</p>
                </div>
              </div>
            </motion.div>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className={`text-lg font-semibold ${isHigh ? 'text-green-400' : 'text-orange-400'}`}
            >
              {isHigh 
                ? '🎉 Your video has strong potential for success!'
                : `⚠️ Your video has a ${(result.prob_high * 100).toFixed(1)}% chance of high engagement. Consider the improvements below.`
              }
            </motion.p>
          </GlassCard>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          {/* Positive Signals */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <GlassCard variant="gradient-border">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-insta-gradient flex items-center justify-center">
                  <Zap className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-xl font-bold">Positive Signals</h3>
              </div>
              <p className="text-sm text-gray-400 mb-4">What you did well 🔥</p>
              <div className="space-y-3">
                {result.top_positive_features.length > 0 ? (
                  result.top_positive_features.map((feature, index) => (
                    <motion.div
                      key={feature.feature}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.4 + index * 0.1 }}
                      className="glass rounded-xl p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-300">
                          {getFeatureLabel(feature.feature)}
                        </span>
                        <span className="text-xs text-green-400 font-semibold">
                          +{(feature.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-xs text-green-400/80 mb-2">
                        {getPositiveInsight(feature)}
                      </p>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(feature.importance * 1000, 100)}%` }}
                          transition={{ delay: 0.5 + index * 0.1, duration: 0.8 }}
                        />
                      </div>
                    </motion.div>
                  ))
                ) : (
                  <p className="text-gray-500 text-sm">No positive features identified</p>
                )}
              </div>
            </GlassCard>
          </motion.div>

          {/* Areas to Improve */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <GlassCard variant="gradient-border">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-insta-gradient flex items-center justify-center">
                  <AlertCircle className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-xl font-bold">Areas to Improve</h3>
              </div>
              <p className="text-sm text-gray-400 mb-4">Optimization opportunities ✂️</p>
              <div className="space-y-3">
                {filteredNegativeFeatures.length > 0 ? (
                  filteredNegativeFeatures.map((feature, index) => (
                    <motion.div
                      key={feature.feature}
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      className="glass rounded-xl p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-300">
                          {getFeatureLabel(feature.feature)}
                        </span>
                        <span className="text-xs text-orange-400 font-semibold">
                          -{(feature.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-xs text-orange-400/80 mb-2">
                        {getImprovementSuggestion(feature)}
                      </p>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-orange-500 to-red-400 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(feature.importance * 1000, 100)}%` }}
                          transition={{ delay: 0.6 + index * 0.1, duration: 0.8 }}
                        />
                      </div>
                    </motion.div>
                  ))
                ) : (
                  <p className="text-gray-500 text-sm">No areas to improve identified</p>
                )}
              </div>
            </GlassCard>
          </motion.div>
        </div>

        {/* Recommendations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-6"
        >
          <GlassCard variant="gradient-border">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-insta-gradient flex items-center justify-center">
                <Lightbulb className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-xl font-bold">Personalized Recommendations</h3>
            </div>
            <div className="space-y-3">
              {result.recommendations.map((rec, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7 + index * 0.1 }}
                  className="flex items-start gap-3 glass rounded-xl p-4"
                >
                  <Sparkles className="w-5 h-5 text-purple flex-shrink-0 mt-0.5" />
                  <p className="text-gray-300">{rec}</p>
                </motion.div>
              ))}
            </div>
          </GlassCard>
        </motion.div>

        {/* Niche Exploration Suggestion */}
        {result.top_negative_features.some(f => f.feature.toLowerCase().includes('niche')) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="mb-6"
          >
            <GlassCard variant="gradient-border" className="border-purple/30">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple/20 flex items-center justify-center flex-shrink-0">
                  <TrendingUp className="w-5 h-5 text-purple" />
                </div>
                <div>
                  <h3 className="text-lg font-bold mb-2">Explore Other Niches</h3>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    If you're looking for higher overall engagement, consider exploring other niches 
                    or embedding them with your {isMusic ? 'music' : 'GRWM'} content. Some niches 
                    have shown higher average engagement rates and could complement your current style.
                  </p>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <GradientButton
            onClick={() => navigate('/analyze')}
            size="lg"
            className="w-full sm:w-auto"
          >
            <RotateCcw className="w-5 h-5" />
            Analyze Another Video
          </GradientButton>
          <GradientButton
            onClick={() => navigate('/history')}
            variant="outline"
            size="lg"
            className="w-full sm:w-auto"
          >
            View History
          </GradientButton>
        </motion.div>
      </div>
    </div>
  )
}
