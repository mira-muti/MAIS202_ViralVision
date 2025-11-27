import { useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { CheckCircle, XCircle, RotateCcw, TrendingUp, Sparkles, Lightbulb, Sparkle, Zap, AlertCircle } from 'lucide-react'
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
  // New backend fields
  audio_features?: Record<string, number>
  visual_features?: Record<string, number>
  text_features?: {
    caption_length: number
    hashtag_count: number
    niche: string
  }
  features?: {
    audio?: Record<string, number>
    visual?: Record<string, number>
    text?: {
      caption_length: number
      hashtag_count: number
    }
  }
  positives?: string[]
  improvements?: string[]

  // Legacy / optional fields from older backend
  top_positive_features?: Feature[]
  top_negative_features?: Feature[]
  recommendations?: string[]
  raw_feature_importances?: Record<string, number>
}

export default function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const result = location.state?.result as PredictionResult | undefined

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

  // Gracefully handle missing legacy feature arrays
  const topPositiveFeatures = result.top_positive_features ?? []
  const topNegativeFeatures = result.top_negative_features ?? []
  const recommendations = result.recommendations ?? result.positives ?? []

  // Feature labels
  const getFeatureLabel = (featureName: string) => {
    const featureMap: Record<string, string> = {
      'caption_length': 'Caption Strength',
      'hashtag_count': 'Hashtag Strategy',
      'fft_max_freq': 'Audio Vibe',
      'fft_max_amp': 'Audio Presence',
      'engagement_ratio': 'Engagement Potential',
      'avg_brightness': 'Visual Brightness',
      'avg_color_variance': 'Color Diversity',
      'motion_intensity': 'Motion Dynamics',
      'scene_change_rate': 'Scene Transitions',
      'hue_entropy': 'Color Richness',
      'face_present': 'Face Presence',
      'text_overlay_present': 'Text Overlay',
    }
    return featureMap[featureName] || featureName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  // Positive insights
  const getPositiveInsight = (feature: Feature) => {
    const name = feature.feature
    if (name.includes('fft_max_freq')) return 'Audio vibe matches aesthetic 💅'
    if (name.includes('fft_max_amp')) return 'Audio presence is strong 🎵'
    if (name.includes('caption')) return 'Caption tone is on point ✨'
    if (name.includes('hashtag')) return 'Hashtag diversity is good 📱'
    if (name.includes('brightness')) return 'Visual brightness is optimal ✨'
    if (name.includes('motion')) return 'Motion dynamics are engaging 🎬'
    if (name.includes('face')) return 'Face presence adds personal connection 👤'
    return 'This feature contributes positively'
  }

  // Category-based improvement analysis
  interface ImprovementCategory {
    id: string
    title: string
    description: string
    score: number // 0-1, lower is worse
    impact: number // percentage impact on engagement
    recommendation: string
  }

  const calculateCategoryScores = (): ImprovementCategory[] => {
    const audio = result.features?.audio || result.audio_features || {}
    const visual = result.features?.visual || result.visual_features || {}
    const text = result.features?.text || result.text_features || {}
    
    const categories: ImprovementCategory[] = []

    // 1. Hook Strength (caption_length, avg_brightness, face_presence, tempo)
    const captionLength = (text as any).caption_length || 0
    const brightness = visual.avg_brightness || 0
    const facePresent = visual.face_present || 0
    const tempo = audio.tempo || 0
    
    // Normalize and weight features (0-1 scale, higher is better)
    const captionScore = Math.min(captionLength / 50, 1) // Optimal around 50 chars
    const brightnessScore = Math.min(brightness / 150, 1) // Optimal around 150
    const faceScore = facePresent // Binary
    const tempoScore = tempo > 0 ? Math.min(tempo / 120, 1) : 0.5 // Optimal around 120 BPM
    
    const hookScore = (captionScore * 0.3 + brightnessScore * 0.3 + faceScore * 0.2 + tempoScore * 0.2)
    const hookImpact = topNegativeFeatures
      .filter(f => ['caption_length', 'avg_brightness', 'face_present', 'tempo'].some(n => f.feature.includes(n)))
      .reduce((sum, f) => sum + f.importance, 0) * 100

    if (hookScore < 0.6 && hookImpact > 0) {
      let rec = ''
      if (captionLength < 30) rec = 'Add a punchier opening with bold text or a strong first-second hook.'
      else if (brightness < 100) rec = 'Increase opening brightness by 20-30% to grab attention immediately.'
      else if (facePresent === 0) rec = 'Show your face in the first 1-2 seconds to create instant connection.'
      else if (tempo < 100) rec = 'Add faster-paced intro cuts or sync to a higher-energy beat.'
      else rec = 'Strengthen your first 1-2 seconds with a punchier opening cut or text hook.'
      
      categories.push({
        id: 'hook',
        title: 'Hook Strength',
        description: 'Your first 1-2 seconds aren\'t strong enough to grab attention.',
        score: hookScore,
        impact: hookImpact,
        recommendation: rec
      })
    }

    // 2. Pacing & Movement (motion_intensity, scene_change_rate)
    const motionIntensity = visual.motion_intensity || 0
    const sceneChangeRate = visual.scene_change_rate || 0
    
    // Normalize (typical good values: motion > 10, scene_change > 50)
    const motionScore = Math.min(motionIntensity / 15, 1)
    const sceneScore = Math.min(sceneChangeRate / 100, 1)
    const pacingScore = (motionScore * 0.6 + sceneScore * 0.4)
    const pacingImpact = topNegativeFeatures
      .filter(f => ['motion_intensity', 'scene_change_rate'].some(n => f.feature.includes(n)))
      .reduce((sum, f) => sum + f.importance, 0) * 100

    if (pacingScore < 0.6 && pacingImpact > 0) {
      let rec = ''
      if (motionIntensity < 8) rec = 'Add 10-15% more cuts or transitions to improve pacing.'
      else if (sceneChangeRate < 40) rec = 'Increase scene transitions by adding quick cuts every 2-3 seconds.'
      else rec = 'Speed up pacing with more dynamic movement or quicker cuts.'
      
      categories.push({
        id: 'pacing',
        title: 'Pacing & Movement',
        description: 'Your pacing is too slow. Viewers may lose interest.',
        score: pacingScore,
        impact: pacingImpact,
        recommendation: rec
      })
    }

    // 3. Visual Appeal (avg_brightness, avg_color_variance / color_std_dev)
    const colorVariance = visual.avg_color_variance || visual.color_std_dev || 0
    
    const brightnessVisualScore = Math.min(brightness / 150, 1)
    const colorScore = Math.min(colorVariance / 60, 1) // Optimal around 60
    const visualScore = (brightnessVisualScore * 0.5 + colorScore * 0.5)
    const visualImpact = topNegativeFeatures
      .filter(f => {
        const feat = f.feature.toLowerCase()
        return feat.includes('brightness') || feat.includes('color') || feat.includes('variance')
      })
      .reduce((sum, f) => sum + f.importance, 0) * 100

    if (visualScore < 0.6 && visualImpact > 0) {
      let rec = ''
      if (brightness < 100) rec = 'Increase brightness by 15-25% or shoot in better natural lighting.'
      else if (colorVariance < 40) rec = 'Increase color contrast and saturation to make your video pop.'
      else rec = 'Enhance visual appeal by increasing brightness or color richness.'
      
      categories.push({
        id: 'visual',
        title: 'Visual Appeal (Color & Lighting)',
        description: 'Your lighting or color richness is low, reducing visual impact.',
        score: visualScore,
        impact: visualImpact,
        recommendation: rec
      })
    }

    // 4. Audio Engagement (rms_energy, spectral features, zcr)
    const rmsEnergy = audio.rms_energy || 0
    const spectralCentroid = audio.spectral_centroid || 0
    const zcr = audio.zcr || 0
    
    // Normalize (typical good values: rms > 0.05, spectral_centroid > 2000, zcr varies)
    const rmsScore = Math.min(rmsEnergy / 0.08, 1)
    const spectralScore = Math.min(spectralCentroid / 3000, 1)
    const zcrScore = zcr > 0.1 ? 0.5 : (zcr > 0.05 ? 0.8 : 1) // Lower ZCR often better for music
    const audioScore = (rmsScore * 0.5 + spectralScore * 0.3 + zcrScore * 0.2)
    const audioImpact = topNegativeFeatures
      .filter(f => ['rms_energy', 'zcr', 'spectral_centroid', 'spectral_rolloff', 'fft_max_freq', 'fft_max_amp'].some(n => f.feature.includes(n)))
      .reduce((sum, f) => sum + f.importance, 0) * 100

    if (audioScore < 0.6 && audioImpact > 0) {
      let rec = ''
      if (rmsEnergy < 0.03) rec = 'Boost audio loudness by 10-20% for better clarity and engagement.'
      else if (spectralCentroid < 1500) rec = 'Audio feels flat. Increase brightness of sound or add higher-frequency elements.'
      else rec = 'Enhance audio by increasing volume, removing background noise, or syncing cuts to the beat.'
      
      categories.push({
        id: 'audio',
        title: 'Audio Engagement',
        description: 'Your audio feels flat or quiet, reducing viewer engagement.',
        score: audioScore,
        impact: audioImpact,
        recommendation: rec
      })
    }

    // 5. Text/Caption Effectiveness (hashtag_count, caption_length)
    const hashtagCount = (text as any).hashtag_count || 0
    
    const captionTextScore = captionLength > 0 && captionLength < 100 ? Math.min(captionLength / 60, 1) : 0.5
    const hashtagScore = hashtagCount >= 3 && hashtagCount <= 8 ? 1 : (hashtagCount > 0 ? 0.7 : 0.3)
    const textScore = (captionTextScore * 0.6 + hashtagScore * 0.4)
    const textImpact = topNegativeFeatures
      .filter(f => ['caption_length', 'hashtag_count'].some(n => f.feature.includes(n)))
      .reduce((sum, f) => sum + f.importance, 0) * 100

    if (textScore < 0.6 && textImpact > 0) {
      let rec = ''
      if (captionLength === 0) rec = 'Add a short, punchy caption (3-7 words) with a strong keyword to strengthen the hook.'
      else if (captionLength > 100) rec = 'Shorten your caption to 3-7 words using a strong keyword for better engagement.'
      else if (hashtagCount === 0) rec = 'Add 3-5 relevant hashtags to improve discoverability.'
      else if (hashtagCount > 10) rec = 'Reduce to 3-6 highly relevant hashtags for better focus.'
      else rec = 'Optimize caption format: use shorter, punchier text with 1-3 strong keywords.'
      
      categories.push({
        id: 'text',
        title: 'Text/Caption Effectiveness',
        description: 'Your caption format may hurt engagement. Optimize for TikTok\'s format.',
        score: textScore,
        impact: textImpact,
        recommendation: rec
      })
    }

    // Sort by impact (highest first)
    return categories.sort((a, b) => b.impact - a.impact)
  }

  const improvementCategories = calculateCategoryScores()

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
            <Sparkle className="w-8 h-8 text-magenta" />
            <h1 className="text-4xl md:text-5xl font-bold">
              Your <span className="text-gradient">Results</span>
            </h1>
          </div>
          <p className="text-gray-400">AI-powered insights for Daily Content Creators</p>
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
                {topPositiveFeatures.length > 0 ? (
                  topPositiveFeatures.map((feature, index) => (
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
                <h3 className="text-xl font-bold">Areas to Improve (Content Strategy Insights)</h3>
              </div>
              <p className="text-sm text-gray-400 mb-4">Actionable improvements to boost engagement ✂️</p>
              <div className="space-y-3">
                {improvementCategories.length > 0 ? (
                  improvementCategories.map((category, index) => (
                    <motion.div
                      key={category.id}
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      className="glass rounded-xl p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-300">
                          {category.title}
                        </span>
                        <span className="text-xs text-orange-400 font-semibold">
                          -{category.impact.toFixed(1)}% impact
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 mb-2 leading-relaxed">
                        {category.description}
                      </p>
                      <p className="text-xs text-orange-400/90 mb-3 font-medium">
                        💡 {category.recommendation}
                      </p>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-orange-500 to-red-400 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(category.impact * 2, 100)}%` }}
                          transition={{ delay: 0.6 + index * 0.1, duration: 0.8 }}
                        />
                      </div>
                    </motion.div>
                  ))
                ) : (
                  <p className="text-gray-500 text-sm">No major areas to improve identified. Your content looks strong! 🎉</p>
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
              {recommendations.map((rec, index) => (
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
        {topNegativeFeatures.some(f => f.feature.toLowerCase().includes('niche')) && (
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
                    or embedding them with your GRWM content. Some niches 
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
