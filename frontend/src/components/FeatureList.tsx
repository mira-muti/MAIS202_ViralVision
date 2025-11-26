import { motion } from 'framer-motion'
import { TrendingUp } from 'lucide-react'

interface Feature {
  name: string
  importance: number
}

interface FeatureListProps {
  features: Feature[]
  maxItems?: number
}

export default function FeatureList({ features, maxItems = 5 }: FeatureListProps) {
  const displayFeatures = features.slice(0, maxItems)
  const maxImportance = Math.max(...features.map(f => f.importance))

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-purple" />
        Top Features
      </h3>
      {displayFeatures.map((feature, index) => (
        <motion.div
          key={feature.name}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="glass rounded-xl p-4"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-300">
              {feature.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </span>
            <span className="text-xs text-purple font-semibold">
              {(feature.importance * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-primary rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(feature.importance / maxImportance) * 100}%` }}
              transition={{ delay: index * 0.1 + 0.3, duration: 0.8 }}
            />
          </div>
        </motion.div>
      ))}
    </div>
  )
}

