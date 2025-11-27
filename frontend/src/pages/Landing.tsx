import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Sparkles, Sparkle } from 'lucide-react'
import GradientButton from '../components/GradientButton'
import GlassCard from '../components/GlassCard'
import AnimatedBackground from '../components/AnimatedBackground'
import Footer from '../components/Footer'

export default function Landing() {
  return (
    <div className="min-h-screen">
      <AnimatedBackground />
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center max-w-4xl mx-auto"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="inline-flex items-center gap-2 glass rounded-full px-4 py-2 mb-8"
            >
              <Sparkles className="w-4 h-4 text-purple" />
              <span className="text-sm font-medium">AI-Powered Creator Tool</span>
            </motion.div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="text-gradient">AI Insights</span>
              <br />
              <span className="text-white">for Creators</span>
            </h1>
            
            <p className="text-xl text-gray-300 mb-4">
              For <span className="text-magenta font-semibold">Daily Content Creators</span>
            </p>
            
            <p className="text-lg text-gray-400 mb-10 max-w-2xl mx-auto">
              Predict the virality of your content before you post it. 
              Get instant AI-powered insights tailored to your niche.
            </p>
            
            <Link to="/analyze">
              <GradientButton size="lg" className="w-full sm:w-auto">
                Analyze Your Video
                <ArrowRight className="w-5 h-5" />
              </GradientButton>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Analysis Cards Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-4">
              What We <span className="text-gradient">Analyze</span>
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Tailored insights for Daily Content Creators
            </p>
          </motion.div>

          <div className="grid md:grid-cols-1 gap-8 max-w-3xl mx-auto">
            {/* GRWM Analysis Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
            >
              <GlassCard variant="gradient-border" hover className="h-full">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-16 h-16 rounded-2xl bg-insta-gradient flex items-center justify-center">
                    <Sparkle className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold mb-1">GRWM Analysis</h3>
                    <p className="text-sm text-gray-400">For lifestyle & aesthetic content</p>
                  </div>
                </div>
                <ul className="space-y-3 text-gray-300">
                  <li className="flex items-start gap-2">
                    <Sparkle className="w-5 h-5 text-magenta flex-shrink-0 mt-0.5" />
                    <span>Hook timing & intro pacing</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Sparkle className="w-5 h-5 text-magenta flex-shrink-0 mt-0.5" />
                    <span>Motion & pace consistency</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Sparkle className="w-5 h-5 text-magenta flex-shrink-0 mt-0.5" />
                    <span>Aesthetic consistency</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Sparkle className="w-5 h-5 text-magenta flex-shrink-0 mt-0.5" />
                    <span>Caption length, tone & hashtag diversity</span>
                  </li>
                </ul>
              </GlassCard>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <GlassCard variant="gradient-border" className="text-center">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Ready to <span className="text-gradient">optimize</span> your content?
            </h2>
            <p className="text-gray-300 mb-8">
              Get instant insights tailored to Daily Content Creators
            </p>
            <Link to="/analyze">
              <GradientButton size="lg">
                Start Analyzing
                <ArrowRight className="w-5 h-5" />
              </GradientButton>
            </Link>
          </GlassCard>
        </div>
      </section>

      <Footer />
    </div>
  )
}
