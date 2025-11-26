import { motion } from 'framer-motion'

export default function AnimatedBackground() {
  const blobs = [
    { size: 400, x: '10%', y: '20%', color: 'rgba(138, 43, 226, 0.15)' },
    { size: 300, x: '80%', y: '60%', color: 'rgba(255, 46, 196, 0.15)' },
    { size: 350, x: '50%', y: '80%', color: 'rgba(74, 0, 224, 0.15)' },
  ]

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
      {blobs.map((blob, index) => (
        <motion.div
          key={index}
          className="absolute rounded-full blur-3xl"
          style={{
            width: blob.size,
            height: blob.size,
            background: blob.color,
            left: blob.x,
            top: blob.y,
          }}
          animate={{
            x: [0, 50, 0],
            y: [0, -30, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 20 + index * 5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
      <div className="absolute inset-0 bg-gradient-to-b from-[#0D0D0F] via-[#0D0D0F] to-[#0D0D0F]" />
    </div>
  )
}

