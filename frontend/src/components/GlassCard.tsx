import { HTMLAttributes, ReactNode } from 'react'
import { motion } from 'framer-motion'
import { clsx } from 'clsx'

interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  variant?: 'default' | 'strong' | 'gradient-border'
  hover?: boolean
}

export default function GlassCard({
  children,
  variant = 'default',
  hover = false,
  className,
  ...props
}: GlassCardProps) {
  const variants = {
    default: 'glass',
    strong: 'glass-strong',
    'gradient-border': 'gradient-border glass',
  }
  
  const Component = hover ? motion.div : 'div'
  const motionProps = hover
    ? {
        whileHover: { scale: 1.02, y: -4 },
        transition: { duration: 0.2 },
      }
    : {}
  
  return (
    <Component
      className={clsx(
        'rounded-2xl p-6 shadow-xl',
        variants[variant],
        className
      )}
      {...motionProps}
      {...props}
    >
      {children}
    </Component>
  )
}

