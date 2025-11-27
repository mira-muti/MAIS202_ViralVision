import { useCallback, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, X, Video } from 'lucide-react'
import { clsx } from 'clsx'

interface UploadBoxProps {
  onFileSelect: (file: File) => void
  acceptedTypes?: string[]
  maxSizeMB?: number
}

export default function UploadBox({
  onFileSelect,
  acceptedTypes = ['video/mp4', 'video/mov', 'video/quicktime', 'video/avi'],
  maxSizeMB = 300,
}: UploadBoxProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)

  const validateFile = (file: File): string | null => {
    // Check file extension as fallback (browsers sometimes report incorrect MIME types)
    const fileExt = file.name.toLowerCase().split('.').pop()
    const validExtensions = ['mp4', 'mov', 'avi', 'quicktime']
    const hasValidExtension = validExtensions.includes(fileExt || '')
    const hasValidMimeType = acceptedTypes.includes(file.type) || file.type === ''
    
    if (!hasValidMimeType && !hasValidExtension) {
      return `File type not supported. Accepted: MP4, MOV, AVI, QuickTime`
    }
    if (file.size > maxSizeMB * 1024 * 1024) {
      return `File too large. Maximum size: ${maxSizeMB}MB`
    }
    return null
  }

  const handleFile = useCallback((file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }
    setError(null)
    setSelectedFile(file)
    onFileSelect(file)
  }, [acceptedTypes, maxSizeMB, onFileSelect])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const removeFile = useCallback(() => {
    setSelectedFile(null)
    setError(null)
  }, [])

  return (
    <div className="w-full">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={clsx(
          'relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300',
          isDragging
            ? 'border-purple bg-purple/10 scale-105'
            : 'border-white/30 hover:border-purple/50',
          selectedFile && 'border-purple bg-purple/5'
        )}
      >
        <AnimatePresence>
          {!selectedFile ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center gap-4"
            >
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ repeat: Infinity, duration: 2 }}
              >
                <Upload className="w-16 h-16 text-purple" />
              </motion.div>
              <div className="text-center">
                <p className="text-lg font-semibold mb-2">
                  Drag & drop your video here
                </p>
                <p className="text-sm text-gray-400 mb-4">or</p>
                <label className="cursor-pointer">
                  <span className="text-purple hover:text-magenta underline">
                    Browse files
                  </span>
                  <input
                    type="file"
                    accept={acceptedTypes.join(',') + ',.mov,.mp4,.avi'}
                    onChange={handleInputChange}
                    className="hidden"
                  />
                </label>
                <p className="text-xs text-gray-500 mt-4">
                  Max size: {maxSizeMB}MB • MP4, MOV, AVI, QuickTime • Analyzes first 60 seconds
                </p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="selected"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-4"
            >
              <div className="flex-shrink-0 w-16 h-16 rounded-xl bg-purple/20 flex items-center justify-center">
                <Video className="w-8 h-8 text-purple" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-semibold truncate">{selectedFile.name}</p>
                <p className="text-sm text-gray-400">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={removeFile}
                className="flex-shrink-0 w-10 h-10 rounded-full glass hover:bg-white/20 flex items-center justify-center transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {error && (
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 text-sm text-red-400"
        >
          {error}
        </motion.p>
      )}
    </div>
  )
}

