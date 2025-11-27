export interface Feature {
  feature: string
  importance: number
}

export interface PredictionResult {
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
  positives?: string[]
  improvements?: string[]

  // Legacy fields (optional, for backward compatibility)
  top_positive_features?: Feature[]
  top_negative_features?: Feature[]
  recommendations?: string[]
  raw_feature_importances?: Record<string, number>
}

export async function predictVideo(
  file: File,
  title: string,
  hashtags: string,
  niche: string
): Promise<PredictionResult> {
  const formData = new FormData()
  formData.append('video', file)
  formData.append('title', title)
  formData.append('hashtags', hashtags)
  formData.append('niche', niche)

  // Try proxy first (Vite dev), fallback to direct URL
  const apiUrl = '/api/predict'
  
  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      let errorMessage = 'Failed to predict video'
      
      try {
        const errorJson = JSON.parse(errorText)
        errorMessage = errorJson.error || errorJson.message || errorMessage
      } catch {
        errorMessage = errorText || `Server error: ${response.status} ${response.statusText}`
      }
      
      throw new Error(errorMessage)
    }

    const data = await response.json()
    return data
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('Cannot connect to API server. Make sure the backend is running on port 8000.')
    }
    throw error
  }
}

