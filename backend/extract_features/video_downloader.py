"""
Video downloader module for ViralVision.

What this module does:
- Downloads videos from TikTok, YouTube Shorts, and other platforms using yt-dlp
- Handles local file paths (pass-through)
- Provides robust error handling and retry logic
- Cleans up partial downloads on failure

How it works:
1. Checks if yt-dlp is installed at runtime
2. Handles TikTok URL patterns (vt.tiktok.com redirects, www.tiktok.com/@user/video/...)
3. Forces MP4 format for compatibility
4. Retries once with fallback extractor args if initial download fails
5. Returns None on failure (never crashes) to allow pipeline to continue

Why fallback exists:
- TikTok sometimes requires specific extractor arguments for regional blocks
- Some videos need player_resolutions parameter to download successfully
- Fallback uses --extractor-args "tiktok:player_resolutions=480p" as alternative

How TikTok redirects are handled:
- yt-dlp automatically follows redirects from vt.tiktok.com to full URLs
- Both short links and full URLs are supported
- Photo posts (/photo/) are automatically skipped (blacklisted)
"""

import subprocess
import hashlib
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def _check_ytdlp_installed() -> bool:
    """
    Check if yt-dlp is installed and available.
    
    Returns:
        True if yt-dlp is available, False otherwise
    """
    return shutil.which('yt-dlp') is not None


def _generate_video_id(url: str) -> str:
    """Generate a unique video ID from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def is_local_file(path: str) -> bool:
    """
    Check if path is a local file (not a URL).
    
    Args:
        path: Path or URL string
        
    Returns:
        True if local file, False if URL
    """
    parsed = urlparse(path)
    return not parsed.scheme or parsed.scheme == 'file' or Path(path).exists()


def _is_photo_post(url: str) -> bool:
    """
    Check if URL is a TikTok photo post (not a video).
    
    Photo posts break yt-dlp, so we skip them.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL contains '/photo/', False otherwise
    """
    return '/photo/' in url.lower()


def _cleanup_partial_download(file_path: Path) -> None:
    """
    Clean up partial download file.
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass  # Ignore cleanup errors


def _download_with_ytdlp(
    url: str,
    output_path: Path,
    use_fallback: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Attempt to download video using yt-dlp.
    
    Args:
        url: Video URL
        output_path: Path to save video
        use_fallback: If True, use fallback extractor args
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    # Build base command
    cmd = [
        'yt-dlp',
        '-f', 'best[ext=mp4]/best[ext=mp4]/worst[ext=mp4]/worst',  # Prefer MP4
        '-o', str(output_path),
        '--no-warnings',
        '--no-playlist',
        '--no-check-certificate',  # Handle SSL issues
    ]
    
    # Add fallback extractor args for TikTok if needed
    if use_fallback and 'tiktok' in url.lower():
        cmd.extend(['--extractor-args', 'tiktok:player_resolutions=480p'])
    
    cmd.append(url)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, error_msg.strip()
        
        # Verify file was created and has content
        if not output_path.exists():
            return False, "File was not created"
        
        file_size = output_path.stat().st_size
        if file_size == 0:
            _cleanup_partial_download(output_path)
            return False, "Downloaded file is empty"
        
        return True, None
        
    except subprocess.TimeoutExpired:
        _cleanup_partial_download(output_path)
        return False, "Download timeout (exceeded 5 minutes)"
    except FileNotFoundError:
        return False, "yt-dlp not found. Install: pip install yt-dlp"
    except Exception as e:
        _cleanup_partial_download(output_path)
        return False, f"Download error: {str(e)}"


def download_video(
    url: str,
    output_dir: Path,
    video_id: Optional[str] = None
) -> Optional[Path]:
    """
    Download a video from URL or return local file path.
    
    This function never crashes - it returns None on failure to allow
    the pipeline to continue processing other videos.
    
    Args:
        url: Video URL (TikTok, YouTube, etc.) or local file path
        output_dir: Directory to save downloaded videos
        video_id: Optional custom video ID (auto-generated if None)
        
    Returns:
        Path to video file (.mp4) if successful, None if download failed
        
    Raises:
        FileNotFoundError: If local file doesn't exist (only for local files)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle local files
    if is_local_file(url):
        local_path = Path(url)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {url}")
        return local_path
    
    # Check if yt-dlp is installed
    if not _check_ytdlp_installed():
        print("    ERROR: yt-dlp not installed. Run: pip install yt-dlp")
        return None
    
    # Skip photo posts (they break yt-dlp)
    if _is_photo_post(url):
        print("    SKIPPED: Photo post (not a video)")
        return None
    
    # Generate video ID if not provided
    if video_id is None:
        video_id = _generate_video_id(url)
    
    output_path = output_dir / f"{video_id}.mp4"
    
    # If already downloaded, return existing file
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    
    # Try primary download method
    success, error_msg = _download_with_ytdlp(url, output_path, use_fallback=False)
    
    if success:
        return output_path
    
    # Try fallback method for TikTok videos
    if 'tiktok' in url.lower():
        print(f"    Retrying with fallback extractor...")
        success, error_msg = _download_with_ytdlp(url, output_path, use_fallback=True)
        
        if success:
            return output_path
    
    # Download failed - cleanup and return None
    _cleanup_partial_download(output_path)
    return None
