import os
import tempfile
from typing import Optional, Tuple
import yt_dlp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video downloading and audio extraction from various sources."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the VideoProcessor.
        
        Args:
            cache_dir: Directory to cache downloaded videos. If None, uses a temporary directory.
        """
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
    
    def process_url(self, url: str) -> Tuple[str, str]:
        """
        Process a video URL to extract audio.
        
        Args:
            url: URL of the video to process
            
        Returns:
            Tuple containing (audio_path, video_title)
            
        Raises:
            ValueError: If the URL is invalid or video cannot be processed
        """
        if not url:
            raise ValueError("URL cannot be empty")
            
        logger.info(f"Processing URL: {url}")
        
        try:
            # Setup yt-dlp options for audio extraction
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.cache_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            # First get video info without downloading
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown Title')
                video_id = info.get('id', 'Unknown ID')
                
                # Check for cached version
                expected_path = os.path.join(self.cache_dir, f"{video_title}.wav")
                if os.path.exists(expected_path):
                    logger.info(f"Using cached audio for: {video_title}")
                    return expected_path, video_title
            
            # Download and extract audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the output file
            audio_path = os.path.join(self.cache_dir, f"{video_title}.wav")
            if not os.path.exists(audio_path):
                # Look for any .wav file if exact name doesn't match
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.wav') and file.startswith(video_title[:10]):
                        audio_path = os.path.join(self.cache_dir, file)
                        break
                else:
                    raise FileNotFoundError(f"Could not find extracted audio file for {video_title}")
            
            logger.info(f"Successfully extracted audio to: {audio_path}")
            return audio_path, video_title
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            raise ValueError(f"Failed to process video URL: {str(e)}")
    
    def process_file(self, file_path: str) -> Tuple[str, str]:
        """
        Process a local video file to extract audio.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Tuple containing (audio_path, video_title)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Get the video title from filename
            video_title = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(self.cache_dir, f"{video_title}.wav")
            
            # Check if already processed
            if os.path.exists(output_path):
                logger.info(f"Using cached audio for: {video_title}")
                return output_path, video_title
                
            # Setup yt-dlp options for audio extraction
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.cache_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            # Extract audio using yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([file_path])
                
            if not os.path.exists(output_path):
                # Look for any .wav file if exact name doesn't match
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.wav'):
                        output_path = os.path.join(self.cache_dir, file)
                        break
                else:
                    raise FileNotFoundError(f"Could not find extracted audio file for {video_title}")
            
            logger.info(f"Successfully extracted audio to: {output_path}")
            return output_path, video_title
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise ValueError(f"Failed to process video file: {str(e)}")
            
    def cleanup(self):
        """Clean up any temporary files."""
        # Only remove files if using a temporary directory
        if self.cache_dir.startswith(tempfile.gettempdir()):
            for file in os.listdir(self.cache_dir):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    logger.warning(f"Failed to remove file {file}: {str(e)}") 