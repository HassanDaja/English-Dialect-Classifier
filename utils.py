import os
import re
import json
import csv
import logging
import validators
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    return validators.url(url) is True

def is_video_url(url: str) -> bool:
    """
    Check if a URL points to a video (YouTube, Vimeo, etc.).
    
    Args:
        url: URL to check
        
    Returns:
        True if it's a video URL, False otherwise
    """
    # List of common video hosting domains
    video_domains = [
        'youtube.com', 'youtu.be',
        'vimeo.com',
        'dailymotion.com',
        'facebook.com/watch',
        'twitch.tv',
        'streamable.com'
    ]
    
    # Check if URL contains any of these domains
    for domain in video_domains:
        if domain in url:
            return True
    
    # Check for common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
    for ext in video_extensions:
        if url.lower().endswith(ext):
            return True
    
    return False

def is_supported_video_file(file_path: str) -> bool:
    """
    Check if a file is a supported video format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if supported, False otherwise
    """
    # List of supported video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
    
    # Check if file has a supported extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def format_time(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Limit length
    if len(sanitized) > 100:
        base, ext = os.path.splitext(sanitized)
        sanitized = base[:100-len(ext)] + ext
    
    return sanitized

def export_results(results: Dict[str, Any], format_type: str, output_dir: str, video_title: str) -> str:
    """
    Export analysis results to a file.
    
    Args:
        results: Analysis results
        format_type: Export format ("json" or "csv")
        output_dir: Directory to save the file
        video_title: Title of the video
        
    Returns:
        Path to the exported file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize video title for filename
    safe_title = sanitize_filename(video_title)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type.lower() == "json":
        # Export as JSON
        filename = f"{safe_title}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
    elif format_type.lower() == "csv":
        # Export as CSV
        filename = f"{safe_title}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data for CSV export
        data = []
        
        # Add basic info
        data.append(["Video Title", video_title])
        data.append(["Top Dialect", results.get("top_dialect", "Unknown")])
        data.append([])
        
        # Add confidence scores
        data.append(["Dialect", "Confidence Score"])
        for dialect, score in results.get("confidence_scores", {}).items():
            data.append([dialect, f"{score:.4f}"])
        
        # Add segments if available
        if "segments" in results:
            data.append([])
            data.append(["Segment Analysis"])
            data.append(["Start Time", "End Time", "Text", "Top Dialect", "Confidence"])
            
            for segment in results["segments"]:
                start = format_time(segment["start_time"])
                end = format_time(segment["end_time"])
                text = segment["text"]
                
                # Get top dialect for this segment
                segment_dialect_confs = segment["dialect_confidences"]
                top_dialect = max(segment_dialect_confs.items(), key=lambda x: x[1])
                
                data.append([
                    start, 
                    end, 
                    text, 
                    top_dialect[0], 
                    f"{top_dialect[1]:.4f}"
                ])
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
    
    logger.info(f"Results exported to {filepath}")
    return filepath

def get_dialects_color_map() -> Dict[str, str]:
    """
    Get a color map for dialects visualization.
    
    Returns:
        Dictionary mapping dialect names to colors
    """
    return {
        "American": "#FF5733",
        "British": "#3366FF",
        "Australian": "#33CC33",
        "Canadian": "#FF33FF",
        "Indian": "#FFCC00",
        "South African": "#00CCCC"
    }

def create_visualization_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create data structure for visualization of dialect analysis.
    
    Args:
        results: Analysis results
        
    Returns:
        Data structure suitable for visualization
    """
    # Get color map for dialects
    color_map = get_dialects_color_map()
    
    # Create data for overall pie chart
    overall_data = {
        "labels": list(results["confidence_scores"].keys()),
        "values": list(results["confidence_scores"].values()),
        "colors": [color_map.get(dialect, "#CCCCCC") for dialect in results["confidence_scores"].keys()]
    }
    
    # Create data for segment timeline if available
    timeline_data = []
    if "segments" in results:
        for segment in results["segments"]:
            # Get top dialect for this segment
            segment_dialect_confs = segment["dialect_confidences"]
            top_dialect = max(segment_dialect_confs.items(), key=lambda x: x[1])[0]
            
            timeline_data.append({
                "start": segment["start_time"],
                "end": segment["end_time"],
                "text": segment["text"],
                "dialect": top_dialect,
                "color": color_map.get(top_dialect, "#CCCCCC")
            })
    
    return {
        "overall": overall_data,
        "timeline": timeline_data
    } 