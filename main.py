import os
import sys
import tempfile
import logging
import streamlit as st
import pandas as pd
import altair as alt
import json
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from video_processor import VideoProcessor
from dialect_classifier import DialectClassifier
import utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="English Dialect Classifier",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def load_classifier():
    """Load the classifier model with caching."""
    try:
        return DialectClassifier(
            whisper_model_size="tiny",
            dialect_model_name="xlm-roberta-base"
        )
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None


@st.cache_resource
def get_video_processor():
    """Get the video processor with caching."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    return VideoProcessor(cache_dir=cache_dir)


def display_results(results: Dict[str, Any], video_title: str):
    """
    Display analysis results in the UI.

    Args:
        results: The analysis results
        video_title: Title of the analyzed video
    """
    # Display the top dialect
    st.header("Analysis Results")
    st.subheader(f"Video: {video_title}")

    # Create columns for layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Display top dialect with confidence
        top_dialect = results.get("top_dialect", "Unknown")
        top_confidence = results["confidence_scores"].get(top_dialect, 0.0)
        st.markdown(f"### Top Dialect: **{top_dialect}**")
        st.progress(float(top_confidence))
        st.text(f"Confidence: {top_confidence:.4f}")

        # Display transcription if available
        if "transcription" in results:
            with st.expander("View Transcription", expanded=False):
                st.text_area("", results["transcription"], height=200)

    with col2:
        # Create bar chart for confidence scores
        st.write("### Confidence Scores")
        dialect_data = pd.DataFrame({
            "Dialect": list(results["confidence_scores"].keys()),
            "Confidence": list(results["confidence_scores"].values())
        })

        # Sort by confidence
        dialect_data = dialect_data.sort_values("Confidence", ascending=False)

        # Create chart
        chart = alt.Chart(dialect_data).mark_bar().encode(
            x=alt.X("Confidence:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Dialect:N", sort="-x"),
            color=alt.Color("Dialect:N", legend=None),
            tooltip=["Dialect", "Confidence"]
        ).properties(height=200)

        st.altair_chart(chart, use_container_width=True)

    # Display segment analysis if available
    if "segments" in results and len(results["segments"]) > 0:
        st.subheader("Segment Analysis")

        # Create a dataframe for the segments
        segments_data = []
        for segment in results["segments"]:
            start_time = utils.format_time(segment["start_time"])
            end_time = utils.format_time(segment["end_time"])

            # Get top dialect for this segment
            segment_dialect_confs = segment["dialect_confidences"]
            top_dialect = max(segment_dialect_confs.items(), key=lambda x: x[1])

            segments_data.append({
                "Start": start_time,
                "End": end_time,
                "Text": segment["text"],
                "Dialect": top_dialect[0],
                "Confidence": f"{top_dialect[1]:.4f}"
            })

        df = pd.DataFrame(segments_data)

        # Create timeline visualization
        timeline_data = utils.create_visualization_data(results)["timeline"]
        timeline_df = pd.DataFrame([
            {
                "Start": seg["start"],
                "End": seg["end"],
                "Dialect": seg["dialect"],
                "Text": seg["text"]
            } for seg in timeline_data
        ])

        if not timeline_df.empty:
            # Create a Gantt chart for dialect segments
            timeline_chart = alt.Chart(timeline_df).mark_bar().encode(
                x='Start:Q',
                x2='End:Q',
                y=alt.Y('Dialect:N', axis=alt.Axis(labelAngle=0)),
                color='Dialect:N',
                tooltip=['Start', 'End', 'Dialect', 'Text']
            ).properties(
                width=700,
                height=200,
                title='Dialect Timeline'
            )

            st.altair_chart(timeline_chart, use_container_width=True)

        # Show table of segments
        with st.expander("View Segment Details", expanded=False):
            st.dataframe(df)



def process_video(video_path_or_url: str, is_url: bool = True) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Process a video file or URL and get dialect analysis.

    Args:
        video_path_or_url: Path to video file or URL
        is_url: Whether the input is a URL or file path

    Returns:
        Tuple of (results, video_title) or (None, None) on error
    """
    try:
        video_processor = get_video_processor()
        classifier = load_classifier()

        if classifier is None:
            st.error("Failed to initialize the classifier.")
            return None, None

        # Process video URL or file
        if is_url:
            audio_path, video_title = video_processor.process_url(video_path_or_url)
        else:
            audio_path, video_title = video_processor.process_file(video_path_or_url)

        # Analyze dialect
        use_segments = st.session_state.get("use_segments", False)
        results = classifier.analyze_video(audio_path, segment_analysis=use_segments)

        return results, video_title

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        logger.error(f"Error processing video: {str(e)}")
        return None, None


def main():
    """Main function for the Streamlit application."""
    st.title("English Dialect Classifier")
    st.write("Analyze English dialects in videos using AI.")

    # Sidebar for options
    st.sidebar.title("Options")

    # Segment analysis toggle
    if "use_segments" not in st.session_state:
        st.session_state.use_segments = False

    st.session_state.use_segments = st.sidebar.checkbox(
        "Enable Segment Analysis",
        value=st.session_state.use_segments,
        help="Analyze dialect changes throughout the video (slower but more detailed)"
    )

    # About section
    with st.sidebar.expander("About", expanded=False):
        st.write("""
        This app identifies English dialects in videos using machine learning.

        Supported dialects:
        - American
        - British 
        - Australian
        - Canadian
        - Indian
        - South African

        Technologies:
        - yt-dlp for video processing
        - Whisper ASR for transcription
        - Transformers for dialect classification
        """)

    # URL input section (no tabs needed now)
    st.subheader("Enter Video URL")
    url_input = st.text_input("Video URL (YouTube, direct links, etc.)", key="url_input")

    # URL validation
    if url_input:
        if not utils.is_valid_url(url_input):
            st.error("Invalid URL format. Please enter a valid URL.")
        elif not utils.is_video_url(url_input):
            st.warning("URL might not point to a video. Processing may fail.")

    # Process button for URL
    if st.button("Process URL", disabled=not url_input, key="process_url_btn"):
        if url_input:
            with st.spinner("Processing video..."):
                results, video_title = process_video(url_input, is_url=True)

            if results and video_title:
                display_results(results, video_title)


if __name__ == "__main__":
    main()
