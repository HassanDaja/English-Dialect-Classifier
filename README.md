# English Dialect Classifier

A Python application that classifies English dialects from video URLs or uploaded files. The application can analyze videos from YouTube, direct links, or uploaded files, and identify the dialect being spoken (American, British, Australian, Canadian, Indian, South African).

## Features

- Video URL support (YouTube, direct links, etc.)
- Local video file upload support
- Audio transcription using OpenAI's Whisper
- Dialect classification using transformers
- Confidence scores for multiple dialects
- Timeline-based dialect analysis for longer videos
- Export results as JSON or CSV
- Interactive visualizations

## Architecture

The application uses the following components:

1. **Video Processing**: yt-dlp for downloading and extracting audio from videos
2. **Speech Recognition**: Whisper for transcribing audio to text
3. **Dialect Classification**: Transformer-based model for text classification
4. **User Interface**: Streamlit for web interface

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- CUDA-compatible GPU (optional, but recommended for faster processing)

### Installing FFmpeg

#### Ubuntu/Debian:
```
sudo apt update
sudo apt install ffmpeg
```

#### MacOS:
```
brew install ffmpeg
```

#### Windows:
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Add FFmpeg to your system PATH

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/dialect-classifier.git
cd dialect-classifier
```

2. Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```
streamlit run main.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Use the application:
   - Enter a YouTube URL or direct link to a video
   - OR upload a video file (supported formats: mp4, mov, avi, mkv, webm)
   - Click "Process" to analyze the video
   - View results and export them if needed

## Project Structure

```
dialect-classifier/
├── main.py              # Streamlit web application
├── dialect_classifier.py # Core ML logic
├── video_processor.py   # Video/audio handling
├── utils.py            # Helper functions
├── requirements.txt     # Dependencies
├── README.md           # Documentation
├── cache/              # Cache directory for downloaded videos
└── data/               # Directory for exported results
```

## Model Information

The dialect classifier uses a combination of:

1. **Whisper ASR**: For transcribing audio to text
2. **DistilBERT**: For dialect classification from transcribed text

The model is fine-tuned to recognize the following English dialects:
- American
- British
- Australian
- Canadian
- Indian
- South African

## Performance Considerations

- Processing large videos may take time, especially with larger Whisper models
- Using segment analysis provides more detailed results but increases processing time
- GPU acceleration is recommended for faster processing
- Video files are cached to avoid reprocessing the same content

## Troubleshooting

### Common Issues:

1. **FFmpeg errors**:
   - Ensure FFmpeg is properly installed and available in your PATH
   - Check FFmpeg installation with `ffmpeg -version`

2. **Out of memory errors**:
   - Try using a smaller Whisper model (tiny or base)
   - Process shorter video segments

3. **URL processing failures**:
   - Ensure the URL is accessible and contains valid video content
   - Some websites might block video downloads

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for NLP models
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [Streamlit](https://streamlit.io/) for the web interface 