import os
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DialectClassifier:
    """
    Handles audio transcription and dialect classification.
    """
    
    # List of supported English dialects
    SUPPORTED_DIALECTS = ["American", "British", "Australian", "Canadian", "Indian", "South African"]
    
    # Dialect-specific keywords and patterns
    DIALECT_PATTERNS = {
        "American": [
            r"\b(apartment|elevator|trash|sidewalk|truck|vacation)\b",
            r"\b(center|color|favorite|honor|labor|theater)\b",
            r"\b(realize|analyze|civilize|memorize)\b",
            r"\b(program|catalog|dialog)\b"
        ],
        "British": [
            r"\b(flat|lift|rubbish|pavement|lorry|holiday)\b",
            r"\b(centre|colour|favourite|honour|labour|theatre)\b",
            r"\b(realise|analyse|civilise|memorise)\b",
            r"\b(programme|catalogue|dialogue)\b"
        ],
        "Australian": [
            r"\b(arvo|barbie|brekkie|chook|drongo|fair dinkum)\b",
            r"\b(mozzie|servo|stubby|thongs|ute)\b",
            r"\b(no worries|good on ya|she'll be right)\b"
        ],
        "Canadian": [
            r"\b(loonie|toonie|double-double|eh|keener|poutine)\b",
            r"\b(serviette|chesterfield|hydro|washroom)\b",
            r"\b(aboot|hoose|oot|roof)\b"
        ],
        "Indian": [
            r"\b(please do the needful|kindly revert|prepone|updation)\b",
            r"\b(do one thing|out of station|pass out|timepass)\b",
            r"\b(what is your good name|I am having|I am understanding)\b"
        ],
        "South African": [
            r"\b(lekker|braai|bakkie|robot|just now|now now)\b",
            r"\b(howzit|sharp|eish|ja|ne)\b",
            r"\b(veld|kloof|kopje|spruit)\b"
        ]
    }
    
    def __init__(self, 
                 whisper_model_size: str = "large",
                 dialect_model_name: str = "distilbert-base-uncased",
                 device: Optional[str] = None):
        """
        Initialize the DialectClassifier.
        
        Args:
            whisper_model_size: Size of the Whisper model to use ("tiny", "base", "small", "medium", "large")
            dialect_model_name: Name of the pre-trained model for dialect classification
            device: Device to run models on ("cuda" or "cpu"). If None, uses CUDA if available.
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model for transcription
        try:
            logger.info(f"Loading Whisper model ({whisper_model_size})...")
            self.whisper_model = whisper.load_model(whisper_model_size).to(self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")
        
        # Load dialect classification model
        try:
            logger.info(f"Loading dialect classification model ({dialect_model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(dialect_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                dialect_model_name, 
                num_labels=len(self.SUPPORTED_DIALECTS)
            ).to(self.device)
            
            # Initialize zero-shot classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Dialect classification model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dialect classification model: {str(e)}")
            raise RuntimeError(f"Failed to load dialect classification model: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better classification.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_dialect_features(self, text: str) -> Dict[str, float]:
        """
        Extract dialect-specific features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of dialect scores based on patterns
        """
        scores = {dialect: 0.0 for dialect in self.SUPPORTED_DIALECTS}
        total_matches = 0
        
        for dialect, patterns in self.DIALECT_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            scores[dialect] = matches
            total_matches += matches
        
        # Normalize scores
        if total_matches > 0:
            for dialect in scores:
                scores[dialect] /= total_matches
        
        return scores
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"]
            logger.info(f"Transcription completed: {len(transcription)} characters")
            return transcription
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}")
    
    def classify_text(self, text: str) -> Dict[str, float]:
        """
        Classify the dialect of the given text using multiple methods.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary of dialect confidences {dialect_name: confidence_score}
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        logger.info("Classifying dialect...")
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Get pattern-based scores
            pattern_scores = self.extract_dialect_features(processed_text)
            
            # Get transformer-based scores
            inputs = self.tokenizer(processed_text, truncation=True, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                transformer_scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Get zero-shot classification scores
            zero_shot_results = self.classifier(
                processed_text,
                candidate_labels=self.SUPPORTED_DIALECTS,
                multi_label=False
            )
            zero_shot_scores = dict(zip(zero_shot_results['labels'], zero_shot_results['scores']))
            
            # Combine scores with weights
            final_scores = {}
            for dialect in self.SUPPORTED_DIALECTS:
                final_scores[dialect] = (
                    0.2* pattern_scores[dialect] +
                    0.1 * transformer_scores[self.SUPPORTED_DIALECTS.index(dialect)] +
                    0.7 * zero_shot_scores[dialect]
                )
            
            # Normalize final scores
            total_score = sum(final_scores.values())
            if total_score > 0:
                final_scores = {k: v/total_score for k, v in final_scores.items()}
            
            logger.info(f"Classification completed with top result: {max(final_scores.items(), key=lambda x: x[1])}")
            return final_scores
            
        except Exception as e:
            logger.error(f"Failed to classify text: {str(e)}")
            raise RuntimeError(f"Failed to classify text: {str(e)}")
    
    def process_audio_segments(self, audio_path: str, segment_duration: int = 30) -> List[Dict]:
        """
        Process audio in segments for timestamp-based dialect analysis.
        
        Args:
            audio_path: Path to the audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of dictionaries with segment info: {start_time, end_time, text, dialect_confidences}
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Processing audio segments from: {audio_path}")
        
        try:
            # Use Whisper to transcribe with segment timestamps
            result = self.whisper_model.transcribe(audio_path)
            segments = result["segments"]
            
            output_segments = []
            for segment in segments:
                # Get text and timestamps
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Classify dialect for this segment
                if len(text.split()) >= 5:  # Only classify if we have enough words
                    dialect_confidences = self.classify_text(text)
                else:
                    dialect_confidences = {dialect: 1.0/len(self.SUPPORTED_DIALECTS) for dialect in self.SUPPORTED_DIALECTS}
                
                output_segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text,
                    "dialect_confidences": dialect_confidences
                })
            
            logger.info(f"Processed {len(output_segments)} audio segments")
            return output_segments
            
        except Exception as e:
            logger.error(f"Failed to process audio segments: {str(e)}")
            raise RuntimeError(f"Failed to process audio segments: {str(e)}")
    
    def analyze_video(self, audio_path: str, segment_analysis: bool = False) -> Dict:
        """
        Analyze the dialect in a video by transcribing audio and classifying the text.
        
        Args:
            audio_path: Path to the audio file extracted from video
            segment_analysis: Whether to perform segment-by-segment analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if segment_analysis:
                # Process segments for timestamp-based analysis
                segments = self.process_audio_segments(audio_path)
                
                # Aggregate results
                all_confidences = []
                for segment in segments:
                    all_confidences.append(list(segment["dialect_confidences"].values()))
                
                if all_confidences:
                    # Calculate average confidence across all segments
                    avg_confidences = np.mean(all_confidences, axis=0)
                    overall_confidences = {dialect: float(conf) for dialect, conf in zip(self.SUPPORTED_DIALECTS, avg_confidences)}
                else:
                    overall_confidences = {dialect: 0.0 for dialect in self.SUPPORTED_DIALECTS}
                
                # Get top dialect
                top_dialect = max(overall_confidences.items(), key=lambda x: x[1])[0]
                
                return {
                    "top_dialect": top_dialect,
                    "confidence_scores": overall_confidences,
                    "segments": segments
                }
            else:
                # Simple full-text analysis
                transcription = self.transcribe_audio(audio_path)
                dialect_confidences = self.classify_text(transcription)
                top_dialect = max(dialect_confidences.items(), key=lambda x: x[1])[0]
                
                return {
                    "transcription": transcription,
                    "top_dialect": top_dialect,
                    "confidence_scores": dialect_confidences
                }
        
        except Exception as e:
            logger.error(f"Failed to analyze video: {str(e)}")
            raise RuntimeError(f"Failed to analyze video: {str(e)}") 
