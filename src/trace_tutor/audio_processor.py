"""
Audio Processing Module

Implements speech-to-text (STT) and text-to-speech (TTS) for TRACE-Tutor.
Supports audio input for questions and audio output for answers.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import tempfile

logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Speech-to-text processor using Whisper.
    
    Transcribes audio questions to text for RAG pipeline.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize Whisper STT.
        
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
            device: 'cpu' or 'cuda'
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        
        try:
            import whisper
            logger.info(f"Loading Whisper {model_size} model...")
            self.model = whisper.load_model(model_size, device=device)
            logger.info(f"Whisper STT ready ({model_size}, {device})")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language code ('en', 'hi', 'ur') or None for auto-detect
            
        Returns:
            Dict with 'text', 'language', 'confidence'
        """
        if self.model is None:
            logger.error("Whisper model not loaded")
            return {"text": "", "language": "unknown", "confidence": 0.0}
        
        try:
            logger.info(f"Transcribing {Path(audio_path).name}...")
            
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                fp16=False  # CPU compatibility
            )
            
            transcription = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "confidence": self._estimate_confidence(result)
            }
            
            logger.info(f"Transcribed ({transcription['language']}): {transcription['text'][:100]}...")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"text": "", "language": "unknown", "confidence": 0.0}
    
    def _estimate_confidence(self, result: Dict) -> float:
        """Estimate transcription confidence from Whisper output."""
        # Whisper doesn't provide direct confidence scores
        # Use heuristics: check for repeated tokens, very short output, etc.
        text = result.get("text", "")
        
        if not text or len(text.strip()) < 5:
            return 0.0
        
        # Heuristic: longer, coherent text = higher confidence
        words = text.split()
        if len(words) < 3:
            return 0.6
        
        return 0.85  # Default reasonable confidence


class TextToSpeech:
    """
    Text-to-speech processor using gTTS.
    
    Converts answer text to audio for playback.
    """
    
    def __init__(self, default_lang: str = "en", slow: bool = False):
        """
        Initialize TTS.
        
        Args:
            default_lang: Default language ('en', 'hi', etc.)
            slow: Speak slowly (for young learners)
        """
        self.default_lang = default_lang
        self.slow = slow
        
        try:
            from gtts import gTTS
            self.gtts = gTTS
            logger.info(f"gTTS ready (lang={default_lang}, slow={slow})")
        except Exception as e:
            logger.error(f"Failed to load gTTS: {e}")
            self.gtts = None
    
    def synthesize(self, text: str, output_path: Optional[str] = None, lang: Optional[str] = None) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to speak
            output_path: Output file path (None = temp file)
            lang: Language code or None for default
            
        Returns:
            Path to generated audio file
        """
        if self.gtts is None:
            logger.error("gTTS not available")
            return ""
        
        if not text or len(text.strip()) < 2:
            logger.warning("Text too short for TTS")
            return ""
        
        try:
            # Use provided path or create temp file
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                output_path = temp_file.name
                temp_file.close()
            
            lang = lang or self.default_lang
            
            logger.info(f"Synthesizing {len(text)} chars to {Path(output_path).name}...")
            
            tts = self.gtts(text=text, lang=lang, slow=self.slow)
            tts.save(output_path)
            
            logger.info(f"TTS complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return ""
    
    def synthesize_answer(self, answer_text: str, grade: int = 3, lang: str = "en") -> str:
        """
        Synthesize answer with grade-appropriate pacing.
        
        Args:
            answer_text: Answer text to speak
            grade: Grade level (1-4) - lower grades get slower speech
            lang: Language code
            
        Returns:
            Path to audio file
        """
        # Adjust speed for younger students
        original_slow = self.slow
        self.slow = (grade <= 2)  # Grades 1-2 get slow speech
        
        audio_path = self.synthesize(answer_text, lang=lang)
        
        self.slow = original_slow  # Restore original
        return audio_path


class AudioRecorder:
    """
    Audio recording utility for dashboard integration.
    
    Records from microphone for question input.
    """
    
    @staticmethod
    def record(duration: int = 10, sample_rate: int = 16000, output_path: Optional[str] = None) -> str:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate (16kHz for Whisper)
            output_path: Output file path (None = temp file)
            
        Returns:
            Path to recorded audio file
        """
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav
            import numpy as np
            
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                output_path = temp_file.name
                temp_file.close()
            
            logger.info(f"Recording {duration}s audio...")
            
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16
            )
            sd.wait()
            
            wav.write(output_path, sample_rate, recording)
            
            logger.info(f"Recorded to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return ""


class AudioQuestionProcessor:
    """
    Complete audio question processing pipeline.
    
    Combines recording, transcription, and language detection.
    """
    
    def __init__(self, whisper_model: str = "base", device: str = "cpu"):
        """
        Initialize audio processor.
        
        Args:
            whisper_model: Whisper model size
            device: 'cpu' or 'cuda'
        """
        self.stt = SpeechToText(model_size=whisper_model, device=device)
        self.recorder = AudioRecorder()
    
    def process_audio_file(self, audio_path: str, expected_lang: Optional[str] = None) -> Dict:
        """
        Process uploaded audio file.
        
        Args:
            audio_path: Path to audio file
            expected_lang: Expected language hint
            
        Returns:
            Dict with 'text', 'language', 'confidence', 'audio_path'
        """
        result = self.stt.transcribe(audio_path, language=expected_lang)
        result["audio_path"] = audio_path
        return result
    
    def process_recording(self, duration: int = 10, expected_lang: Optional[str] = None) -> Dict:
        """
        Record and process microphone input.
        
        Args:
            duration: Recording duration in seconds
            expected_lang: Expected language hint
            
        Returns:
            Dict with 'text', 'language', 'confidence', 'audio_path'
        """
        audio_path = self.recorder.record(duration=duration)
        
        if not audio_path:
            return {"text": "", "language": "unknown", "confidence": 0.0, "audio_path": ""}
        
        return self.process_audio_file(audio_path, expected_lang=expected_lang)


def create_audio_response(answer_text: str, grade: int, language: str = "en") -> str:
    """
    Create audio version of answer.
    
    Args:
        answer_text: Answer text to speak
        grade: Grade level (affects pacing)
        language: Language code
        
    Returns:
        Path to audio file
    """
    tts = TextToSpeech(default_lang=language)
    return tts.synthesize_answer(answer_text, grade=grade, lang=language)
