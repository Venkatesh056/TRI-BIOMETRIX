"""
Speech Recognition Module
- Audio transcription using Whisper
- Passphrase content verification with fuzzy matching
"""

import os
from typing import Tuple, Optional

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[SpeechModel] Whisper not available")

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    # Fallback to simple matching


class SpeechRecognitionModel:
    """Speech-to-text transcription and passphrase verification."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model_size = self.config.get('model', 'tiny')
        self.language = self.config.get('language', 'en')
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        if not WHISPER_AVAILABLE:
            print("[SpeechModel] Whisper not installed, transcription disabled")
            return
        
        try:
            print(f"[SpeechModel] Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            print("[SpeechModel] Model loaded successfully")
        except Exception as e:
            print(f"[SpeechModel] Failed to load model: {e}")
            self.model = None
    
    def transcribe(self, audio_path: str = None, audio_bytes: bytes = None) -> Tuple[str, float]:
        """Transcribe audio to text."""
        if self.model is None:
            return "", 0.0
        
        try:
            import tempfile
            temp_path = None
            
            # Save bytes to temp file if needed
            if audio_bytes:
                # Check if WebM and convert
                if audio_bytes[:4] == b'\x1a\x45\xdf\xa3' or b'webm' in audio_bytes[:50].lower():
                    try:
                        from utils.audio import convert_webm_to_wav
                        audio_bytes = convert_webm_to_wav(audio_bytes)
                    except ImportError:
                        pass
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(audio_bytes)
                    temp_path = f.name
                audio_path = temp_path
            
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                fp16=False
            )
            
            text = result.get('text', '').strip()
            confidence = 1.0 - result.get('no_speech_prob', 0.0)
            
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            
            return text, confidence
            
        except Exception as e:
            print(f"[SpeechModel] Transcription error: {e}")
            return "", 0.0
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def verify_passphrase(self, audio_path: str = None, audio_bytes: bytes = None,
                         expected_text: str = "", threshold: float = 0.8) -> Tuple[bool, float, str, str]:
        """Verify spoken passphrase matches expected text."""
        transcribed, confidence = self.transcribe(audio_path=audio_path, audio_bytes=audio_bytes)
        
        if not transcribed:
            return False, 0.0, "", "Failed to transcribe audio"
        
        # Normalize both texts
        transcribed_norm = self.normalize_text(transcribed)
        expected_norm = self.normalize_text(expected_text)
        
        # Calculate similarity
        if RAPIDFUZZ_AVAILABLE:
            similarity = fuzz.token_sort_ratio(transcribed_norm, expected_norm) / 100.0
        else:
            # Simple word overlap
            trans_words = set(transcribed_norm.split())
            exp_words = set(expected_norm.split())
            if exp_words:
                similarity = len(trans_words & exp_words) / len(exp_words)
            else:
                similarity = 0.0
        
        if similarity >= threshold:
            return True, similarity, transcribed, f"Passphrase correct: '{transcribed}'"
        else:
            return False, similarity, transcribed, f"Expected: '{expected_text}', Got: '{transcribed}'"
    
    def is_available(self) -> bool:
        """Check if speech recognition is available."""
        return self.model is not None
