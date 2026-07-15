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
        # Use None for auto-detect, or specific language code
        self.language = self.config.get('language', None)  # Auto-detect by default
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
    
    def transcribe(self, audio_path: str = None, audio_bytes: bytes = None) -> Tuple[str, float, str]:
        """Transcribe audio to text. Returns (text, confidence, detected_language)."""
        if self.model is None:
            return "", 0.0, "unknown"
        
        try:
            import tempfile
            import numpy as np
            temp_path = None
            audio_array = None
            
            # Save bytes to temp file if needed
            if audio_bytes:
                # Check if WebM and convert
                if audio_bytes[:4] == b'\x1a\x45\xdf\xa3' or b'webm' in audio_bytes[:100].lower():
                    try:
                        from utils.audio import convert_webm_to_wav
                        converted = convert_webm_to_wav(audio_bytes)
                        if converted:
                            audio_bytes = converted
                            print("[SpeechModel] WebM converted to WAV successfully")
                        else:
                            print("[SpeechModel] WebM conversion returned None")
                            return "", 0.0, "unknown"
                    except Exception as e:
                        print(f"[SpeechModel] WebM conversion failed: {e}")
                        return "", 0.0, "unknown"
                
                # Load audio as numpy array using soundfile (bypasses Whisper's ffmpeg dependency)
                try:
                    import soundfile as sf
                    import io
                    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                    
                    # Convert to mono if stereo
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # Resample to 16kHz if needed (Whisper expects 16kHz)
                    if sr != 16000:
                        # Simple resampling
                        duration = len(audio_array) / sr
                        new_length = int(duration * 16000)
                        audio_array = np.interp(
                            np.linspace(0, len(audio_array), new_length),
                            np.arange(len(audio_array)),
                            audio_array
                        )
                    
                    # Ensure float32
                    audio_array = audio_array.astype(np.float32)
                    
                    print(f"[SpeechModel] Audio loaded: {len(audio_array)} samples, {len(audio_array)/16000:.2f}s")
                    
                except Exception as e:
                    print(f"[SpeechModel] Failed to load audio with soundfile: {e}")
                    # Fallback: write to temp file
                    suffix = '.wav' if audio_bytes[:4] == b'RIFF' else '.webm'
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                        f.write(audio_bytes)
                        temp_path = f.name
                    audio_path = temp_path
            
            # Transcribe
            if audio_array is not None:
                # Pass numpy array directly to Whisper
                transcribe_opts = {'fp16': False}
                if self.language:
                    transcribe_opts['language'] = self.language
                
                result = self.model.transcribe(audio_array, **transcribe_opts)
            elif audio_path:
                # Fallback to file path
                transcribe_opts = {'fp16': False}
                if self.language:
                    transcribe_opts['language'] = self.language
                
                result = self.model.transcribe(audio_path, **transcribe_opts)
            else:
                return "", 0.0, "unknown"
            
            text = result.get('text', '').strip()
            confidence = 1.0 - result.get('no_speech_prob', 0.0)
            detected_lang = result.get('language', 'unknown')
            
            print(f"[SpeechModel] Transcribed: '{text}' (lang: {detected_lang}, conf: {confidence:.2f})")
            
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            
            return text, confidence, detected_lang
            
        except Exception as e:
            print(f"[SpeechModel] Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0, "unknown"
    
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
        transcribed, confidence, detected_lang = self.transcribe(audio_path=audio_path, audio_bytes=audio_bytes)
        
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
