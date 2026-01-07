"""
Audio utilities for format conversion and processing.
Handles WebM to WAV conversion for browser audio.
"""

import io
import os
import tempfile
import subprocess
from typing import Tuple, Optional
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def convert_webm_to_wav(webm_bytes: bytes) -> Optional[bytes]:
    """Convert WebM audio bytes to WAV format."""
    
    # Try pydub first (requires ffmpeg)
    if PYDUB_AVAILABLE:
        try:
            audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            return wav_buffer.read()
        except Exception as e:
            print(f"[AudioUtils] pydub conversion failed: {e}")
    
    # Fallback: use ffmpeg directly
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_bytes)
            webm_path = webm_file.name
        
        wav_path = webm_path.replace('.webm', '.wav')
        
        # Run ffmpeg
        result = subprocess.run([
            'ffmpeg', '-y', '-i', webm_path,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            wav_path
        ], capture_output=True, timeout=30)
        
        if os.path.exists(wav_path):
            with open(wav_path, 'rb') as f:
                wav_bytes = f.read()
            os.remove(wav_path)
            os.remove(webm_path)
            return wav_bytes
        
    except Exception as e:
        print(f"[AudioUtils] ffmpeg conversion failed: {e}")
    
    # Last resort: return original bytes and hope soundfile can handle it
    return webm_bytes


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """Load audio from bytes, converting format if needed."""
    
    # Check if it's WebM (starts with specific bytes)
    if audio_bytes[:4] == b'\x1a\x45\xdf\xa3' or b'webm' in audio_bytes[:50].lower():
        audio_bytes = convert_webm_to_wav(audio_bytes)
        if audio_bytes is None:
            return None, 0
    
    if not SOUNDFILE_AVAILABLE:
        return None, 0
    
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        return audio, sr
        
    except Exception as e:
        print(f"[AudioUtils] Failed to load audio: {e}")
        return None, 0
