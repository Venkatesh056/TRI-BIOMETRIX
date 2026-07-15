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

# Try to get ffmpeg path from imageio-ffmpeg BEFORE importing pydub
FFMPEG_PATH = None
FFPROBE_PATH = None
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    # ffprobe is in the same directory as ffmpeg
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    possible_ffprobe = os.path.join(ffmpeg_dir, 'ffprobe.exe')
    if os.path.exists(possible_ffprobe):
        FFPROBE_PATH = possible_ffprobe
    else:
        # Use ffmpeg path for ffprobe (some builds include it)
        FFPROBE_PATH = FFMPEG_PATH
except ImportError:
    pass

# Now import pydub and set paths
try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
    
    # Set converter paths before any pydub operations
    if FFMPEG_PATH:
        AudioSegment.converter = FFMPEG_PATH
        AudioSegment.ffmpeg = FFMPEG_PATH
        AudioSegment.ffprobe = FFPROBE_PATH or FFMPEG_PATH
except ImportError:
    PYDUB_AVAILABLE = False


def convert_webm_to_wav(webm_bytes: bytes) -> Optional[bytes]:
    """Convert WebM audio bytes to WAV format using ffmpeg directly."""
    
    if not FFMPEG_PATH:
        print("[AudioUtils] No ffmpeg path available")
        return None
    
    try:
        # Write webm to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_bytes)
            webm_path = webm_file.name
        
        wav_path = webm_path.replace('.webm', '.wav')
        
        # Run ffmpeg directly (bypass pydub)
        result = subprocess.run([
            FFMPEG_PATH, '-y', '-i', webm_path,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            wav_path
        ], capture_output=True, timeout=30)
        
        if result.returncode != 0:
            print(f"[AudioUtils] ffmpeg error: {result.stderr.decode()}")
            os.remove(webm_path)
            return None
        
        if os.path.exists(wav_path):
            with open(wav_path, 'rb') as f:
                wav_bytes = f.read()
            os.remove(wav_path)
            os.remove(webm_path)
            return wav_bytes
        
        os.remove(webm_path)
        return None
        
    except Exception as e:
        print(f"[AudioUtils] Conversion failed: {e}")
        return None


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
