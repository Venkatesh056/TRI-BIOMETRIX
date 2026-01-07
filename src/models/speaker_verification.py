"""
Speaker Verification Module
- Audio preprocessing
- Speaker embedding extraction (MFCC or ECAPA-TDNN)
- Speaker verification via cosine similarity
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from sklearn.metrics.pairwise import cosine_similarity

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[SpeakerModel] librosa not available, using basic audio processing")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class SpeakerVerificationModel:
    """Speaker verification using MFCC features."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.n_mfcc = self.config.get('n_mfcc', 13)
        self.threshold = 0.7
        
        # User embeddings storage
        self.user_embeddings: Dict[str, np.ndarray] = {}
        
        print(f"[SpeakerModel] Initialized with MFCC ({self.n_mfcc} coefficients)")
    
    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio file and return waveform."""
        try:
            if SOUNDFILE_AVAILABLE:
                audio, sr = sf.read(audio_path)
            elif LIBROSA_AVAILABLE:
                audio, sr = librosa.load(audio_path, sr=None)
            else:
                return None, 0
            
            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            return audio, sr
        except Exception as e:
            print(f"[SpeakerModel] Error loading audio: {e}")
            return None, 0
    
    def load_audio_bytes(self, audio_bytes: bytes) -> Tuple[Optional[np.ndarray], int]:
        """Load audio from bytes, handling WebM conversion."""
        try:
            # Import audio utilities for format conversion
            from utils.audio import load_audio_bytes as load_audio_util
            return load_audio_util(audio_bytes, self.sample_rate)
        except ImportError:
            # Fallback to direct loading
            import io
            try:
                if SOUNDFILE_AVAILABLE:
                    audio, sr = sf.read(io.BytesIO(audio_bytes))
                else:
                    return None, 0
                
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                return audio, sr
            except Exception as e:
                print(f"[SpeakerModel] Error loading audio bytes: {e}")
                return None, 0
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio: resample, normalize, trim silence."""
        # Resample if needed
        if LIBROSA_AVAILABLE and sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Trim silence
        if LIBROSA_AVAILABLE:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pre-emphasis filter
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        return audio
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio."""
        if not LIBROSA_AVAILABLE:
            # Fallback: simple statistical features
            return self._extract_simple_features(audio)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=160
        )
        
        # Delta and delta-delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Pool statistics across time
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])
        
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _extract_simple_features(self, audio: np.ndarray) -> np.ndarray:
        """Simple feature extraction without librosa."""
        # Basic statistical features
        features = [
            np.mean(audio),
            np.std(audio),
            np.max(audio),
            np.min(audio),
            np.median(audio)
        ]
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        features.append(zcr)
        
        # Energy
        energy = np.sum(audio ** 2) / len(audio)
        features.append(energy)
        
        # Pad to standard size
        features = np.array(features)
        if len(features) < 52:
            features = np.pad(features, (0, 52 - len(features)))
        
        return features[:52]
    
    def get_speaker_embedding(self, audio_path: str = None, audio_bytes: bytes = None) -> Tuple[Optional[np.ndarray], str]:
        """Extract speaker embedding from audio file or bytes."""
        if audio_bytes:
            audio, sr = self.load_audio_bytes(audio_bytes)
        elif audio_path:
            audio, sr = self.load_audio(audio_path)
        else:
            return None, "No audio provided"
        
        if audio is None:
            return None, "Failed to load audio"
        
        if len(audio) < 1000:
            return None, "Audio too short"
        
        audio = self.preprocess_audio(audio, sr)
        embedding = self.extract_embedding(audio)
        
        return embedding, "ok"
    
    def enroll_user(self, user_id: str, audio_samples: List[str] = None, 
                    audio_bytes_list: List[bytes] = None) -> Tuple[bool, str]:
        """Enroll user with multiple voice samples."""
        embeddings = []
        samples = audio_bytes_list or audio_samples or []
        
        for i, sample in enumerate(samples):
            if isinstance(sample, bytes):
                embedding, msg = self.get_speaker_embedding(audio_bytes=sample)
            else:
                embedding, msg = self.get_speaker_embedding(audio_path=sample)
            
            if embedding is None:
                return False, f"Failed on sample {i+1}: {msg}"
            embeddings.append(embedding)
        
        if not embeddings:
            return False, "No valid audio samples"
        
        # Calculate mean embedding
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-7)
        
        self.user_embeddings[user_id] = mean_embedding
        return True, f"Enrolled {len(embeddings)} voice samples"
    
    def verify(self, user_id: str, audio_path: str = None, audio_bytes: bytes = None,
               threshold: float = None) -> Tuple[bool, float, str]:
        """Verify if voice matches enrolled user."""
        threshold = threshold or self.threshold
        
        if user_id not in self.user_embeddings:
            return False, 0.0, f"No voice enrollment found for '{user_id}'"
        
        embedding, msg = self.get_speaker_embedding(audio_path=audio_path, audio_bytes=audio_bytes)
        if embedding is None:
            return False, 0.0, msg
        
        enrolled = self.user_embeddings[user_id]
        
        # Ensure same dimensions
        min_len = min(len(embedding), len(enrolled))
        embedding = embedding[:min_len]
        enrolled = enrolled[:min_len]
        
        similarity = cosine_similarity([embedding], [enrolled])[0][0]
        
        if similarity >= threshold:
            return True, similarity, f"Voice match: {similarity:.1%}"
        else:
            return False, similarity, f"Voice mismatch: {similarity:.1%} (threshold: {threshold:.1%})"
    
    def identify(self, audio_path: str = None, audio_bytes: bytes = None,
                 threshold: float = None) -> Tuple[Optional[str], float, str]:
        """Identify speaker from voice (1:N matching)."""
        threshold = threshold or self.threshold
        
        if not self.user_embeddings:
            return None, 0.0, "No enrolled users"
        
        embedding, msg = self.get_speaker_embedding(audio_path=audio_path, audio_bytes=audio_bytes)
        if embedding is None:
            return None, 0.0, msg
        
        best_match = None
        best_score = 0.0
        
        for user_id, enrolled in self.user_embeddings.items():
            min_len = min(len(embedding), len(enrolled))
            sim = cosine_similarity([embedding[:min_len]], [enrolled[:min_len]])[0][0]
            if sim > best_score:
                best_score = sim
                best_match = user_id
        
        if best_score >= threshold:
            return best_match, best_score, f"Identified as {best_match}: {best_score:.1%}"
        else:
            return None, best_score, f"No match found (best: {best_score:.1%})"
    
    def load_embeddings(self, embeddings_dict: Dict[str, np.ndarray]):
        """Load user embeddings from database."""
        self.user_embeddings = embeddings_dict
        print(f"[SpeakerModel] Loaded {len(embeddings_dict)} user embeddings")
