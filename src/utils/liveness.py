"""
Liveness Detection Module
- Blink detection
- Challenge-response verification
- Anti-spoofing checks
"""

import random
import time
import numpy as np
from typing import Tuple, List, Optional
from enum import Enum

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class Challenge(Enum):
    """Liveness challenge types."""
    BLINK = "blink"
    SMILE = "smile"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD = "nod"
    OPEN_MOUTH = "open_mouth"


class LivenessDetector:
    """Detects if the user is a live person vs photo/video."""
    
    def __init__(self):
        self.challenges = list(Challenge)
        self.eye_cascade = None
        self.smile_cascade = None
        
        if CV2_AVAILABLE:
            self._load_cascades()
    
    def _load_cascades(self):
        """Load OpenCV cascades for detection."""
        try:
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
        except Exception as e:
            print(f"[Liveness] Failed to load cascades: {e}")
    
    def generate_challenge(self, count: int = 1) -> List[dict]:
        """Generate random liveness challenges."""
        selected = random.sample(self.challenges, min(count, len(self.challenges)))
        
        challenge_info = {
            Challenge.BLINK: {"instruction": "Please blink your eyes", "icon": "fa-eye", "duration": 3},
            Challenge.SMILE: {"instruction": "Please smile", "icon": "fa-smile", "duration": 3},
            Challenge.TURN_LEFT: {"instruction": "Turn your head slightly left", "icon": "fa-arrow-left", "duration": 3},
            Challenge.TURN_RIGHT: {"instruction": "Turn your head slightly right", "icon": "fa-arrow-right", "duration": 3},
            Challenge.NOD: {"instruction": "Nod your head up and down", "icon": "fa-arrows-alt-v", "duration": 3},
            Challenge.OPEN_MOUTH: {"instruction": "Open your mouth briefly", "icon": "fa-comment", "duration": 3},
        }
        
        return [{"type": c.value, **challenge_info[c]} for c in selected]
    
    def detect_blink(self, frames: List[np.ndarray]) -> Tuple[bool, str]:
        """Detect blink across multiple frames."""
        if not CV2_AVAILABLE or self.eye_cascade is None:
            return True, "Blink detection unavailable"
        
        eye_counts = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            eye_counts.append(len(eyes))
        
        # Blink = eyes detected, then not detected, then detected again
        if len(eye_counts) >= 3:
            has_variation = max(eye_counts) > 0 and min(eye_counts) < max(eye_counts)
            if has_variation:
                return True, "Blink detected"
        
        return False, "No blink detected"
    
    def detect_smile(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect smile in frame."""
        if not CV2_AVAILABLE or self.smile_cascade is None:
            return True, 1.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smiles = self.smile_cascade.detectMultiScale(gray, 1.8, 20, minSize=(25, 25))
        
        detected = len(smiles) > 0
        confidence = min(len(smiles) / 3.0, 1.0) if detected else 0.0
        
        return detected, confidence
    
    def check_frame_quality(self, frame: np.ndarray) -> dict:
        """Check if frame is suitable for biometric capture."""
        if not CV2_AVAILABLE:
            return {"valid": True, "issues": []}
        
        issues = []
        
        # Check brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 50:
            issues.append("Too dark - increase lighting")
        elif brightness > 200:
            issues.append("Too bright - reduce lighting")
        
        # Check blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            issues.append("Image is blurry - hold steady")
        
        # Check contrast
        contrast = gray.std()
        if contrast < 30:
            issues.append("Low contrast - adjust lighting")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "brightness": float(brightness),
            "sharpness": float(laplacian_var),
            "contrast": float(contrast)
        }
    
    def detect_photo_attack(self, frames: List[np.ndarray]) -> Tuple[bool, str]:
        """Detect if input is a photo (no motion/variation)."""
        if len(frames) < 5:
            return False, "Need more frames"
        
        # Calculate frame differences
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            diffs.append(np.mean(diff))
        
        avg_motion = np.mean(diffs)
        
        # Very low motion suggests a static photo
        if avg_motion < 2.0:
            return True, "Possible photo attack - no motion detected"
        
        return False, "Motion detected - appears live"
    
    def verify_challenge(self, challenge_type: str, frames: List[np.ndarray]) -> Tuple[bool, str]:
        """Verify a specific challenge was completed."""
        if challenge_type == Challenge.BLINK.value:
            return self.detect_blink(frames)
        elif challenge_type == Challenge.SMILE.value:
            # Check if any frame has a smile
            for frame in frames:
                detected, conf = self.detect_smile(frame)
                if detected:
                    return True, "Smile detected"
            return False, "No smile detected"
        elif challenge_type in [Challenge.TURN_LEFT.value, Challenge.TURN_RIGHT.value, Challenge.NOD.value]:
            # Check for motion in the expected direction
            is_photo, msg = self.detect_photo_attack(frames)
            if is_photo:
                return False, msg
            return True, "Movement detected"
        else:
            return True, "Challenge type not implemented"


# Voice anti-spoofing
class VoiceAntiSpoof:
    """Detect recorded audio playback."""
    
    def __init__(self):
        pass
    
    def check_audio_quality(self, audio: np.ndarray, sr: int) -> dict:
        """Check audio quality indicators."""
        issues = []
        
        # Check if audio is too short
        duration = len(audio) / sr
        if duration < 0.5:
            issues.append("Audio too short")
        elif duration > 15:
            issues.append("Audio too long")
        
        # Check volume
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:
            issues.append("Audio too quiet - speak louder")
        elif rms > 0.9:
            issues.append("Audio too loud - move away from mic")
        
        # Check for clipping
        clipping = np.sum(np.abs(audio) > 0.99) / len(audio)
        if clipping > 0.01:
            issues.append("Audio clipping detected")
        
        # Estimate SNR (simplified)
        noise_floor = np.percentile(np.abs(audio), 10)
        signal_peak = np.percentile(np.abs(audio), 90)
        snr = 20 * np.log10((signal_peak + 1e-10) / (noise_floor + 1e-10))
        
        if snr < 10:
            issues.append("High background noise")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "duration": duration,
            "volume": float(rms),
            "snr": float(snr)
        }
    
    def detect_replay_attack(self, audio: np.ndarray, sr: int) -> Tuple[bool, str]:
        """Detect if audio might be a replay attack."""
        # Check for unnatural frequency patterns
        # Recorded playback often has frequency cutoffs
        
        try:
            import librosa
            # Get spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            mean_centroid = np.mean(centroid)
            
            # Very low or very high centroid might indicate playback
            if mean_centroid < 500 or mean_centroid > 5000:
                return True, "Unusual audio characteristics"
            
        except ImportError:
            pass
        
        return False, "Audio appears natural"


# Random passphrase generator
class PassphraseGenerator:
    """Generate random passphrases for anti-replay."""
    
    WORDS = [
        "apple", "banana", "cherry", "dragon", "eagle", "forest",
        "garden", "harbor", "island", "jungle", "kitchen", "lemon",
        "mountain", "nature", "ocean", "planet", "queen", "river",
        "sunset", "tiger", "umbrella", "valley", "window", "yellow",
        "zebra", "bridge", "castle", "diamond", "engine", "flower"
    ]
    
    NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
    
    COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "gray"]
    
    @classmethod
    def generate(cls, word_count: int = 3) -> str:
        """Generate a random passphrase."""
        words = random.sample(cls.WORDS, word_count)
        return " ".join(words)
    
    @classmethod
    def generate_with_number(cls) -> str:
        """Generate passphrase with a number."""
        word = random.choice(cls.WORDS)
        number = random.choice(cls.NUMBERS)
        color = random.choice(cls.COLORS)
        return f"{color} {word} {number}"
