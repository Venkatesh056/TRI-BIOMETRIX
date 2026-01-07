"""
Quality Check Module
- Face image quality validation
- Audio quality validation
- Real-time feedback
"""

import numpy as np
from typing import Tuple, Dict, List

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class FaceQualityChecker:
    """Validates face image quality before capture."""
    
    def __init__(self):
        self.face_cascade = None
        if CV2_AVAILABLE:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def check_quality(self, frame: np.ndarray) -> Dict:
        """
        Comprehensive quality check for face capture.
        Returns dict with quality metrics and issues.
        """
        result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "metrics": {},
            "face_detected": False,
            "face_centered": False,
            "face_size_ok": False
        }
        
        if not CV2_AVAILABLE:
            return result
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Brightness check
        brightness = np.mean(gray)
        result["metrics"]["brightness"] = round(brightness, 1)
        
        if brightness < 60:
            result["issues"].append("Too dark - increase lighting")
            result["valid"] = False
        elif brightness > 200:
            result["issues"].append("Too bright - reduce lighting")
            result["valid"] = False
        elif brightness < 80 or brightness > 180:
            result["warnings"].append("Lighting could be better")
        
        # 2. Contrast check
        contrast = gray.std()
        result["metrics"]["contrast"] = round(contrast, 1)
        
        if contrast < 25:
            result["issues"].append("Low contrast - adjust lighting")
            result["valid"] = False
        
        # 3. Blur/sharpness check (warning only, don't block capture for laptop webcams)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        result["metrics"]["sharpness"] = round(laplacian_var, 1)
        
        if laplacian_var < 20:
            result["warnings"].append("Image is blurry - try holding steady")
        elif laplacian_var < 50:
            result["warnings"].append("Image slightly blurry")
        
        # 4. Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        if len(faces) == 0:
            result["issues"].append("No face detected - look at camera")
            result["valid"] = False
            result["face_detected"] = False
        elif len(faces) > 1:
            result["warnings"].append("Multiple faces detected - only one person please")
            faces = [max(faces, key=lambda f: f[2] * f[3])]  # Use largest
        
        if len(faces) >= 1:
            result["face_detected"] = True
            x, y, fw, fh = faces[0]
            
            # 5. Face size check (should be 20-80% of frame)
            face_area_ratio = (fw * fh) / (w * h)
            result["metrics"]["face_size"] = round(face_area_ratio * 100, 1)
            
            if face_area_ratio < 0.08:
                result["issues"].append("Face too small - move closer")
                result["valid"] = False
            elif face_area_ratio > 0.7:
                result["issues"].append("Face too large - move back")
                result["valid"] = False
            else:
                result["face_size_ok"] = True
            
            # 6. Face centering check
            face_center_x = x + fw / 2
            face_center_y = y + fh / 2
            frame_center_x = w / 2
            frame_center_y = h / 2
            
            offset_x = abs(face_center_x - frame_center_x) / w
            offset_y = abs(face_center_y - frame_center_y) / h
            
            result["metrics"]["center_offset_x"] = round(offset_x * 100, 1)
            result["metrics"]["center_offset_y"] = round(offset_y * 100, 1)
            
            if offset_x > 0.25:
                result["warnings"].append("Center your face horizontally")
            if offset_y > 0.25:
                result["warnings"].append("Center your face vertically")
            
            if offset_x <= 0.2 and offset_y <= 0.2:
                result["face_centered"] = True
            
            # Store face bbox for UI overlay
            result["face_bbox"] = {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)}
        
        # Overall score
        score = 100
        score -= len(result["issues"]) * 25
        score -= len(result["warnings"]) * 10
        result["score"] = max(0, min(100, score))
        
        return result
    
    def get_quality_color(self, score: int) -> str:
        """Get color based on quality score."""
        if score >= 80:
            return "#10b981"  # Green
        elif score >= 50:
            return "#f59e0b"  # Yellow
        else:
            return "#ef4444"  # Red


class AudioQualityChecker:
    """Validates audio quality for voice capture."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def check_quality(self, audio: np.ndarray) -> Dict:
        """
        Check audio quality.
        Returns dict with quality metrics and issues.
        """
        result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "metrics": {}
        }
        
        # 1. Duration check
        duration = len(audio) / self.sample_rate
        result["metrics"]["duration"] = round(duration, 2)
        
        if duration < 0.5:
            result["issues"].append("Recording too short - speak longer")
            result["valid"] = False
        elif duration > 15:
            result["warnings"].append("Recording quite long")
        
        # 2. Volume check (RMS)
        rms = np.sqrt(np.mean(audio**2))
        result["metrics"]["volume"] = round(rms * 100, 1)
        
        if rms < 0.01:
            result["issues"].append("Too quiet - speak louder")
            result["valid"] = False
        elif rms < 0.03:
            result["warnings"].append("Volume is low")
        elif rms > 0.8:
            result["warnings"].append("Volume very high - move back from mic")
        
        # 3. Clipping check
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        result["metrics"]["clipping"] = round(clipping_ratio * 100, 2)
        
        if clipping_ratio > 0.05:
            result["issues"].append("Audio clipping - reduce volume")
            result["valid"] = False
        elif clipping_ratio > 0.01:
            result["warnings"].append("Some audio clipping detected")
        
        # 4. Silence ratio
        silence_threshold = 0.02
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        result["metrics"]["silence_ratio"] = round(silence_ratio * 100, 1)
        
        if silence_ratio > 0.8:
            result["issues"].append("Mostly silence - speak into microphone")
            result["valid"] = False
        
        # 5. SNR estimate
        noise_floor = np.percentile(np.abs(audio), 10)
        signal_peak = np.percentile(np.abs(audio), 90)
        snr = 20 * np.log10((signal_peak + 1e-10) / (noise_floor + 1e-10))
        result["metrics"]["snr"] = round(snr, 1)
        
        if snr < 10:
            result["issues"].append("High background noise - find quieter location")
            result["valid"] = False
        elif snr < 15:
            result["warnings"].append("Some background noise detected")
        
        # Overall score
        score = 100
        score -= len(result["issues"]) * 25
        score -= len(result["warnings"]) * 10
        result["score"] = max(0, min(100, score))
        
        return result
    
    def get_volume_level(self, audio_chunk: np.ndarray) -> float:
        """Get current volume level (0-100) for real-time meter."""
        rms = np.sqrt(np.mean(audio_chunk**2))
        # Convert to 0-100 scale with some headroom
        level = min(100, rms * 200)
        return level
