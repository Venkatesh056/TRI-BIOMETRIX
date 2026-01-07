"""
Gesture Recognition Module
- Hand detection using MediaPipe
- Gesture classification (thumbs up, wave, peace, etc.)
- Liveness detection support
"""

import numpy as np
from typing import Tuple, Optional, List
from enum import Enum

try:
    import mediapipe as mp
    # Check if solutions attribute exists (newer versions)
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
    else:
        MEDIAPIPE_AVAILABLE = False
        print("[GestureModel] MediaPipe version incompatible")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[GestureModel] MediaPipe not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class GestureType(Enum):
    """Supported gesture types."""
    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    WAVE = "wave"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    OK = "ok"


class GestureRecognitionModel:
    """Hand gesture detection and classification."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.required_gesture = self.config.get('required_gesture', 'thumbs_up')
        
        self.hands = None
        if MEDIAPIPE_AVAILABLE and self.enabled:
            self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Hands."""
        try:
            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[GestureModel] MediaPipe Hands initialized")
        except Exception as e:
            print(f"[GestureModel] Failed to initialize: {e}")
            self.hands = None
    
    def detect_hand(self, image: np.ndarray) -> Tuple[bool, Optional[List]]:
        """Detect hand in image and return landmarks."""
        if self.hands is None:
            return False, None
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return True, landmarks
        
        return False, None
    
    def classify_gesture(self, landmarks) -> Tuple[GestureType, float]:
        """Classify gesture from hand landmarks."""
        if landmarks is None:
            return GestureType.NONE, 0.0
        
        # Extract landmark positions
        points = []
        for lm in landmarks.landmark:
            points.append([lm.x, lm.y, lm.z])
        points = np.array(points)
        
        # Finger tip and base indices
        # 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
        thumb_tip = points[4]
        thumb_ip = points[3]
        index_tip = points[8]
        index_pip = points[6]
        middle_tip = points[12]
        middle_pip = points[10]
        ring_tip = points[16]
        ring_pip = points[14]
        pinky_tip = points[20]
        pinky_pip = points[18]
        wrist = points[0]
        
        # Check if fingers are extended
        thumb_extended = thumb_tip[0] < thumb_ip[0]  # For right hand
        index_extended = index_tip[1] < index_pip[1]
        middle_extended = middle_tip[1] < middle_pip[1]
        ring_extended = ring_tip[1] < ring_pip[1]
        pinky_extended = pinky_tip[1] < pinky_pip[1]
        
        fingers_extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        num_extended = sum(fingers_extended)
        
        # Classify based on finger positions
        confidence = 0.8
        
        # Thumbs up: only thumb extended, pointing up
        if thumb_extended and not any(fingers_extended[1:]):
            if thumb_tip[1] < wrist[1]:  # Thumb above wrist
                return GestureType.THUMBS_UP, confidence
            else:
                return GestureType.THUMBS_DOWN, confidence
        
        # Peace sign: index and middle extended
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return GestureType.PEACE, confidence
        
        # Fist: no fingers extended
        if num_extended == 0:
            return GestureType.FIST, confidence
        
        # Open palm: all fingers extended
        if num_extended >= 4:
            return GestureType.OPEN_PALM, confidence
        
        # Wave: all fingers extended (same as open palm for static image)
        # In real implementation, would track motion
        
        return GestureType.NONE, 0.0
    
    def detect_gesture(self, image: np.ndarray) -> Tuple[str, float, str]:
        """Detect and classify gesture in image."""
        if not self.enabled:
            return "disabled", 1.0, "Gesture detection disabled"
        
        if self.hands is None:
            return "unavailable", 0.0, "MediaPipe not available"
        
        hand_detected, landmarks = self.detect_hand(image)
        
        if not hand_detected:
            return "none", 0.0, "No hand detected"
        
        gesture, confidence = self.classify_gesture(landmarks)
        
        return gesture.value, confidence, f"Detected: {gesture.value} ({confidence:.1%})"
    
    def verify_gesture(self, image: np.ndarray, expected: str = None) -> Tuple[bool, str, float]:
        """Verify if detected gesture matches expected."""
        expected = expected or self.required_gesture
        
        detected, confidence, msg = self.detect_gesture(image)
        
        if detected == "disabled":
            return True, "Gesture verification disabled", 1.0
        
        if detected == "unavailable":
            return False, "Gesture detection unavailable", 0.0
        
        if detected == "none":
            return False, "No gesture detected", 0.0
        
        if detected == expected:
            return True, f"Gesture verified: {detected}", confidence
        else:
            return False, f"Expected {expected}, got {detected}", confidence
    
    def is_available(self) -> bool:
        """Check if gesture recognition is available."""
        return self.hands is not None and self.enabled
    
    def get_supported_gestures(self) -> List[str]:
        """Return list of supported gestures."""
        return [g.value for g in GestureType if g != GestureType.NONE]
