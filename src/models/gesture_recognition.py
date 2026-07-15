"""
Gesture Recognition Module
- Hand detection using MediaPipe Tasks API (0.10+)
- Gesture classification (thumbs up, wave, peace, etc.)
- Liveness detection support
"""

import os
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum

MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[GestureModel] MediaPipe not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Path to the hand landmarker model file
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'models', 'hand_landmarker.task'
)


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

        self.landmarker = None
        if MEDIAPIPE_AVAILABLE and self.enabled:
            self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe HandLandmarker (Tasks API)."""
        if not os.path.exists(_MODEL_PATH):
            print(f"[GestureModel] Model file not found: {_MODEL_PATH}")
            return
        try:
            base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
            print("[GestureModel] MediaPipe HandLandmarker initialized")
        except Exception as e:
            print(f"[GestureModel] Failed to initialize: {e}")
            self.landmarker = None

    def detect_hand(self, image: np.ndarray) -> Tuple[bool, Optional[List]]:
        """Detect hand in image and return landmarks."""
        if self.landmarker is None:
            return False, None

        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            return True, result.hand_landmarks[0]

        return False, None

    def classify_gesture(self, landmarks) -> Tuple[GestureType, float]:
        """Classify gesture from hand landmarks."""
        if landmarks is None:
            return GestureType.NONE, 0.0

        # Extract landmark positions (NormalizedLandmark objects with .x .y .z)
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Finger tip and joint indices
        # 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
        thumb_tip = points[4]
        thumb_ip  = points[3]
        index_tip = points[8]
        index_pip = points[6]
        middle_tip = points[12]
        middle_pip = points[10]
        ring_tip  = points[16]
        ring_pip  = points[14]
        pinky_tip = points[20]
        pinky_pip = points[18]
        wrist     = points[0]

        # Check if fingers are extended (tip above pip in image coords = smaller y)
        thumb_extended  = thumb_tip[0] < thumb_ip[0]  # Horizontal for thumb (right hand)
        index_extended  = index_tip[1]  < index_pip[1]
        middle_extended = middle_tip[1] < middle_pip[1]
        ring_extended   = ring_tip[1]   < ring_pip[1]
        pinky_extended  = pinky_tip[1]  < pinky_pip[1]

        fingers_extended = [thumb_extended, index_extended, middle_extended,
                            ring_extended, pinky_extended]
        num_extended = sum(fingers_extended)

        confidence = 0.8

        # Thumbs up/down: only thumb extended
        if thumb_extended and not any(fingers_extended[1:]):
            if thumb_tip[1] < wrist[1]:  # Thumb above wrist
                return GestureType.THUMBS_UP, confidence
            else:
                return GestureType.THUMBS_DOWN, confidence

        # Peace sign: index and middle extended only
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return GestureType.PEACE, confidence

        # Fist: no fingers extended
        if num_extended == 0:
            return GestureType.FIST, confidence

        # Open palm: all fingers extended
        if num_extended >= 4:
            return GestureType.OPEN_PALM, confidence

        return GestureType.NONE, 0.0

    def detect_gesture(self, image: np.ndarray) -> Tuple[str, float, str]:
        """Detect and classify gesture in image."""
        if not self.enabled:
            return "disabled", 1.0, "Gesture detection disabled"

        if self.landmarker is None:
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
        return self.landmarker is not None and self.enabled

    def get_supported_gestures(self) -> List[str]:
        """Return list of supported gestures."""
        return [g.value for g in GestureType if g != GestureType.NONE]
