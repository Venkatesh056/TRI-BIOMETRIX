"""
Face Recognition Module
- Face detection using OpenCV Haar Cascade
- Face embedding extraction (FaceNet-style)
- Face verification via cosine similarity
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognitionModel:
    """Face detection and recognition using OpenCV and embeddings."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.input_size = tuple(self.config.get('input_size', [160, 160]))
        self.embedding_size = self.config.get('embedding_size', 128)
        self.threshold = 0.6
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # User embeddings storage (loaded from database)
        self.user_embeddings: Dict[str, np.ndarray] = {}
        
        # Simple embedding model (placeholder - in production use FaceNet)
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """Initialize face embedding model."""
        # For demo: use a simple feature extractor
        # In production: load FaceNet, VGGFace, or similar
        self.embedding_model = None
        print("[FaceModel] Using simplified embedding extraction")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image, return bounding boxes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def preprocess_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop and preprocess face for embedding extraction."""
        x, y, w, h = bbox
        
        # Add margin
        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, self.input_size)
        face = face.astype(np.float32) / 255.0
        
        return face
    
    def extract_embedding(self, face: np.ndarray) -> np.ndarray:
        """Extract embedding from preprocessed face."""
        # Simplified embedding: use image statistics and histogram features
        # In production: use actual FaceNet model
        
        # Convert to grayscale for feature extraction
        if len(face.shape) == 3:
            gray = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (face * 255).astype(np.uint8)
        
        # Extract features
        features = []
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        features.extend(hist)
        
        # LBP-like features (simplified)
        resized = cv2.resize(gray, (16, 16))
        features.extend(resized.flatten() / 255.0)
        
        # Statistical features
        features.extend([
            np.mean(gray) / 255.0,
            np.std(gray) / 255.0,
            np.median(gray) / 255.0
        ])
        
        # Pad or truncate to embedding size
        embedding = np.array(features[:self.embedding_size])
        if len(embedding) < self.embedding_size:
            embedding = np.pad(embedding, (0, self.embedding_size - len(embedding)))
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_face_embedding(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """Detect face and extract embedding from image."""
        faces = self.detect_faces(image)
        
        if not faces:
            return None, "No face detected in image"
        
        # Use largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        preprocessed = self.preprocess_face(image, largest_face)
        embedding = self.extract_embedding(preprocessed)
        
        return embedding, "ok"
    
    def enroll_user(self, user_id: str, images: List[np.ndarray]) -> Tuple[bool, str]:
        """Enroll user with multiple face images."""
        embeddings = []
        
        for i, img in enumerate(images):
            embedding, msg = self.get_face_embedding(img)
            if embedding is None:
                return False, f"Failed on image {i+1}: {msg}"
            embeddings.append(embedding)
        
        # Calculate mean embedding
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-7)
        
        self.user_embeddings[user_id] = mean_embedding
        return True, f"Enrolled {len(images)} face samples"
    
    def verify(self, image: np.ndarray, user_id: str, threshold: float = None) -> Tuple[bool, float, str]:
        """Verify if face matches enrolled user."""
        threshold = threshold or self.threshold
        
        if user_id not in self.user_embeddings:
            return False, 0.0, f"No face enrollment found for '{user_id}'"
        
        embedding, msg = self.get_face_embedding(image)
        if embedding is None:
            return False, 0.0, msg
        
        enrolled = self.user_embeddings[user_id]
        similarity = cosine_similarity([embedding], [enrolled])[0][0]
        
        if similarity >= threshold:
            return True, similarity, f"Face match: {similarity:.1%}"
        else:
            return False, similarity, f"Face mismatch: {similarity:.1%} (threshold: {threshold:.1%})"
    
    def identify(self, image: np.ndarray, threshold: float = None) -> Tuple[Optional[str], float, str]:
        """Identify user from face (1:N matching)."""
        threshold = threshold or self.threshold
        
        if not self.user_embeddings:
            return None, 0.0, "No enrolled users"
        
        embedding, msg = self.get_face_embedding(image)
        if embedding is None:
            return None, 0.0, msg
        
        best_match = None
        best_score = 0.0
        
        for user_id, enrolled in self.user_embeddings.items():
            similarity = cosine_similarity([embedding], [enrolled])[0][0]
            if similarity > best_score:
                best_score = similarity
                best_match = user_id
        
        if best_score >= threshold:
            return best_match, best_score, f"Identified as {best_match}: {best_score:.1%}"
        else:
            return None, best_score, f"No match found (best: {best_score:.1%})"
    
    def load_embeddings(self, embeddings_dict: Dict[str, np.ndarray]):
        """Load user embeddings from database."""
        self.user_embeddings = embeddings_dict
        print(f"[FaceModel] Loaded {len(embeddings_dict)} user embeddings")
