"""
Encryption utilities for biometric data protection.
Uses Fernet symmetric encryption for embeddings.
"""

import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
import numpy as np


class EncryptionManager:
    """Handles encryption/decryption of biometric embeddings."""
    
    def __init__(self, key: bytes = None):
        """Initialize with optional key, or generate new one."""
        if key:
            self.key = key
        else:
            self.key = self._load_or_generate_key()
        self.fernet = Fernet(self.key)
    
    def _load_or_generate_key(self) -> bytes:
        """Load existing key or generate new one."""
        # Get project root directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, "data")
        key_path = os.path.join(data_dir, ".encryption_key")
        
        os.makedirs(data_dir, exist_ok=True)
        
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            return key
    
    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Encrypt a numpy embedding array."""
        embedding_bytes = embedding.tobytes()
        return self.fernet.encrypt(embedding_bytes)
    
    def decrypt_embedding(self, encrypted_data: bytes, shape: tuple, dtype=np.float64) -> np.ndarray:
        """Decrypt bytes back to numpy array."""
        decrypted_bytes = self.fernet.decrypt(encrypted_data)
        arr = np.frombuffer(decrypted_bytes, dtype=dtype)
        # If shape is (-1,), return flat array
        if shape == (-1,):
            return arr
        return arr.reshape(shape)
    
    def encrypt_text(self, text: str) -> bytes:
        """Encrypt text string."""
        return self.fernet.encrypt(text.encode('utf-8'))
    
    def decrypt_text(self, encrypted_data: bytes) -> str:
        """Decrypt bytes back to text."""
        return self.fernet.decrypt(encrypted_data).decode('utf-8')


def hash_passphrase(passphrase: str) -> str:
    """Hash passphrase using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(passphrase.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_passphrase(passphrase: str, hashed: str) -> bool:
    """Verify passphrase against hash."""
    return bcrypt.checkpw(passphrase.encode('utf-8'), hashed.encode('utf-8'))


def derive_key_from_password(password: str, salt: bytes = None) -> tuple:
    """Derive encryption key from password using PBKDF2."""
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt
