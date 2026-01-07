"""
Database utilities supporting SQLite and MongoDB.
"""

import os
import sqlite3
import json
import pickle
from datetime import datetime
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod

try:
    from pymongo import MongoClient
    from bson import ObjectId
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False


class DatabaseInterface(ABC):
    """Abstract database interface."""
    
    @abstractmethod
    def create_user(self, user_data: dict) -> str:
        pass
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[dict]:
        pass
    
    @abstractmethod
    def update_user(self, user_id: str, data: dict) -> bool:
        pass
    
    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_all_users(self) -> List[dict]:
        pass
    
    @abstractmethod
    def log_event(self, event: dict) -> str:
        pass
    
    @abstractmethod
    def get_audit_logs(self, user_id: str = None, limit: int = 100) -> List[dict]:
        pass


class SQLiteDatabase(DatabaseInterface):
    """SQLite implementation for development."""
    
    def __init__(self, db_path: str = "data/biometric.db"):
        # Make path absolute relative to project root
        if not os.path.isabs(db_path):
            # Get the directory where this file is located
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base_dir, db_path)
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_tables()
    
    def _get_conn(self):
        return sqlite3.connect(self.db_path)
    
    def _init_tables(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                full_name TEXT,
                family_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                face_embeddings BLOB,
                speaker_embeddings BLOB,
                passphrase_hash TEXT,
                passphrase_encrypted BLOB,
                gesture_preference TEXT DEFAULT 'thumbs_up',
                enrollment_complete INTEGER DEFAULT 0,
                face_sample_count INTEGER DEFAULT 0,
                voice_sample_count INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                user_id TEXT,
                success INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                session_id TEXT,
                details TEXT,
                reason TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, user_data: dict) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (user_id, name, full_name, family_name, gesture_preference)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_data['user_id'],
            user_data['name'],
            user_data.get('full_name', ''),
            user_data.get('family_name', ''),
            user_data.get('gesture_preference', 'thumbs_up')
        ))
        
        conn.commit()
        conn.close()
        return user_data['user_id']
    
    def get_user(self, user_id: str) -> Optional[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = ['id', 'user_id', 'name', 'full_name', 'family_name',
                      'created_at', 'updated_at', 'is_active', 'face_embeddings',
                      'speaker_embeddings', 'passphrase_hash', 'passphrase_encrypted',
                      'gesture_preference', 'enrollment_complete', 'face_sample_count',
                      'voice_sample_count']
            return dict(zip(columns, row))
        return None
    
    def update_user(self, user_id: str, data: dict) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        values = list(data.values()) + [user_id]
        
        cursor.execute(f'''
            UPDATE users SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', values)
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
    
    def delete_user(self, user_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
    
    def get_all_users(self) -> List[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, name, family_name, is_active, enrollment_complete FROM users')
        rows = cursor.fetchall()
        conn.close()
        
        return [{'user_id': r[0], 'name': r[1], 'family_name': r[2],
                'is_active': bool(r[3]), 'enrollment_complete': bool(r[4])} for r in rows]
    
    def log_event(self, event: dict) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_logs (event_type, user_id, success, ip_address, session_id, details, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.get('event_type'),
            event.get('user_id'),
            1 if event.get('success') else 0,
            event.get('ip_address'),
            event.get('session_id'),
            json.dumps(event.get('details', {})),
            event.get('reason')
        ))
        
        conn.commit()
        log_id = cursor.lastrowid
        conn.close()
        return str(log_id)
    
    def get_audit_logs(self, user_id: str = None, limit: int = 100) -> List[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute(
                'SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
                (user_id, limit)
            )
        else:
            cursor.execute(
                'SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'event_type', 'user_id', 'success', 'timestamp',
                  'ip_address', 'session_id', 'details', 'reason']
        return [dict(zip(columns, row)) for row in rows]


class MongoDatabase(DatabaseInterface):
    """MongoDB implementation for production."""
    
    def __init__(self, uri: str, database: str):
        if not MONGO_AVAILABLE:
            raise ImportError("pymongo not installed")
        
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.users = self.db.users
        self.audit_logs = self.db.audit_logs
    
    def create_user(self, user_data: dict) -> str:
        user_data['created_at'] = datetime.utcnow()
        user_data['updated_at'] = datetime.utcnow()
        user_data['is_active'] = True
        user_data['enrollment_complete'] = False
        
        result = self.users.insert_one(user_data)
        return user_data['user_id']
    
    def get_user(self, user_id: str) -> Optional[dict]:
        user = self.users.find_one({'user_id': user_id})
        if user:
            user['_id'] = str(user['_id'])
        return user
    
    def update_user(self, user_id: str, data: dict) -> bool:
        data['updated_at'] = datetime.utcnow()
        result = self.users.update_one({'user_id': user_id}, {'$set': data})
        return result.modified_count > 0
    
    def delete_user(self, user_id: str) -> bool:
        result = self.users.delete_one({'user_id': user_id})
        return result.deleted_count > 0
    
    def get_all_users(self) -> List[dict]:
        users = self.users.find({}, {'user_id': 1, 'name': 1, 'family_name': 1,
                                     'is_active': 1, 'enrollment_complete': 1})
        return [{'user_id': u['user_id'], 'name': u['name'],
                'family_name': u.get('family_name', ''),
                'is_active': u.get('is_active', True),
                'enrollment_complete': u.get('enrollment_complete', False)} for u in users]
    
    def log_event(self, event: dict) -> str:
        event['timestamp'] = datetime.utcnow()
        result = self.audit_logs.insert_one(event)
        return str(result.inserted_id)
    
    def get_audit_logs(self, user_id: str = None, limit: int = 100) -> List[dict]:
        query = {'user_id': user_id} if user_id else {}
        logs = self.audit_logs.find(query).sort('timestamp', -1).limit(limit)
        result = []
        for log in logs:
            log['_id'] = str(log['_id'])
            result.append(log)
        return result


def get_database(config: dict) -> DatabaseInterface:
    """Factory function to get appropriate database."""
    db_type = config.get('database', {}).get('type', 'sqlite')
    
    if db_type == 'mongodb' and MONGO_AVAILABLE:
        mongo_config = config['database']['mongodb']
        return MongoDatabase(mongo_config['uri'], mongo_config['database'])
    else:
        sqlite_config = config['database'].get('sqlite', {})
        return SQLiteDatabase(sqlite_config.get('path', 'data/biometric.db'))
