"""
Security logging utilities for audit trails.
"""

import os
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


class SecurityLogger:
    """Handles security event logging with JSON audit trails."""
    
    def __init__(self, log_file: str = "logs/security.log"):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        self.logger = logging.getLogger("security")
        self.logger.setLevel(logging.INFO)
        
        # Rotating file handler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(console)
    
    def log_event(self, event_type: str, user_id: str = None, success: bool = True,
                  details: dict = None, ip_address: str = None, reason: str = None):
        """Log a security event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "success": success,
            "ip_address": ip_address,
            "details": details or {},
            "reason": reason
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, json.dumps(event))
        return event
    
    def log_enrollment(self, user_id: str, success: bool, details: dict = None):
        """Log enrollment event."""
        return self.log_event("enrollment", user_id, success, details)
    
    def log_authentication(self, user_id: str, success: bool, details: dict = None,
                          ip_address: str = None, reason: str = None):
        """Log authentication attempt."""
        return self.log_event("authentication", user_id, success, details, ip_address, reason)
    
    def log_admin_action(self, admin_id: str, action: str, target_user: str = None):
        """Log admin actions."""
        return self.log_event("admin_action", admin_id, True, {
            "action": action,
            "target_user": target_user
        })


# Global logger instance
security_logger = SecurityLogger()
