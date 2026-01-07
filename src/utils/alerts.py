"""
Alert System Module
- Email notifications
- SMS alerts (placeholder)
- Webhook triggers
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AlertManager:
    """Manages security alerts and notifications."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.email_config = self.config.get('email', {})
        self.webhooks = self.config.get('webhooks', [])
        self.alert_history: List[dict] = []
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email notification."""
        if not self.email_config.get('enabled'):
            print("[Alerts] Email not configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_email')
            msg['To'] = to
            msg['Subject'] = f"[Biometric Security] {subject}"
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(
                self.email_config.get('smtp_host', 'smtp.gmail.com'),
                self.email_config.get('smtp_port', 587)
            )
            server.starttls()
            server.login(
                self.email_config.get('username'),
                self.email_config.get('password')
            )
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"[Alerts] Email failed: {e}")
            return False
    
    def trigger_webhook(self, event_type: str, data: dict) -> List[dict]:
        """Trigger all configured webhooks."""
        if not REQUESTS_AVAILABLE:
            return []
        
        results = []
        
        for webhook in self.webhooks:
            if not webhook.get('enabled', True):
                continue
            
            # Check if webhook is subscribed to this event type
            events = webhook.get('events', ['all'])
            if 'all' not in events and event_type not in events:
                continue
            
            try:
                payload = {
                    "event": event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data
                }
                
                response = requests.post(
                    webhook['url'],
                    json=payload,
                    headers=webhook.get('headers', {}),
                    timeout=10
                )
                
                results.append({
                    "url": webhook['url'],
                    "success": response.status_code < 400,
                    "status": response.status_code
                })
            except Exception as e:
                results.append({
                    "url": webhook['url'],
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def alert_failed_attempts(self, user_id: str, attempt_count: int, ip_address: str):
        """Alert on multiple failed authentication attempts."""
        subject = f"Security Alert: Multiple Failed Attempts for {user_id}"
        body = f"""
        <h2>Security Alert</h2>
        <p><strong>User:</strong> {user_id}</p>
        <p><strong>Failed Attempts:</strong> {attempt_count}</p>
        <p><strong>IP Address:</strong> {ip_address}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Please review the security logs for more details.</p>
        """
        
        admin_email = self.config.get('admin_email')
        if admin_email:
            self.send_email(admin_email, subject, body)
        
        self.trigger_webhook('failed_attempts', {
            'user_id': user_id,
            'attempt_count': attempt_count,
            'ip_address': ip_address
        })
        
        self._log_alert('failed_attempts', user_id, {'count': attempt_count, 'ip': ip_address})
    
    def alert_successful_auth(self, user_id: str, ip_address: str, details: dict):
        """Trigger webhook on successful authentication."""
        self.trigger_webhook('auth_success', {
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details
        })
    
    def alert_new_enrollment(self, user_id: str, name: str):
        """Alert on new user enrollment."""
        self.trigger_webhook('new_enrollment', {
            'user_id': user_id,
            'name': name
        })
    
    def alert_after_hours(self, user_id: str, ip_address: str):
        """Alert on after-hours access attempt."""
        subject = f"After-Hours Access: {user_id}"
        body = f"""
        <h2>After-Hours Access Attempt</h2>
        <p><strong>User:</strong> {user_id}</p>
        <p><strong>IP Address:</strong> {ip_address}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        admin_email = self.config.get('admin_email')
        if admin_email:
            self.send_email(admin_email, subject, body)
        
        self.trigger_webhook('after_hours', {
            'user_id': user_id,
            'ip_address': ip_address
        })
    
    def _log_alert(self, alert_type: str, user_id: str, details: dict):
        """Log alert to history."""
        self.alert_history.append({
            'type': alert_type,
            'user_id': user_id,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only last 1000 alerts in memory
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def get_recent_alerts(self, limit: int = 50) -> List[dict]:
        """Get recent alerts."""
        return self.alert_history[-limit:][::-1]


class LockoutManager:
    """Manages account lockouts after failed attempts."""
    
    def __init__(self, max_attempts: int = 5, lockout_minutes: int = 15):
        self.max_attempts = max_attempts
        self.lockout_minutes = lockout_minutes
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.lockouts: Dict[str, datetime] = {}
    
    def record_failure(self, user_id: str) -> Tuple[int, bool]:
        """Record a failed attempt. Returns (attempt_count, is_locked)."""
        now = datetime.utcnow()
        
        # Clean old attempts (older than lockout period)
        cutoff = now - timedelta(minutes=self.lockout_minutes)
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Remove old attempts
        self.failed_attempts[user_id] = [
            t for t in self.failed_attempts[user_id] if t > cutoff
        ]
        
        # Add new attempt
        self.failed_attempts[user_id].append(now)
        count = len(self.failed_attempts[user_id])
        
        # Check if should lock
        if count >= self.max_attempts:
            self.lockouts[user_id] = now
            return count, True
        
        return count, False
    
    def is_locked(self, user_id: str) -> Tuple[bool, Optional[int]]:
        """Check if user is locked out. Returns (is_locked, minutes_remaining)."""
        if user_id not in self.lockouts:
            return False, None
        
        lockout_time = self.lockouts[user_id]
        unlock_time = lockout_time + timedelta(minutes=self.lockout_minutes)
        now = datetime.utcnow()
        
        if now >= unlock_time:
            # Lockout expired
            del self.lockouts[user_id]
            self.failed_attempts.pop(user_id, None)
            return False, None
        
        remaining = int((unlock_time - now).total_seconds() / 60) + 1
        return True, remaining
    
    def clear_lockout(self, user_id: str):
        """Manually clear a lockout (admin action)."""
        self.lockouts.pop(user_id, None)
        self.failed_attempts.pop(user_id, None)
    
    def record_success(self, user_id: str):
        """Clear failed attempts on successful auth."""
        self.failed_attempts.pop(user_id, None)
