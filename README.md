# 🔐 Biometric Security System

A comprehensive multi-modal biometric authentication system for home/office security.

## Features

### 🔒 Security Features
- **Multi-Modal Authentication**: Face + Voice + Gesture verification
- **Liveness Detection**: Blink detection, challenge-response, anti-photo spoofing
- **Voice Anti-Spoofing**: Replay attack detection, audio quality analysis
- **Account Lockout**: Auto-lock after failed attempts (configurable)
- **Encrypted Storage**: Fernet encryption for all biometric embeddings
- **Audit Logging**: Complete security event trail

### 📊 Admin Dashboard
- Real-time authentication statistics
- Success/failure rate monitoring
- User management (add, edit, delete, unlock)
- Activity log with filtering
- System status overview
- Configurable thresholds

### 🎯 Quality Checks
- **Face Quality**: Brightness, contrast, blur, face size, centering
- **Audio Quality**: Volume, SNR, clipping, duration
- Real-time feedback during capture

### 🔔 Alert System
- Email notifications for security events
- Webhook support for integrations
- After-hours access monitoring
- Failed attempt alerts

### 💾 Progress Saving
- Save incomplete enrollments
- Resume enrollment later

## Quick Start

```bash
# Activate virtual environment
secure\Scripts\activate

# Run the application
python run.py
```

Open `http://localhost:5000` in your browser.

## Pages

- `/` - Authentication page
- `/enrollment` - User enrollment wizard
- `/admin` - Admin dashboard

## API Endpoints

### Authentication
- `POST /api/authenticate` - Authenticate user
- `GET /api/liveness/challenge` - Get liveness challenge
- `POST /api/liveness/verify` - Verify liveness

### Enrollment
- `POST /api/enroll` - Enroll new user
- `POST /api/enrollment/save` - Save progress
- `GET /api/enrollment/load/<id>` - Load progress

### Quality
- `POST /api/quality/face` - Check face image quality
- `POST /api/transcribe` - Transcribe audio

### Users
- `GET /api/users` - List all users
- `GET /api/users/<id>` - Get user details
- `PUT /api/users/<id>` - Update user
- `DELETE /api/users/<id>` - Delete user
- `POST /api/users/<id>/unlock` - Unlock user

### System
- `GET /api/stats` - Get statistics
- `GET /api/logs` - Get audit logs
- `GET /api/status` - System status

## Configuration

Edit `src/config/config.yaml`:

```yaml
auth:
  require_face: true
  require_voice: true
  require_gesture: true
  thresholds:
    face: 0.6
    speaker: 0.7

security:
  max_failed_attempts: 5
  lockout_duration_minutes: 15
  liveness:
    enabled: true
```

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **ML Models**: OpenCV, Whisper, MediaPipe
- **Database**: SQLite (dev) / MongoDB (prod)
- **Security**: Fernet encryption, bcrypt

## Enrollment Flow

1. Enter family/member info
2. Capture 6 face images (2 straight + 4 directions)
3. Record 3 voice samples
4. Complete enrollment

## Authentication Flow

1. (Optional) Enter user ID
2. Capture face with gesture
3. Record voice passphrase
4. System verifies all modalities
5. Access granted/denied with confidence scores
