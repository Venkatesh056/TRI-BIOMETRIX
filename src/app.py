"""
Biometric Security System - Flask Application
Main entry point with all features.
"""
import os
import io
import base64
import yaml
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

from models import FaceRecognitionModel, SpeakerVerificationModel, SpeechRecognitionModel, GestureRecognitionModel
from utils import (
    get_database, EncryptionManager, security_logger, hash_passphrase,
    FaceQualityChecker, AudioQualityChecker, LivenessDetector, 
    PassphraseGenerator, AlertManager, LockoutManager
)

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = config['app']['secret_key']

# Initialize components
db = get_database(config)
encryption = EncryptionManager()
face_model = FaceRecognitionModel(config.get('models', {}).get('face', {}))
speaker_model = SpeakerVerificationModel(config.get('models', {}).get('speaker', {}))
speech_model = SpeechRecognitionModel(config.get('models', {}).get('speech', {}))
gesture_model = GestureRecognitionModel(config.get('models', {}).get('gesture', {}))

# New components
face_quality = FaceQualityChecker()
audio_quality = AudioQualityChecker()
liveness = LivenessDetector()
alert_manager = AlertManager(config.get('alerts', {}))
lockout_manager = LockoutManager(
    config.get('security', {}).get('max_failed_attempts', 5),
    config.get('security', {}).get('lockout_duration_minutes', 15)
)

# Store for incomplete enrollments (progress saving)
enrollment_progress = {}

# Store for authentication stats
auth_stats = {
    'total_attempts': 0,
    'successful': 0,
    'failed': 0,
    'hourly_stats': {}
}


# Load existing embeddings from database on startup
def load_all_embeddings():
    """Load all user embeddings from database into models."""
    users = db.get_all_users()
    for user_info in users:
        user = db.get_user(user_info['user_id'])
        if user and user.get('face_embeddings'):
            try:
                face_size = config.get('models', {}).get('face', {}).get('embedding_size', 128)
                emb = encryption.decrypt_embedding(user['face_embeddings'], (-1,))
                face_model.user_embeddings[user['user_id']] = emb
            except Exception as e:
                print(f"[Warning] Failed to load face embedding for {user_info['user_id']}: {e}")
        if user and user.get('speaker_embeddings'):
            try:
                emb = encryption.decrypt_embedding(user['speaker_embeddings'], (-1,))
                speaker_model.user_embeddings[user['user_id']] = emb
            except Exception as e:
                print(f"[Warning] Failed to load speaker embedding for {user_info['user_id']}: {e}")

load_all_embeddings()


def is_after_hours() -> bool:
    """Check if current time is in after-hours period."""
    ah_config = config.get('security', {}).get('after_hours', {})
    if not ah_config.get('enabled'):
        return False
    
    current_hour = datetime.now().hour
    start = ah_config.get('start_hour', 22)
    end = ah_config.get('end_hour', 6)
    
    if start > end:  # Spans midnight
        return current_hour >= start or current_hour < end
    else:
        return start <= current_hour < end


def update_auth_stats(success: bool):
    """Update authentication statistics."""
    auth_stats['total_attempts'] += 1
    if success:
        auth_stats['successful'] += 1
    else:
        auth_stats['failed'] += 1
    
    # Hourly stats
    hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
    if hour_key not in auth_stats['hourly_stats']:
        auth_stats['hourly_stats'][hour_key] = {'success': 0, 'failed': 0}
    
    if success:
        auth_stats['hourly_stats'][hour_key]['success'] += 1
    else:
        auth_stats['hourly_stats'][hour_key]['failed'] += 1


# ============== ROUTES ==============

@app.route('/')
def index():
    """Home page - Authentication interface."""
    return render_template('index.html', config=config)

@app.route('/enrollment')
def enrollment():
    """Enrollment page for new users."""
    return render_template('enrollment.html', 
                          max_members=config['enrollment']['max_family_members'],
                          face_samples=config['enrollment']['face_samples'],
                          voice_samples=config['enrollment']['voice_samples'],
                          gestures=gesture_model.get_supported_gestures(),
                          config=config)

@app.route('/admin')
def admin():
    """Admin dashboard."""
    users = db.get_all_users()
    logs = db.get_audit_logs(limit=50)
    return render_template('admin.html', 
                          users=users, 
                          logs=logs, 
                          stats=auth_stats,
                          config=config,
                          lockouts=lockout_manager.lockouts)


# ============== API ENDPOINTS ==============

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """Transcribe audio to text."""
    if 'audio' not in request.files:
        data = request.get_json()
        if data and data.get('audio'):
            audio_b64 = data['audio']
            audio_bytes = base64.b64decode(audio_b64.split(',')[1] if ',' in audio_b64 else audio_b64)
        else:
            return jsonify({'success': False, 'error': 'No audio file'}), 400
    else:
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
    
    text, confidence, detected_lang = speech_model.transcribe(audio_bytes=audio_bytes)
    
    return jsonify({
        'success': True,
        'text': text,
        'confidence': confidence,
        'language': detected_lang
    })


@app.route('/api/verify_passphrase', methods=['POST'])
def api_verify_passphrase():
    """Verify that spoken audio matches the expected passphrase text."""
    data = request.get_json()
    audio_b64 = data.get('audio')
    expected_text = data.get('expected_text', '')
    threshold = data.get('threshold', 0.6)  # Lower threshold for multi-language
    
    if not audio_b64:
        return jsonify({'success': False, 'error': 'No audio provided'}), 400
    
    if not expected_text:
        return jsonify({'success': False, 'error': 'No expected text provided'}), 400
    
    try:
        audio_bytes = base64.b64decode(audio_b64.split(',')[1] if ',' in audio_b64 else audio_b64)
        
        matched, similarity, transcribed, message = speech_model.verify_passphrase(
            audio_bytes=audio_bytes,
            expected_text=expected_text,
            threshold=threshold
        )
        
        return jsonify({
            'success': True,
            'matched': matched,
            'similarity': round(similarity * 100, 1),
            'transcribed': transcribed,
            'expected': expected_text,
            'message': message
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quality/face', methods=['POST'])
def api_check_face_quality():
    """Check face image quality before capture."""
    data = request.get_json()
    image_b64 = data.get('image')
    
    if not image_b64:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    try:
        import cv2
        img_data = base64.b64decode(image_b64.split(',')[1] if ',' in image_b64 else image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        quality_result = face_quality.check_quality(img)
        return jsonify({'success': True, **quality_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/liveness/challenge', methods=['GET'])
def api_get_liveness_challenge():
    """Get a random liveness challenge."""
    count = request.args.get('count', 1, type=int)
    challenges = liveness.generate_challenge(count)
    
    # Store in session for verification
    session['liveness_challenges'] = [c['type'] for c in challenges]
    
    return jsonify({'success': True, 'challenges': challenges})


@app.route('/api/liveness/verify', methods=['POST'])
def api_verify_liveness():
    """Verify liveness challenge completion."""
    data = request.get_json()
    frames_b64 = data.get('frames', [])
    challenge_type = data.get('challenge_type')
    
    if not frames_b64:
        return jsonify({'success': False, 'error': 'No frames provided'}), 400
    
    try:
        import cv2
        frames = []
        for b64 in frames_b64:
            img_data = base64.b64decode(b64.split(',')[1] if ',' in b64 else b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        
        passed, message = liveness.verify_challenge(challenge_type, frames)
        
        return jsonify({
            'success': True,
            'passed': passed,
            'message': message
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/passphrase/generate', methods=['GET'])
def api_generate_passphrase():
    """Generate a random passphrase for anti-replay."""
    passphrase = PassphraseGenerator.generate_with_number()
    session['expected_passphrase'] = passphrase
    
    return jsonify({
        'success': True,
        'passphrase': passphrase
    })


@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """Enroll a new user with biometric data."""
    data = request.get_json()
    
    user_id = data.get('user_id', '').lower().replace(' ', '_')
    name = data.get('name', '')
    family_name = data.get('family_name', '')
    passphrase = data.get('passphrase', '')
    face_images_b64 = data.get('face_images', [])
    voice_samples_b64 = data.get('voice_samples', [])
    gesture_pref = data.get('gesture_preference', 'thumbs_up')
    
    if not user_id or not name:
        return jsonify({'success': False, 'error': 'Missing user_id or name'}), 400
    
    if db.get_user(user_id):
        return jsonify({'success': False, 'error': 'User already exists'}), 400
    
    try:
        import cv2
        
        # Process face images
        face_images = []
        for b64 in face_images_b64:
            img_data = base64.b64decode(b64.split(',')[1] if ',' in b64 else b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                face_images.append(img)
        
        face_success, face_msg = face_model.enroll_user(user_id, face_images)
        if not face_success:
            return jsonify({'success': False, 'error': f'Face enrollment failed: {face_msg}'}), 400
        
        # Process voice samples
        voice_bytes = []
        for b64 in voice_samples_b64:
            audio_data = base64.b64decode(b64.split(',')[1] if ',' in b64 else b64)
            voice_bytes.append(audio_data)
        
        voice_success, voice_msg = speaker_model.enroll_user(user_id, audio_bytes_list=voice_bytes)
        if not voice_success:
            return jsonify({'success': False, 'error': f'Voice enrollment failed: {voice_msg}'}), 400
        
        # Create user in database
        user_data = {
            'user_id': user_id,
            'name': name,
            'family_name': family_name,
            'gesture_preference': gesture_pref
        }
        db.create_user(user_data)
        
        # Save encrypted embeddings
        face_emb_encrypted = encryption.encrypt_embedding(face_model.user_embeddings[user_id])
        speaker_emb_encrypted = encryption.encrypt_embedding(speaker_model.user_embeddings[user_id])
        passphrase_hash = hash_passphrase(passphrase) if passphrase else None
        
        db.update_user(user_id, {
            'face_embeddings': face_emb_encrypted,
            'speaker_embeddings': speaker_emb_encrypted,
            'passphrase_hash': passphrase_hash,
            'enrollment_complete': 1,
            'face_sample_count': len(face_images),
            'voice_sample_count': len(voice_bytes)
        })
        
        security_logger.log_enrollment(user_id, True, {
            'face_samples': len(face_images),
            'voice_samples': len(voice_bytes)
        })
        
        # Trigger webhook
        alert_manager.alert_new_enrollment(user_id, name)
        
        return jsonify({'success': True, 'message': f'User {name} enrolled successfully'})
        
    except Exception as e:
        security_logger.log_enrollment(user_id, False, {'error': str(e)})
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/authenticate', methods=['POST'])
def api_authenticate():
    """Authenticate user with biometric data."""
    data = request.get_json()
    
    user_id = data.get('user_id', '').lower().replace(' ', '_') if data.get('user_id') else ''
    face_image_b64 = data.get('face_image')
    voice_sample_b64 = data.get('voice_sample')
    gesture_detected = data.get('gesture_detected', False)
    gesture_type = data.get('gesture_type', 'none')
    
    # Check lockout
    if user_id:
        is_locked, minutes_remaining = lockout_manager.is_locked(user_id)
        if is_locked:
            return jsonify({
                'success': False,
                'message': f'Account locked. Try again in {minutes_remaining} minutes.',
                'locked': True,
                'minutes_remaining': minutes_remaining
            })
    
    # Check after-hours
    if is_after_hours():
        ah_config = config.get('security', {}).get('after_hours', {})
        if ah_config.get('alert_on_access'):
            alert_manager.alert_after_hours(user_id or 'unknown', request.remote_addr)
    
    results = {
        'face': {'success': False, 'score': 0.0, 'message': 'Not checked'},
        'voice': {'success': False, 'score': 0.0, 'message': 'Not checked'},
        'gesture': {'success': False, 'message': 'Not checked'},
        'liveness': {'success': True, 'message': 'Not required'}
    }
    
    auth_config = config.get('auth', {})
    thresholds = auth_config.get('thresholds', {})
    
    try:
        import cv2
        
        # Face verification
        if auth_config.get('require_face', True) and face_image_b64:
            img_data = base64.b64decode(face_image_b64.split(',')[1] if ',' in face_image_b64 else face_image_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Get threshold (adaptive if enabled)
            face_threshold = thresholds.get('face', 0.6)
            
            if user_id:
                success, score, msg = face_model.verify(img, user_id, face_threshold)
            else:
                identified, score, msg = face_model.identify(img, face_threshold)
                success = identified is not None
                if success:
                    user_id = identified
            
            results['face'] = {'success': success, 'score': round(float(score) * 100, 1), 'message': msg}
        
        # Voice verification
        if auth_config.get('require_voice', True) and voice_sample_b64:
            audio_data = base64.b64decode(voice_sample_b64.split(',')[1] if ',' in voice_sample_b64 else voice_sample_b64)
            
            speaker_threshold = thresholds.get('speaker', 0.7)
            
            if user_id:
                success, score, msg = speaker_model.verify(user_id, audio_bytes=audio_data, threshold=speaker_threshold)
            else:
                identified, score, msg = speaker_model.identify(audio_bytes=audio_data, threshold=speaker_threshold)
                success = identified is not None
                if success and not user_id:
                    user_id = identified
            
            results['voice'] = {'success': success, 'score': round(float(score) * 100, 1), 'message': msg}
        
        # Gesture verification
        if auth_config.get('require_gesture', True):
            if gesture_detected:
                user = db.get_user(user_id) if user_id else None
                expected = user.get('gesture_preference', 'thumbs_up') if user else 'thumbs_up'
                
                if gesture_type == expected or gesture_type in ['thumbs_up', 'open_palm']:
                    results['gesture'] = {'success': True, 'message': f'Gesture verified: {gesture_type}'}
                else:
                    results['gesture'] = {'success': False, 'message': f'Expected {expected}, got {gesture_type}'}
            else:
                results['gesture'] = {'success': False, 'message': 'No gesture detected'}
        else:
            results['gesture'] = {'success': True, 'message': 'Gesture not required'}
        
        # Decision
        required_checks = []
        if auth_config.get('require_face', True):
            required_checks.append(results['face']['success'])
        if auth_config.get('require_voice', True):
            required_checks.append(results['voice']['success'])
        if auth_config.get('require_gesture', True):
            required_checks.append(results['gesture']['success'])
        
        overall_success = all(required_checks) if required_checks else False
        
        # Build failure reasons
        reasons = []
        if not results['face']['success']:
            reasons.append(f"Face: {results['face']['message']}")
        if not results['voice']['success']:
            reasons.append(f"Voice: {results['voice']['message']}")
        if not results['gesture']['success']:
            reasons.append(f"Gesture: {results['gesture']['message']}")
        
        # Update stats
        update_auth_stats(overall_success)
        
        # Handle lockout
        if not overall_success and user_id:
            attempt_count, is_now_locked = lockout_manager.record_failure(user_id)
            
            if is_now_locked:
                alert_manager.alert_failed_attempts(user_id, attempt_count, request.remote_addr)
            
            results['attempts_remaining'] = max(0, lockout_manager.max_attempts - attempt_count)
        elif overall_success and user_id:
            lockout_manager.record_success(user_id)
        
        # Log event
        security_logger.log_authentication(
            user_id or 'unknown',
            overall_success,
            results,
            request.remote_addr,
            '; '.join(reasons) if reasons else 'ok'
        )
        
        db.log_event({
            'event_type': 'authentication',
            'user_id': user_id,
            'success': overall_success,
            'ip_address': request.remote_addr,
            'details': results,
            'reason': '; '.join(reasons) if reasons else 'ok'
        })
        
        if overall_success:
            session['user_id'] = user_id
            alert_manager.alert_successful_auth(user_id, request.remote_addr, results)
            
            return jsonify({
                'success': True,
                'message': f'Access granted! Welcome, {user_id}',
                'user_id': user_id,
                'details': results
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Authentication failed',
                'reasons': reasons,
                'details': results
            })
            
    except Exception as e:
        security_logger.log_authentication(user_id or 'unknown', False, {'error': str(e)}, request.remote_addr)
        return jsonify({'success': False, 'error': str(e)}), 500


# ============== ENROLLMENT PROGRESS ==============

@app.route('/api/enrollment/save', methods=['POST'])
def api_save_enrollment_progress():
    """Save enrollment progress for later resumption."""
    data = request.get_json()
    session_id = data.get('session_id') or session.get('enrollment_session')
    
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        session['enrollment_session'] = session_id
    
    enrollment_progress[session_id] = {
        'data': data,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/enrollment/load/<session_id>', methods=['GET'])
def api_load_enrollment_progress(session_id):
    """Load saved enrollment progress."""
    if session_id in enrollment_progress:
        return jsonify({
            'success': True,
            'data': enrollment_progress[session_id]['data']
        })
    return jsonify({'success': False, 'error': 'Session not found'}), 404


# ============== USER MANAGEMENT ==============

@app.route('/api/users', methods=['GET'])
def api_get_users():
    """Get all enrolled users."""
    users = db.get_all_users()
    return jsonify({'success': True, 'users': users})


@app.route('/api/users/<user_id>', methods=['GET'])
def api_get_user(user_id):
    """Get single user details."""
    user = db.get_user(user_id)
    if user:
        # Remove sensitive data
        user.pop('face_embeddings', None)
        user.pop('speaker_embeddings', None)
        user.pop('passphrase_hash', None)
        user.pop('passphrase_encrypted', None)
        return jsonify({'success': True, 'user': user})
    return jsonify({'success': False, 'error': 'User not found'}), 404


@app.route('/api/users/<user_id>', methods=['PUT'])
def api_update_user(user_id):
    """Update user profile."""
    data = request.get_json()
    
    # Only allow updating certain fields
    allowed_fields = ['name', 'family_name', 'gesture_preference', 'is_active']
    update_data = {k: v for k, v in data.items() if k in allowed_fields}
    
    if db.update_user(user_id, update_data):
        security_logger.log_admin_action('admin', 'update_user', user_id)
        return jsonify({'success': True, 'message': f'User {user_id} updated'})
    return jsonify({'success': False, 'error': 'User not found'}), 404


@app.route('/api/users/<user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    """Delete a user."""
    if db.delete_user(user_id):
        face_model.user_embeddings.pop(user_id, None)
        speaker_model.user_embeddings.pop(user_id, None)
        security_logger.log_admin_action('admin', 'delete_user', user_id)
        return jsonify({'success': True, 'message': f'User {user_id} deleted'})
    return jsonify({'success': False, 'error': 'User not found'}), 404


@app.route('/api/users/<user_id>/unlock', methods=['POST'])
def api_unlock_user(user_id):
    """Manually unlock a locked user account."""
    lockout_manager.clear_lockout(user_id)
    security_logger.log_admin_action('admin', 'unlock_user', user_id)
    return jsonify({'success': True, 'message': f'User {user_id} unlocked'})


# ============== STATS & LOGS ==============

@app.route('/api/stats', methods=['GET'])
def api_get_stats():
    """Get authentication statistics."""
    # Get recent hourly stats
    recent_hours = sorted(auth_stats['hourly_stats'].items())[-24:]
    
    return jsonify({
        'success': True,
        'total_attempts': auth_stats['total_attempts'],
        'successful': auth_stats['successful'],
        'failed': auth_stats['failed'],
        'success_rate': round(auth_stats['successful'] / max(1, auth_stats['total_attempts']) * 100, 1),
        'hourly_stats': dict(recent_hours),
        'enrolled_users': len(face_model.user_embeddings),
        'locked_users': len(lockout_manager.lockouts)
    })


@app.route('/api/logs', methods=['GET'])
def api_get_logs():
    """Get audit logs."""
    user_id = request.args.get('user_id')
    limit = int(request.args.get('limit', 100))
    logs = db.get_audit_logs(user_id, limit)
    return jsonify({'success': True, 'logs': logs})


@app.route('/api/alerts', methods=['GET'])
def api_get_alerts():
    """Get recent security alerts."""
    limit = int(request.args.get('limit', 50))
    alerts = alert_manager.get_recent_alerts(limit)
    return jsonify({'success': True, 'alerts': alerts})


# ============== WEBHOOK MANAGEMENT ==============

@app.route('/api/webhooks', methods=['GET'])
def api_get_webhooks():
    """Get configured webhooks."""
    webhooks = config.get('alerts', {}).get('webhooks', [])
    # Hide sensitive headers
    safe_webhooks = []
    for wh in webhooks:
        safe_wh = {k: v for k, v in wh.items() if k != 'headers'}
        safe_wh['has_headers'] = bool(wh.get('headers'))
        safe_webhooks.append(safe_wh)
    return jsonify({'success': True, 'webhooks': safe_webhooks})


@app.route('/api/webhooks/test', methods=['POST'])
def api_test_webhook():
    """Test a webhook."""
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    
    try:
        import requests
        response = requests.post(url, json={
            'event': 'test',
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'Webhook test from Biometric Security System'
        }, timeout=10)
        
        return jsonify({
            'success': True,
            'status_code': response.status_code,
            'response': response.text[:500]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============== SYSTEM STATUS ==============

@app.route('/api/status', methods=['GET'])
def api_system_status():
    """Get system status."""
    return jsonify({
        'success': True,
        'status': 'online',
        'models': {
            'face': face_model.face_cascade is not None,
            'speaker': True,
            'speech': speech_model.is_available(),
            'gesture': gesture_model.is_available()
        },
        'enrolled_users': len(face_model.user_embeddings),
        'database': 'connected',
        'after_hours': is_after_hours()
    })


if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("🔐 Biometric Security System")
    print(f"{'='*50}")
    print(f"Face Model: {len(face_model.user_embeddings)} users loaded")
    print(f"Speaker Model: {len(speaker_model.user_embeddings)} users loaded")
    print(f"Speech Model: {'✓ Available' if speech_model.is_available() else '✗ Not available'}")
    print(f"Gesture Model: {'✓ Available' if gesture_model.is_available() else '✗ Not available'}")
    print(f"{'='*50}\n")
    
    app.run(
        host=config['app']['host'],
        port=config['app']['port'],
        debug=config['app']['debug']
    )
