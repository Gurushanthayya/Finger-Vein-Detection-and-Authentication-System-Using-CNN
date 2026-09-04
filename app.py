"""
VeinGuard — Finger Vein Biometric Authentication System
Flask backend web application mapping routes to database and verification pipelines.
"""
import os
import time
import cv2
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

from config import Config
from database.db import DatabaseManager
from preprocessing.preprocess import get_visual_stages
from authentication.verify import FingerVeinFeatureExtractor

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Update dataset folder path to look at training split folder
app.config['DATASET_FOLDER'] = os.path.join(Config.BASE_DIR, 'dataset', 'train')

# Ensure upload folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize singletons
extractor = FingerVeinFeatureExtractor()
db = DatabaseManager()


def auto_enroll_sample_users(num_subjects=5):
    """
    Scans the dataset/train directory on startup. If database is empty, auto-enrolls 
    a small cohort of subjects so the system starts with demo users.
    """
    dataset_dir = Path(app.config['DATASET_FOLDER'])
    if not dataset_dir.exists():
        logger.warning(f"[App] Dataset directory not found at {dataset_dir}. Skipping auto-enrollment.")
        return 0

    existing_users = db.get_all()
    if len(existing_users) > 0:
        logger.info(f"[App] Database already has {len(existing_users)} enrolled users. Skipping auto-enrollment.")
        return 0

    existing_ids = {u['user_id'] for u in existing_users}
    subject_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    enrolled_count = 0

    print("[App] Enrolling sample users from dataset...")
    for subject_dir in subject_dirs:
        if enrolled_count >= num_subjects:
            break
            
        subject_id = subject_dir.name
        for side in ['left', 'right']:
            side_dir = subject_dir / side
            if not side_dir.exists():
                continue
                
            files = [f for f in side_dir.glob('*') if f.suffix.lower() in ('.jpg', '.png', '.bmp')]
            
            # Group by finger type (e.g. index, middle, ring)
            finger_groups = {}
            for f in files:
                finger_name = f.stem.split('_')[0]
                finger_groups.setdefault(finger_name, []).append(f)
                
            for finger_name, img_paths in finger_groups.items():
                user_id = f"{subject_id}_{side}_{finger_name}"
                if user_id in existing_ids:
                    continue
                    
                features_list = []
                # Use first 3 images for template enrollment
                for p in sorted(img_paths)[:3]:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    try:
                        features_list.append(extractor.extract_features(img))
                    except Exception as e:
                        logger.warning(f"[App] Failed to extract features from {p.name}: {e}")
                        
                if features_list:
                    name = f"Subject {subject_id} {side.capitalize()} {finger_name.capitalize()}"
                    db.register(name, user_id, features_list)
                    logger.info(f"[App] Enrolled sample user: {name}")
                    
        enrolled_count += 1
        
    return enrolled_count


# Auto-enroll demo users on module load if database is fresh
try:
    auto_enroll_sample_users(num_subjects=5)
except Exception as _e:
    logger.error(f"[App] Startup enrollment error: {_e}")



# ─── HTTP ROUTES ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify({'users': db.get_all()})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    users = db.get_all()
    logs = db.get_auth_logs(limit=100)
    accepts = [l for l in logs if l['result'] == 'accept']
    rejects = [l for l in logs if l['result'] == 'reject']
    
    avg_conf = 0.0
    if accepts:
        avg_conf = round(sum(l['confidence'] for l in accepts) / len(accepts) * 100, 1)
        
    return jsonify({
        'total_users': len(users),
        'total_auths': len(logs),
        'accepted': len(accepts),
        'rejected': len(rejects),
        'avg_confidence': avg_conf,
        'model_type': 'CNN' if extractor.use_cnn else 'Handcrafted Gabor/HOG',
    })


@app.route('/api/register', methods=['POST'])
def register():
    name = request.form.get('name', '').strip()
    user_id = request.form.get('user_id', '').strip()
    
    if not name or not user_id:
        return jsonify({'error': 'Name and User ID are required'}), 400
        
    if any(u['user_id'] == user_id for u in db.get_all()):
        return jsonify({'error': 'User ID already exists. Please choose a different ID.'}), 409
        
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'error': 'Please upload at least one valid finger vein image'}), 400

    features_list = []
    first_img_stages = None
    
    for idx, f in enumerate(files):
        try:
            img_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            features_list.append(extractor.extract_features(img))
            
            # Save visual stages for the very first image
            if idx == 0:
                first_img_stages = get_visual_stages(img)
        except Exception as e:
            logger.error(f"[App] Error processing image during registration: {e}")

    if not features_list:
        return jsonify({'error': 'Could not process or extract features from any of the uploaded images'}), 400

    user_info = db.register(name, user_id, features_list)
    
    return jsonify({
        'success': True,
        'uid': user_info['uid'],
        'name': name,
        'samples': len(features_list),
        'stages': first_img_stages,
        'message': f"Successfully enrolled {name} with {len(features_list)} vein sample(s)."
    })


@app.route('/api/authenticate', methods=['POST'])
def authenticate():
    if 'image' not in request.files:
        return jsonify({'error': 'No authentication image provided'}), 400
        
    f = request.files['image']
    try:
        img_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Invalid image file format'}), 400
    except Exception as e:
        return jsonify({'error': f"Failed to read image: {e}"}), 400

    start_time = time.time()
    try:
        query_feat = extractor.extract_features(img)
        visual_stages = get_visual_stages(img)
    except Exception as e:
        return jsonify({'error': f"Feature extraction pipeline failed: {e}"}), 500

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    
    # Authenticate matching features against enrolled templates
    result = db.authenticate(query_feat, ip_address=request.remote_addr)
    confidence_score = round(result['score'] * 100, 2)

    return jsonify({
        'authenticated': result['match'],
        'uid': result.get('matched_uid'),
        'name': result.get('matched_user'),
        'confidence': confidence_score,
        'processing_time_ms': elapsed_ms,
        'stages': visual_stages,
        'message': f"Identity verified — {result['matched_user']}" if result['match']
                   else "Authentication failed — No matching vein pattern found."
    })


@app.route('/api/demo-authenticate/<path:subject>/<filename>')
def demo_authenticate(subject, filename):
    """
    Helper API for selecting a dataset sample image on UI and matching it immediately.
    """
    img_path = Path(app.config['DATASET_FOLDER']) / subject / filename
    if not img_path.exists():
        return jsonify({'error': 'Sample image not found'}), 404
        
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'Could not read sample image'}), 500
        
    start_time = time.time()
    query_feat = extractor.extract_features(img)
    visual_stages = get_visual_stages(img)
    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    
    result = db.authenticate(query_feat)
    confidence_score = round(result['score'] * 100, 2)
    
    return jsonify({
        'authenticated': result['match'],
        'uid': result.get('matched_uid'),
        'name': result.get('matched_user'),
        'confidence': confidence_score,
        'processing_time_ms': elapsed_ms,
        'stages': visual_stages,
    })


@app.route('/api/dataset-list')
def dataset_list():
    """
    Returns lists of available subject images for demo selection on the UI dashboard.
    """
    dataset_dir = Path(app.config['DATASET_FOLDER'])
    result = {}
    if not dataset_dir.exists():
        return jsonify(result)
        
    # Return first 10 subject directories to keep dataset menu simple
    for subject_dir in sorted(dataset_dir.iterdir())[:10]:
        if not subject_dir.is_dir():
            continue
        for side in ['left', 'right']:
            side_dir = subject_dir / side
            if side_dir.exists():
                images = sorted(f.name for f in side_dir.glob('*') if f.suffix.lower() in ('.jpg', '.png', '.bmp'))
                if images:
                    result[f"{subject_dir.name}/{side}"] = images
    return jsonify(result)


@app.route('/api/dataset-sample/<path:subject>/<filename>')
def dataset_image(subject, filename):
    """Serves the raw image from the dataset for visualization"""
    return send_from_directory(os.path.join(app.config['DATASET_FOLDER'], subject), filename)


@app.route('/api/auth-logs')
def auth_logs():
    return jsonify(db.get_auth_logs(limit=30))


@app.route('/api/users/<uid>', methods=['DELETE'])
def delete_user(uid):
    success = db.delete_user(uid)
    if success:
        return jsonify({'success': True, 'message': 'User deleted successfully'})
    return jsonify({'error': 'User not found'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_debug = os.environ.get('FLASK_ENV', 'development').lower() == 'development'
    logger.info(f"[App] Starting server on port {port} (debug={is_debug})...")
    app.run(debug=is_debug, host='0.0.0.0', port=port)

