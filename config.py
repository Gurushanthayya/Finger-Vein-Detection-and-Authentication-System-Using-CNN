import os
from pathlib import Path

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'veinguard-secret-key-1298471')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max upload size
    
    # Path settings
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset', 'Finger Vein Database')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Database configuration
    # Can be 'sqlite' or 'mysql'
    DB_TYPE = os.environ.get('DB_TYPE', 'sqlite').lower()
    SQLITE_PATH = os.path.join(BASE_DIR, 'database', 'veinid_db.sqlite')
    
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'veinid_db')
    
    # Biometric Threshold
    # Scores >= threshold will be accepted as matches.
    # Recommended: Cosine similarity >= 0.82
    THRESHOLD = 0.82
