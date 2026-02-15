import os
from datetime import timedelta

# Server Configuration
DEBUG = True
HOST = 'localhost'
PORT = 5000

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

# Database Configuration
DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'infra_scope.db')
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')

# Model Configuration
MODELS_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'models')
CNN_MODEL_PATH = os.path.join(MODELS_FOLDER, 'cnn_model.h5')
SVM_MODEL_PATH = os.path.join(MODELS_FOLDER, 'svm_model.pkl')
KNN_MODEL_PATH = os.path.join(MODELS_FOLDER, 'knn_model.pkl')

# Image Processing
IMAGE_SIZE = (224, 224)
NORMALIZE_PIXEL = True
APPLY_BLUR = True
BLUR_KERNEL = (5, 5)

# Model Prediction
CONFIDENCE_THRESHOLD = 0.5
SEVERITY_LEVELS = {
    'critic': 0.9,
    'medium': 0.7,
    'low': 0.0
}

# CORS Configuration
CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8000', 'file://']

# Cache Configuration
CACHE_PREDICTIONS = True
CACHE_TTL = timedelta(hours=24)

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'logs', 'app.log')

# Performance
BATCH_SIZE = 32
NUM_WORKERS = 4

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
