"""
InfraScope Backend Package
Smart Infrastructure Damage Detection System
"""

__version__ = "1.0.0"
__author__ = "InfraScope Team"

from app import app
from train_models import InfraScope
from database import Database
from preprocess import preprocess_image

__all__ = [
    'app',
    'InfraScope',
    'Database',
    'preprocess_image'
]
