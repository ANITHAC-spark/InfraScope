"""
InfraScope Model Training Script
Demonstrates how to train CNN, SVM, and KNN models on infrastructure damage dataset
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from train_models import InfraScope
from preprocess import preprocess_image, extract_handcrafted_features

def generate_dummy_dataset(num_samples=100):
    """
    Generate dummy dataset for demonstration
    In production, use real infrastructure images
    """
    print(f"Generating {num_samples} dummy training samples...")
    
    X_train = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 3, num_samples)
    
    X_test = np.random.rand(20, 224, 224, 3).astype(np.float32)
    y_test = np.random.randint(0, 3, 20)
    
    return X_train, y_train, X_test, y_test

def train_models():
    """Train all models"""
    print("=" * 60)
    print("InfraScope Model Training")
    print("=" * 60)
    
    # Initialize models
    print("\n[1/4] Initializing models...")
    infra_scope = InfraScope()
    print("✓ Models initialized")
    
    # Generate or load dataset
    print("\n[2/4] Loading dataset...")
    print("      Note: Using dummy data. Replace with real infrastructure images.")
    print("      Dataset should contain crack, erosion, and no-damage images.")
    
    X_train, y_train, X_test, y_test = generate_dummy_dataset(num_samples=100)
    print(f"✓ Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"  Classes: 0=Crack, 1=Erosion, 2=No Damage")
    
    # Train models
    print("\n[3/4] Training models...")
    print("      This may take several minutes...")
    
    infra_scope.train_models(X_train, y_train, X_test, y_test)
    
    # Save models
    print("\n[4/4] Saving models...")
    infra_scope.save_models()
    print("✓ Models saved to models/ directory")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

def create_sample_dataset_directory():
    """Create directory structure for training data"""
    dataset_dir = "data/training_dataset"
    classes = ["crack", "erosion", "no_damage"]
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    print(f"\nPlace your training images in:")
    for class_name in classes:
        print(f"  - data/training_dataset/{class_name}/")

def demo_prediction():
    """Demonstrate prediction on a single image"""
    print("\n" + "=" * 60)
    print("Demo Prediction")
    print("=" * 60)
    
    # Create dummy image
    dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Initialize model
    infra_scope = InfraScope()
    
    # Make prediction
    print("\nMaking predictions on dummy image...")
    results = infra_scope.predict(dummy_image, {
        'cnn': True,
        'svm': True,
        'knn': True
    })
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 60)
    
    for model_name, prediction in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Defect Type: {prediction['defect_type']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"  Severity: {prediction['severity']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='InfraScope Model Training Script')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--demo', action='store_true', help='Run demo prediction')
    parser.add_argument('--setup-dirs', action='store_true', help='Create training dataset directories')
    parser.add_argument('--all', action='store_true', help='Run all (setup + train + demo)')
    
    args = parser.parse_args()
    
    try:
        if args.setup_dirs or args.all:
            print("\nSetting up directories...")
            create_sample_dataset_directory()
        
        if args.train or args.all:
            train_models()
        
        if args.demo or args.all:
            demo_prediction()
        
        if not any([args.train, args.demo, args.setup_dirs, args.all]):
            print("Usage: python train.py [--train] [--demo] [--setup-dirs] [--all]")
            print("\nExamples:")
            print("  Setup directories:  python train.py --setup-dirs")
            print("  Train models:       python train.py --train")
            print("  Demo prediction:    python train.py --demo")
            print("  Complete setup:     python train.py --all")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
