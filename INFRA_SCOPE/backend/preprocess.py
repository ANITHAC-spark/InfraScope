import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    """
    Preprocess image for model inference
    Steps:
    1. Resize to 224x224
    2. Normalize pixel values
    3. Apply Gaussian blur for noise reduction
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            # Try PIL if cv2 fails
            img = Image.open(image_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Resize to 224x224
        img_resized = cv2.resize(img, (224, 224))
        
        # Apply Gaussian blur for noise reduction
        img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_blurred.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor((img_normalized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        
        return img_rgb
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        # Return dummy array if preprocessing fails
        return np.zeros((224, 224, 3), dtype=np.float32)

def augment_image(img_array):
    """
    Data augmentation for training
    """
    # Random rotation
    angle = np.random.uniform(-15, 15)
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotated = cv2.warpAffine(img_array, matrix, (w, h))
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    img_bright = np.clip(img_rotated * brightness, 0, 1)
    
    return img_bright

def extract_handcrafted_features(img_array):
    """
    Extract handcrafted features for SVM/KNN
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        img_gray = img_array
    
    features = []
    
    # 1. Edge detection (Canny)
    edges = cv2.Canny((img_gray * 255).astype(np.uint8), 100, 200)
    edge_density = np.sum(edges) / edges.size
    features.append(edge_density)
    
    # 2. Contrast
    contrast = np.std(img_gray)
    features.append(contrast)
    
    # 3. Mean brightness
    brightness = np.mean(img_gray)
    features.append(brightness)
    
    # 4. Texture (using LABp)
    laplacian = cv2.Laplacian((img_gray * 255).astype(np.uint8), cv2.CV_64F)
    texture = np.std(laplacian) / 255.0
    features.append(texture)
    
    # 5. Histogram
    hist = cv2.calcHist([img_gray], [0], None, [32], [0, 1])
    hist = hist.flatten() / (hist.max() + 1e-6)
    features.extend(hist[:5])  # First 5 bins
    
    # 6. HSV features (if color image)
    if len(img_array.shape) == 3:
        hsv = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        features.append(np.mean(hsv[:, :, 0]))  # Hue mean
        features.append(np.mean(hsv[:, :, 1]))  # Saturation mean
        features.append(np.std(hsv[:, :, 2]))   # Value std
    
    return np.array(features, dtype=np.float32)
