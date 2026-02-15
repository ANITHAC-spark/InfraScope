# InfraScope Setup Guide

## Complete Installation & Running Instructions

### Prerequisites
- **Python 3.8+** (Download from python.org)
- **pip** (comes with Python)
- **Modern Web Browser** (Chrome, Firefox, Edge, Safari)
- **Git** (optional, for cloning)
- **Text Editor/IDE** (VS Code recommended)

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 2GB for models + dependencies
- **Internet**: Required for initial setup
- **OS**: Windows, macOS, or Linux

---

## STEP-BY-STEP SETUP

### 1️⃣ Setup Backend (Python)

#### 1.1 Create Virtual Environment
```bash
# Navigate to backend folder
cd INFRA_SCOPE/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 1.2 Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# This will install:
# - Flask (REST API)
# - TensorFlow (CNN)
# - Scikit-learn (SVM, KNN)
# - OpenCV (Image processing)
# - Other utilities
```

**Installation time**: ~5-10 minutes (depends on internet speed)

#### 1.3 Verify Installation
```bash
# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Check other libraries
python -c "import sklearn; import cv2; print('All good!')"
```

### 2️⃣ Run Backend Server

```bash
# From backend folder with activated venv
python app.py
```

**Expected output:**
```
==================================================
InfraScope Backend Server
==================================================
Starting Flask server on http://localhost:5000
==================================================
 * Running on http://localhost:5000
```

✅ **Backend is ready** when you see above message

---

### 3️⃣ Run Frontend

#### Option A: Using VS Code Live Server (Recommended)
1. Open the project in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"
4. Browser will automatically open

#### Option B: Using Python HTTP Server
```bash
# From project root
python -m http.server 8000

# Then visit: http://localhost:8000
```

#### Option C: Direct File Opening
```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

---

## USAGE GUIDE

### 🖼️ Upload an Image
1. Click "Upload Infrastructure Image" card
2. Drag & drop an image OR click to browse
3. Supported formats: JPG, PNG, WebP
4. Maximum file size: 25 MB

### 📊 View Results
After upload, you'll see:
- **Image Preview**: Your uploaded image
- **CNN Results**: Defect type, confidence, severity
- **SVM Results**: Alternative model prediction
- **KNN Results**: Third model for comparison
- **Prediction Time**: How long analysis took
- **Critical Alert**: If severity is high

### 📈 Analytics Dashboard
- Navigate to "Analytics" tab
- View performance metrics for each model
- Compare accuracy, precision, recall, F1-score
- See response time comparison

### 📋 Inspection History
- Navigate to "History" tab
- Search by filename
- Filter by severity level (Low, Medium, Critical)
- Download inspection records

### ⚙️ Settings
- Enable/disable individual models
- Configure alert thresholds
- Monitor system status
- View database info

---

## API ENDPOINTS (For Developers)

### Health Check
```
GET http://localhost:5000/health
```

### Make Prediction
```
POST http://localhost:5000/predict
Content-Type: multipart/form-data

Body:
- image: [binary image file]
- models: {"cnn": true, "svm": true, "knn": true}
```

### Get Prediction History
```
GET http://localhost:5000/history
```

### Get System Statistics
```
GET http://localhost:5000/stats
```

### Get Performance Metrics
```
GET http://localhost:5000/performance-metrics
```

---

## TROUBLESHOOTING

### Issue: Backend not starting

**Error**: "Address already in use"
```bash
# Kill process on port 5000
# Windows
netstat -ano | findstr :5000
taskkill /PID [PID] /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

**Error**: Dependency installation failed
```bash
# Try updating pip first
python -m pip install --upgrade pip

# Then retry
pip install -r requirements.txt --upgrade
```

### Issue: Frontend can't connect to backend

**Problem**: CORS error in browser console

**Solution**:
1. Verify backend is running on port 5000
2. Check firewall isn't blocking port 5000
3. Ensure both frontend and backend are running
4. Refresh browser (Ctrl+Shift+R)

### Issue: Image upload failing

**Problem**: "Invalid file format"

**Solution**:
- Use proper image format (JPG, PNG, WebP)
- Ensure file size < 25 MB
- Check file extension is correct

**Problem**: "No file provided"

**Solution**:
- Ensure file is selected
- Try different image
- Check file isn't corrupted

### Issue: Models not loading

**Problem**: "ModuleNotFoundError"

**Solution**:
```bash
# Ensure all dependencies installed
pip install -r requirements.txt -v

# Check specific package
pip show tensorflow
```

**Problem**: "File not found" (models)

**Solution**:
- Models auto-generate on first run
- Or download pre-trained weights
- Check `models/` directory exists

---

## PERFORMANCE OPTIMIZATION

### To speed up predictions:
```python
# In app.py, use GPU:
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### For production deployment:
```bash
# Use Gunicorn
pip install gunicorn
gunicorn -w 4 -b localhost:5000 app:app

# Use Nginx as reverse proxy
# Configuration: (see deployment guide)
```

---

## DATABASE

### View Prediction History
```
Location: data/predictions.json
```

### Database Operations
```python
from backend.database import db

# Get statistics
stats = db.get_statistics()

# Get predictions
predictions = db.get_predictions(limit=10)

# Export to JSON
db.export_predictions_json()
```

---

## ADVANCED CONFIGURATION

### Training Custom Models

```python
# Save your training data in:
# data/training_images/
# data/training_labels.csv

# Run training:
python train_custom_models.py
```

### Adjusting Model Parameters

Edit `backend/train_models.py`:
- CNN layers: Modify architecture
- SVM kernel: Change RBF to linear/poly
- KNN neighbors: Adjust k value (default=5)

### Database Backup

```bash
# Backup database
cp data/infra_scope.db data/infra_scope_backup.db

# Export predictions
python -c "from backend.database import db; db.export_predictions_json()"
```

---

## MONITORING

### System Logs

```bash
# View logs
tail -f logs/app.log

# Clear logs
> logs/app.log
```

### Performance Metrics

```bash
# Check CPU/RAM usage
# Windows: Task Manager
# macOS: Activity Monitor
# Linux: top or htop
```

---

## DEPLOYMENT

### Docker (Recommended)

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r backend/requirements.txt
CMD ["python", "backend/app.py"]
```

### Heroku

```bash
heroku login
heroku create infrascope-app
git push heroku main
```

### AWS/Azure

See `deployment/cloud-setup.md` for detailed instructions

---

## FILE STRUCTURE AFTER RUNNING

```
INFRA_SCOPE/
├── frontend/
│   └── [HTML/CSS/JS files]
├── backend/
│   ├── app.py (running)
│   ├── models.py
│   └── [other backend files]
├── models/ (auto-created)
│   ├── cnn_model.h5
│   ├── svm_model.pkl
│   └── knn_model.pkl
├── uploads/ (auto-created)
│   └── [user uploaded images]
├── data/ (auto-created)
│   ├── predictions.json
│   └── infra_scope.db
└── logs/ (auto-created)
    └── app.log
```

---

## QUICK START COMMANDS

```bash
# Complete setup (Windows)
cd INFRA_SCOPE
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

# In another terminal:
cd .. 
python -m http.server 8000

# Visit: http://localhost:8000
```

---

## NEXT STEPS

1. ✅ Setup complete! Open http://localhost:8000
2. 📤 Upload your first infrastructure image
3. 📊 View AI predictions from 3 models
4. 📈 Check analytics dashboard
5. 💾 Review inspection history
6. ⚙️ Configure settings as needed

---

## SUPPORT

### Getting Help
- Check README.md for detailed documentation
- Review API endpoints in app.py comments
- Check browser console for errors (F12)
- Verify all services running properly

### Common Issues
- Backend not running? → Check port 5000
- CORS errors? → Backend must be running
- Images not processing? → Check image format
- Slow predictions? → Close other applications

---

## VERSION INFO

- **Version**: 1.0.0
- **Python**: 3.8+
- **TensorFlow**: 2.14.0
- **Flask**: 3.0.0
- **Last Updated**: February 2026

---

**Happy Infrastructure Inspection! 🚀**
