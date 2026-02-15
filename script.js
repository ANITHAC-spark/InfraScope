// API Configuration
const API_URL = 'http://localhost:5000';

// State Management
const state = {
    inspectionHistory: [],
    modelSettings: {
        cnn: true,
        svm: true,
        knn: true
    },
    performanceMetrics: {
        cnn: { accuracy: 94.2, precision: 93.8, recall: 94.6, f1: 94.2, time: 245 },
        svm: { accuracy: 89.7, precision: 89.1, recall: 90.3, f1: 89.7, time: 156 },
        knn: { accuracy: 87.3, precision: 86.8, recall: 87.9, f1: 87.3, time: 189 }
    }
};

let performanceChart = null;

// Global Test Function for Debugging
function testRouting(route) {
    console.log(`Testing route: ${route}`);
    const sections = document.querySelectorAll('.content-section');
    console.log('All sections:', Array.from(sections).map(s => ({ id: s.id, active: s.classList.contains('active') })));
    
    window.location.hash = route;
    
    setTimeout(() => {
        console.log(`Current hash: ${window.location.hash}`);
        const activeSections = Array.from(document.querySelectorAll('.content-section')).filter(s => s.classList.contains('active'));
        console.log('Active sections after navigation:', activeSections.map(s => s.id));
        
        const activeNav = Array.from(document.querySelectorAll('.nav-item')).filter(n => n.classList.contains('active'));
        console.log('Active nav items:', activeNav.map(n => n.textContent.trim()));
    }, 500);
}

// Function to manually navigate to a route (useful for testing)
function manualNavigate(route) {
    console.log(`Manual navigation to: ${route}`);
    navigateToRoute(route);
}

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    console.log('=== InfraScope App Initializing ===');
    console.log('DOM Content Loaded Event Fired');
    
    try {
        console.log('Step 1: Initializing Event Listeners');
        initializeEventListeners();
        console.log('✓ Event Listeners Initialized');
        
        console.log('Step 2: Initializing Routing');
        initializeRouting();
        console.log('✓ Routing Initialized');
        
        console.log('Step 3: Starting Time Display');
        updateTime();
        setInterval(updateTime, 1000);
        console.log('✓ Time Display Started');
        
        console.log('Step 4: Checking Backend Connection');
        checkBackendConnection();
        console.log('✓ Backend Check Initiated');
        
        console.log('Step 5: Initializing Chart');
        initializeChart();
        console.log('✓ Chart Initialized');
        
        console.log('Step 6: Loading Inspection History');
        loadInspectionHistory();
        console.log('✓ History Loaded');
        
        console.log('=== App Initialization Complete ===');
    } catch (error) {
        console.error('Error during app initialization:', error);
    }
});

// Initialize Event Listeners
function initializeEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', handleNavigation);
    });

    // Analytics Tabs
    const tabButtons = document.querySelectorAll('.tab-btn');
    console.log('Found tab buttons:', tabButtons.length);
    tabButtons.forEach(btn => {
        console.log('Attaching click listener to tab:', btn.getAttribute('data-model'));
        btn.addEventListener('click', handleTabClick);
    });

    // Upload Area
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    // Settings
    const enableCNN = document.getElementById('enableCNN');
    const enableSVM = document.getElementById('enableSVM');
    const enableKNN = document.getElementById('enableKNN');
    
    if (enableCNN) enableCNN.addEventListener('change', (e) => {
        state.modelSettings.cnn = e.target.checked;
    });
    if (enableSVM) enableSVM.addEventListener('change', (e) => {
        state.modelSettings.svm = e.target.checked;
    });
    if (enableKNN) enableKNN.addEventListener('change', (e) => {
        state.modelSettings.knn = e.target.checked;
    });

    // History Filters
    const searchHistory = document.getElementById('searchHistory');
    const filterSeverity = document.getElementById('filterSeverity');
    if (searchHistory) searchHistory.addEventListener('input', filterHistory);
    if (filterSeverity) filterSeverity.addEventListener('change', filterHistory);

    // Sidebar Toggle
    document.querySelector('.sidebar-toggle').addEventListener('click', toggleSidebar);
}

// Handle Navigation - Simply update hash and let routing handle the rest
function handleNavigation(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const element = e.currentTarget;
    const route = element.getAttribute('data-route') || element.getAttribute('data-section');
    
    console.log('=== NAVIGATION CLICKED ===');
    console.log('Element:', element.textContent.trim());
    console.log('Route:', route);
    console.log('Current hash before:', window.location.hash);
    
    if (!route) {
        console.error('No route found on navigation element!');
        return;
    }
    
    // Update URL hash - this will trigger hashchange event
    window.location.hash = route;
    console.log('New hash after:', window.location.hash);
}

// Initialize Routing with Hash Navigation
function initializeRouting() {
    console.log('=== Initializing Routing ===');
    console.log('Current URL:', window.location.href);
    console.log('Current Hash:', window.location.hash);
    
    // Validate sections exist
    const sections = document.querySelectorAll('.content-section');
    console.log('✓ Found sections:', sections.length);
    sections.forEach(s => console.log('  -', s.id));
    
    // Validate nav items exist
    const navItems = document.querySelectorAll('.nav-item');
    console.log('✓ Found nav items:', navItems.length);
    navItems.forEach(n => console.log('  -', n.getAttribute('data-route'), ':', n.textContent.trim()));
    
    // Handle hash changes (back button support)
    window.addEventListener('hashchange', () => {
        const route = window.location.hash.substring(1) || 'dashboard';
        console.log('>>> HASH CHANGED EVENT FIRED <<<');
        console.log('New route from hash:', route);
        navigateToRoute(route);
    });
    
    console.log('✓ hashchange event listener attached');
    
    // Load initial route from hash or default to dashboard
    const initialRoute = window.location.hash.substring(1) || 'dashboard';
    console.log('Initial route:', initialRoute);
    navigateToRoute(initialRoute);
    
    console.log('✓ Routing initialization complete');
}

// Navigate to a specific route - Centralized routing logic
function navigateToRoute(route) {
    console.log('navigateToRoute called with:', route);
    
    // Map routes to sections and titles
    const routeMap = {
        'dashboard': { section: 'upload', title: 'Dashboard', subtitle: 'Upload & Predict Infrastructure Damage' },
        'analytics': { section: 'analytics', title: 'Analytics', subtitle: 'Model Performance & Metrics Analysis' },
        'history': { section: 'history', title: 'Inspection History', subtitle: 'View and Manage Past Inspections' },
        'model-performance': { section: 'model-performance', title: 'Model Performance', subtitle: 'View Model Training Performance Metrics' },
        'settings': { section: 'settings', title: 'Settings', subtitle: 'System Configuration & Preferences' }
    };
    
    const config = routeMap[route] || routeMap['dashboard'];
    const section = config.section;
    
    console.log('Route config:', config);
    
    // 1. Update header title
    const headerTitle = document.querySelector('.header-title h1');
    const headerSubtitle = document.querySelector('.header-title p');
    if (headerTitle) {
        headerTitle.textContent = config.title;
        console.log('Updated header title to:', config.title);
    }
    if (headerSubtitle) {
        headerSubtitle.textContent = config.subtitle;
    }
    
    // 2. Update browser tab title
    document.title = `InfraScope - ${config.title}`;
    
    // 3. Hide all sections
    const allSections = document.querySelectorAll('.content-section');
    console.log('Total sections found:', allSections.length);
    allSections.forEach(sec => {
        sec.classList.remove('active');
        console.log('Hiding section:', sec.id);
    });
    
    // 4. Show the target section
    const targetSection = document.getElementById(`${section}-section`);
    if (targetSection) {
        targetSection.classList.add('active');
        console.log('Showing section:', `${section}-section`);
    } else {
        console.error(`Section not found: ${section}-section`);
        // Check what sections exist
        allSections.forEach(sec => console.log('Available section:', sec.id));
    }
    
    // 5. Update active nav item
    const navItems = document.querySelectorAll('.nav-item');
    console.log('Total nav items found:', navItems.length);
    navItems.forEach(item => {
        const itemRoute = item.getAttribute('data-route');
        const itemSection = item.getAttribute('data-section');
        console.log('Checking nav item - route:', itemRoute, 'section:', itemSection);
        
        if (itemRoute === route || itemSection === section) {
            item.classList.add('active');
            console.log('Activated nav item');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Load analytics data if navigating to analytics
    if (route === 'analytics') {
        loadAnalyticsData();
    }
    
    // Load model performance data if navigating to model-performance
    if (route === 'model-performance') {
        loadModelPerformance();
    }
    
    console.log('=== Route Navigation Summary ===');
    console.log('Route:', route);
    console.log('Section ID:', `${section}-section`);
    console.log('Page Title:', config.title);
    const activeSections = Array.from(document.querySelectorAll('.content-section.active')).map(s => s.id);
    console.log('Active Sections:', activeSections);
    console.log('=== Navigation Complete ===');
}

// Toggle Sidebar
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.toggle('collapsed');
    } else {
        console.warn('Sidebar element not found');
    }
}

// Switch Analytics Tab
function switchAnalyticsTab(event, model) {
    event.preventDefault();
    event.stopPropagation();
    
    console.log('Switching to tab:', model);
    
    // Update active tab button
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.closest('.tab-btn').classList.add('active');

    // Show/hide tab content
    document.querySelectorAll('.analytics-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    const tabContent = document.getElementById(`${model}-tab`);
    if (tabContent) {
        tabContent.classList.add('active');
        console.log('Showing tab:', model);
        
        // Initialize charts for specific models
        setTimeout(() => {
            if (model === 'cnn') initializeCNNChart();
            else if (model === 'svm') initializeSVMChart();
            else if (model === 'knn') initializeKNNChart();
        }, 100);
    } else {
        console.error(`Tab content not found: ${model}-tab`);
    }
}

// Handle Analytics Tab Switching
function handleTabClick(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const model = e.currentTarget.getAttribute('data-model');
    console.log('Tab clicked:', model);
    
    // Update active tab button
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    e.currentTarget.classList.add('active');

    // Show/hide tab content
    document.querySelectorAll('.analytics-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    const tabContent = document.getElementById(`${model}-tab`);
    console.log('Tab content element:', tabContent);
    
    if (tabContent) {
        tabContent.classList.add('active');
        
        // Initialize charts for specific models when tab is opened
        if (model === 'cnn') {
            console.log('Initializing CNN chart');
            initializeCNNChart();
        }
        else if (model === 'svm') {
            console.log('Initializing SVM chart');
            initializeSVMChart();
        }
        else if (model === 'knn') {
            console.log('Initializing KNN chart');
            initializeKNNChart();
        }
    }
}

// Initialize Individual Model Charts
function initializeCNNChart() {
    const ctx = document.getElementById('cnnChart');
    if (!ctx) return;
    const m = state.performanceMetrics.cnn || {};
    const acc = m.accuracy || m.accuracy_score || 0;
    const prec = m.precision || 0;
    const rec = m.recall || 0;
    const f1 = m.f1 || m.f1_score || 0;

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'CNN Performance',
                data: [acc, prec, rec, f1],
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.15)',
                borderWidth: 2,
                pointBackgroundColor: '#00d4ff',
                pointBorderColor: '#fff',
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#a0aec0' },
                    grid: { color: 'rgba(0, 212, 255, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                }
            }
        }
    });
}

function initializeSVMChart() {
    const ctx = document.getElementById('svmChart');
    if (!ctx) return;
    const m = state.performanceMetrics.svm || {};
    const acc = m.accuracy || m.accuracy_score || 0;
    const prec = m.precision || 0;
    const rec = m.recall || 0;
    const f1 = m.f1 || m.f1_score || 0;

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'SVM Performance',
                data: [acc, prec, rec, f1],
                borderColor: '#ffa500',
                backgroundColor: 'rgba(255, 165, 0, 0.15)',
                borderWidth: 2,
                pointBackgroundColor: '#ffa500',
                pointBorderColor: '#fff',
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#a0aec0' },
                    grid: { color: 'rgba(255, 165, 0, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                }
            }
        }
    });
}

function initializeKNNChart() {
    const ctx = document.getElementById('knnChart');
    if (!ctx) return;
    const m = state.performanceMetrics.knn || {};
    const acc = m.accuracy || m.accuracy_score || 0;
    const prec = m.precision || 0;
    const rec = m.recall || 0;
    const f1 = m.f1 || m.f1_score || 0;

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'KNN Performance',
                data: [acc, prec, rec, f1],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.15)',
                borderWidth: 2,
                pointBackgroundColor: '#ff6b6b',
                pointBorderColor: '#fff',
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#a0aec0' },
                    grid: { color: 'rgba(255, 107, 107, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                }
            }
        }
    });
}

// Handle File Upload
async function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file', 'error');
        return;
    }

    if (file.size > 25 * 1024 * 1024) {
        showToast('File size must be less than 25 MB', 'error');
        return;
    }

    // Show loading spinner
    document.getElementById('loadingSpinner').style.display = 'flex';
    
    // Show that new analysis is starting
    showToast('📊 Analyzing uploaded image based on content...', 'info');

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);
    formData.append('models', JSON.stringify({
        cnn: state.modelSettings.cnn,
        svm: state.modelSettings.svm,
        knn: state.modelSettings.knn
    }));

    try {
        const startTime = performance.now();
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const endTime = performance.now();
        const predictionTime = Math.round(endTime - startTime);

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        displayResults(data, predictionTime, file.name);

    } catch (error) {
        console.error('Error:', error);
        showToast('Backend server is not running. Showing demo results.', 'info');
        displayDemoResults(predictionTime, file.name);
    } finally {
        document.getElementById('loadingSpinner').style.display = 'none';
    }
}

// Display Results
function displayResults(data, predictionTime, fileName) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';

    // Display summary at the top (why this prediction was made)
    let summaryHTML = '';
    if (data.summary) {
        summaryHTML = data.summary;
    } else {
        summaryHTML = 'Analysis based on uploaded image content';
    }
    
    const summaryEl = document.getElementById('predictionMode');
    if (summaryEl) {
        summaryEl.innerHTML = `
            <div style="border-left: 4px solid #2196F3; padding: 12px; background: #f0f4ff; border-radius: 5px;">
                <strong style="color: #1976D2;">📊 PREDICTION SUMMARY</strong><br>
                <span style="color: #333; line-height: 1.6;">${summaryHTML}</span><br>
                <small style="color: #666;">Mode: Image-Based Analysis | File: ${fileName}</small>
            </div>
        `;
    }

    // Display image analytics if available
    if (data.cnn && (data.cnn.edge_density !== undefined || data.cnn.contrast !== undefined)) {
        const analyticsEl = document.getElementById('imageAnalytics');
        if (analyticsEl) {
            let analyticsHTML = '<div style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 12px;">';
            analyticsHTML += '<strong>📊 Image Analysis Metrics (Why this prediction?):</strong><br>';
            if (data.cnn.edge_density !== undefined) {
                // Edge density is on 0-255 scale from Canny edge detection, normalize to 0-1
                const normalizedEdgeDensity = Math.min(1.0, data.cnn.edge_density / 255);
                analyticsHTML += `<strong>Edge Density:</strong> ${(normalizedEdgeDensity * 100).toFixed(1)}% `;
                if (normalizedEdgeDensity > 0.2) {
                    analyticsHTML += '<span style="color: #f44336;"><strong>(HIGH - Likely Defect)</strong></span><br>';
                } else if (normalizedEdgeDensity > 0.1) {
                    analyticsHTML += '<span style="color: #ff9800;"><strong>(MEDIUM - Possible Issue)</strong></span><br>';
                } else {
                    analyticsHTML += '<span style="color: #4CAF50;"><strong>(LOW - No Defect)</strong></span><br>';
                }
            }
            if (data.cnn.contrast !== undefined) {
                analyticsHTML += `<strong>Contrast:</strong> ${(data.cnn.contrast * 100).toFixed(1)}% `;
                if (data.cnn.contrast > 0.15) {
                    analyticsHTML += '<span style="color: #f44336;"><strong>(HIGH)</strong></span><br>';
                } else if (data.cnn.contrast > 0.08) {
                    analyticsHTML += '<span style="color: #ff9800;"><strong>(MEDIUM)</strong></span><br>';
                } else {
                    analyticsHTML += '<span style="color: #4CAF50;"><strong>(LOW)</strong></span><br>';
                }
            }
            analyticsHTML += '</div>';
            analyticsEl.innerHTML = analyticsHTML;
        }
    }

    // Update results
    if (data.cnn) {
        document.querySelector('.cnn-defect').textContent = data.cnn.defect_type;
        document.querySelector('.cnn-confidence').textContent = (data.cnn.confidence * 100).toFixed(1) + '%';
        document.querySelector('.cnn-severity').textContent = data.cnn.severity;
        document.querySelector('.cnn-fill').style.width = (data.cnn.confidence * 100) + '%';
    }

    if (data.svm) {
        document.querySelector('.svm-defect').textContent = data.svm.defect_type;
        document.querySelector('.svm-confidence').textContent = (data.svm.confidence * 100).toFixed(1) + '%';
        document.querySelector('.svm-severity').textContent = data.svm.severity;
        document.querySelector('.svm-fill').style.width = (data.svm.confidence * 100) + '%';
    }

    if (data.knn) {
        document.querySelector('.knn-defect').textContent = data.knn.defect_type;
        document.querySelector('.knn-confidence').textContent = (data.knn.confidence * 100).toFixed(1) + '%';
        document.querySelector('.knn-severity').textContent = data.knn.severity;
        document.querySelector('.knn-fill').style.width = (data.knn.confidence * 100) + '%';
    }

    // Update prediction time
    document.getElementById('predictionTime').textContent = `Prediction Time: ${predictionTime}ms`;

    // Check for critical alert
    let maxSeverity = 'Low';
    let bestModel = 'CNN';
    let bestConfidence = 0;

    if (data.cnn && data.cnn.confidence > bestConfidence) {
        bestConfidence = data.cnn.confidence;
        maxSeverity = data.cnn.severity;
        bestModel = 'CNN';
    }
    if (data.svm && data.svm.confidence > bestConfidence) {
        bestConfidence = data.svm.confidence;
        maxSeverity = data.svm.severity;
        bestModel = 'SVM';
    }
    if (data.knn && data.knn.confidence > bestConfidence) {
        bestConfidence = data.knn.confidence;
        maxSeverity = data.knn.severity;
        bestModel = 'KNN';
    }

    const alertSection = document.getElementById('alertSection');
    if (maxSeverity === 'Critical') {
        alertSection.style.display = 'flex';
        document.getElementById('alertMessage').textContent = `Critical damage detected! Immediate inspection required.`;
    } else {
        alertSection.style.display = 'none';
    }

    // Add to history
    const historyEntry = {
        timestamp: new Date().toLocaleString(),
        fileName: fileName,
        defectType: maxSeverity !== 'Low' ? 'Yes' : 'No Damage',
        bestModel: bestModel,
        confidence: (bestConfidence * 100).toFixed(1),
        severity: maxSeverity,
        time: predictionTime
    };

    state.inspectionHistory.unshift(historyEntry);
    updateHistoryTable();
    updateSystemStats();

    showToast('Prediction completed successfully!', 'success');
    
    // Load and display image-specific performance metrics
    loadImagePerformanceMetrics(fileName);
}

// Load image-specific performance metrics
async function loadImagePerformanceMetrics(fileName) {
    try {
        // Get the file input to re-upload for performance calculation
        const fileInput = document.getElementById('fileInput');
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('image', file);
        formData.append('models', JSON.stringify({
            cnn: state.modelSettings.cnn,
            svm: state.modelSettings.svm,
            knn: state.modelSettings.knn
        }));

        const response = await fetch(`${API_URL}/image-performance`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            console.warn('Could not load image performance metrics');
            return;
        }

        const perfData = await response.json();
        displayImagePerformanceMetrics(perfData);

    } catch (error) {
        console.error('Error loading image performance metrics:', error);
    }
}

// Display image-specific performance metrics
function displayImagePerformanceMetrics(perfData) {
    const perfContainer = document.getElementById('imagePerformanceContainer');
    if (!perfContainer) {
        // Create container if it doesn't exist
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            const container = document.createElement('div');
            container.id = 'imagePerformanceContainer';
            container.style.marginTop = '20px';
            resultsSection.appendChild(container);
        }
        return;
    }

    let html = '<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white; margin-top: 15px;">';
    html += '<h3 style="margin-top: 0; color: white;">📊 Model Performance on This Image</h3>';

    if (perfData.model_metrics) {
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 10px;">';

        for (const [model, metrics] of Object.entries(perfData.model_metrics)) {
            html += `
                <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.3);">
                    <strong style="font-size: 14px; text-transform: uppercase;">${model}</strong>
                    <div style="margin-top: 8px; font-size: 12px;">
                        <div>Confidence: <strong>${(metrics.confidence * 100).toFixed(1)}%</strong></div>
                        <div>Accuracy: <strong>${(metrics.accuracy * 100).toFixed(1)}%</strong></div>
                        <div>Precision: <strong>${(metrics.precision * 100).toFixed(1)}%</strong></div>
                        <div>Recall: <strong>${(metrics.recall * 100).toFixed(1)}%</strong></div>
                        <div>F1-Score: <strong>${(metrics.f1_score * 100).toFixed(1)}%</strong></div>
                    </div>
                </div>
            `;
        }

        html += '</div>';
    }

    if (perfData.image_analysis) {
        html += `
            <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 6px;">
                <strong style="font-size: 12px;">Image Analysis Features:</strong><br>
                <small>Edge Density: ${(perfData.image_analysis.edge_density || 0).toFixed(4)} | Contrast: ${(perfData.image_analysis.contrast || 0).toFixed(4)}</small>
            </div>
        `;
    }

    if (perfData.plot_url) {
        html += `<div style="margin-top: 15px; text-align: center;">
                    <small style="color: rgba(255,255,255,0.8);">Performance plot generated</small>
                </div>`;
    }

    html += '</div>';

    const container = document.getElementById('imagePerformanceContainer');
    if (container) {
        container.innerHTML = html;
    }
}

function displayDemoResults(predictionTime, fileName) {
    document.getElementById('resultsSection').style.display = 'block';

    const demoResults = {
        cnn: { defect_type: 'Crack', confidence: 0.94, severity: 'Medium' },
        svm: { defect_type: 'Crack', confidence: 0.89, severity: 'Low' },
        knn: { defect_type: 'Crack', confidence: 0.87, severity: 'Low' }
    };

    displayResults(demoResults, predictionTime, fileName);
}

// Initialize Chart
function initializeChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    const c = state.performanceMetrics.cnn || {};
    const s = state.performanceMetrics.svm || {};
    const k = state.performanceMetrics.knn || {};

    const cnnData = [c.accuracy || c.accuracy_score || 0, c.precision || 0, c.recall || 0, c.f1 || c.f1_score || 0];
    const svmData = [s.accuracy || s.accuracy_score || 0, s.precision || 0, s.recall || 0, s.f1 || s.f1_score || 0];
    const knnData = [k.accuracy || k.accuracy_score || 0, k.precision || 0, k.recall || 0, k.f1 || k.f1_score || 0];

    performanceChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [
                {
                    label: 'CNN',
                    data: cnnData,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.15)',
                    borderWidth: 2.5,
                    pointBackgroundColor: '#00d4ff',
                    pointBorderColor: '#ffffff',
                    pointRadius: 5,
                    pointHoverRadius: 6
                },
                {
                    label: 'SVM',
                    data: svmData,
                    borderColor: '#ffa500',
                    backgroundColor: 'rgba(255, 165, 0, 0.15)',
                    borderWidth: 2.5,
                    pointBackgroundColor: '#ffa500',
                    pointBorderColor: '#ffffff',
                    pointRadius: 5,
                    pointHoverRadius: 6
                },
                {
                    label: 'KNN',
                    data: knnData,
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.15)',
                    borderWidth: 2.5,
                    pointBackgroundColor: '#ff6b6b',
                    pointBorderColor: '#ffffff',
                    pointRadius: 5,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#a0aec0',
                        font: { size: 12, weight: 'bold' },
                        padding: 20
                    }
                }
            },
            scales: {
                r: {
                    grid: { color: 'rgba(0, 212, 255, 0.1)' },
                    ticks: { color: '#a0aec0', backdropColor: 'transparent' },
                    max: 100
                }
            }
        }
    });
}

// Update History Table
function updateHistoryTable() {
    const tbody = document.getElementById('historyTableBody');
    tbody.innerHTML = '';

    if (state.inspectionHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="no-data">No inspection records yet.</td></tr>';
        return;
    }

    state.inspectionHistory.forEach(entry => {
        const row = `
            <tr>
                <td>${entry.timestamp}</td>
                <td>${entry.fileName}</td>
                <td>${entry.defectType}</td>
                <td>${entry.bestModel}</td>
                <td>${entry.confidence}%</td>
                <td><span class="severity-${entry.severity.toLowerCase()}">${entry.severity}</span></td>
                <td>${entry.time}ms</td>
                <td><button class="action-btn">View</button></td>
            </tr>
        `;
        tbody.innerHTML += row;
    });
}

// Filter History
function filterHistory() {
    const searchTerm = document.getElementById('searchHistory').value.toLowerCase();
    const severityFilter = document.getElementById('filterSeverity').value;

    document.querySelectorAll('.history-table tbody tr').forEach(row => {
        const fileName = row.cells[1]?.textContent.toLowerCase() || '';
        const severity = row.cells[5]?.textContent || '';

        let visible = true;

        if (searchTerm && !fileName.includes(searchTerm)) {
            visible = false;
        }

        if (severityFilter && severity !== severityFilter) {
            visible = false;
        }

        row.style.display = visible ? '' : 'none';
    });
}

// Load Inspection History
function loadInspectionHistory() {
    updateHistoryTable();
    updateSystemStats();
}

// Update System Stats
function updateSystemStats() {
    document.getElementById('imageCount').textContent = state.inspectionHistory.length;
    document.getElementById('predictionCount').textContent = state.inspectionHistory.length;
}

// Update Time Display
function updateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true
    });
    const dateString = now.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
    
    document.getElementById('timeDisplay').textContent = `${timeString} | ${dateString}`;
}

// Check Backend Connection
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            document.getElementById('backendStatus').textContent = 'Online';
            document.getElementById('backendStatus').className = 'status-online';
            document.getElementById('statusIndicator').style.background = '#10b981';
            // Fetch performance metrics when backend is online
            try {
                const metricsRes = await fetch(`${API_URL}/performance-metrics`);
                if (metricsRes.ok) {
                    const json = await metricsRes.json();
                    if (json && json.metrics) {
                        state.performanceMetrics = json.metrics;
                        // Reinitialize chart with live metrics
                        if (performanceChart) {
                            performanceChart.destroy();
                        }
                        initializeChart();
                    }
                }
            } catch (err) {
                console.warn('Could not fetch performance metrics:', err);
            }
        }
    } catch (error) {
        document.getElementById('backendStatus').textContent = 'Offline';
        document.getElementById('backendStatus').className = 'status-offline';
        document.getElementById('statusIndicator').style.background = '#ef4444';
    }
}

// Load Analytics Data from Backend
async function loadAnalyticsData() {
    console.log('Loading analytics data from backend...');
    try {
        // Fetch performance metrics
        const metricsRes = await fetch(`${API_URL}/performance-metrics`);
        if (metricsRes.ok) {
            const metricsData = await metricsRes.json();
            if (metricsData && metricsData.metrics) {
                state.performanceMetrics = metricsData.metrics;
                console.log('Performance metrics loaded:', metricsData.metrics);
                updateAnalyticsDisplay();
            }
        }
        
        // Fetch system stats
        const statsRes = await fetch(`${API_URL}/stats`);
        if (statsRes.ok) {
            const statsData = await statsRes.json();
            if (statsData && statsData.success) {
                console.log('Stats loaded:', statsData);
                updateStatsDisplay(statsData);
            }
        }
    } catch (error) {
        console.error('Error loading analytics data:', error);
        // Use default metrics if backend is unavailable
        console.log('Using default metrics');
        updateAnalyticsDisplay();
    }
}

// Update Analytics Display with Live Data
function updateAnalyticsDisplay() {
    const metrics = state.performanceMetrics;
    
    // Update overview tab summary metrics
    const cnnAccuracy = metrics.cnn?.accuracy || metrics.cnn?.accuracy_score || 94.2;
    const svmAccuracy = metrics.svm?.accuracy || metrics.svm?.accuracy_score || 89.7;
    const knnAccuracy = metrics.knn?.accuracy || metrics.knn?.accuracy_score || 87.3;
    
    // Update quick summary cards
    const cnnValue = document.querySelector('[style*="color: #00d4ff"]');
    const svmValue = document.querySelector('[style*="color: #ffa500"]');
    const knnValue = document.querySelector('[style*="color: #ff6b6b"]');
    
    if (cnnValue) cnnValue.textContent = cnnAccuracy.toFixed(1) + '%';
    if (svmValue) svmValue.textContent = svmAccuracy.toFixed(1) + '%';
    if (knnValue) knnValue.textContent = knnAccuracy.toFixed(1) + '%';
    
    // Update detailed metrics table
    updateMetricsTable();
    
    // Reinitialize charts with new data
    if (performanceChart) {
        performanceChart.destroy();
        performanceChart = null;
    }
    initializeChart();
}

// Update Metrics Table with Backend Data
function updateMetricsTable() {
    const metrics = state.performanceMetrics;
    
    // Update CNN row
    const cnnMetrics = metrics.cnn || {};
    document.getElementById('cnn-accuracy').textContent = ((cnnMetrics.accuracy || cnnMetrics.accuracy_score || 94.2) * 1).toFixed(1) + '%';
    document.getElementById('cnn-precision').textContent = ((cnnMetrics.precision || 93.8) * 1).toFixed(1) + '%';
    document.getElementById('cnn-recall').textContent = ((cnnMetrics.recall || 94.6) * 1).toFixed(1) + '%';
    document.getElementById('cnn-f1').textContent = ((cnnMetrics.f1_score || cnnMetrics.f1 || 94.2) * 1).toFixed(1) + '%';
    document.getElementById('cnn-time').textContent = (cnnMetrics.avg_response_time || 245) + 'ms';
    
    // Update SVM row
    const svmMetrics = metrics.svm || {};
    document.getElementById('svm-accuracy').textContent = ((svmMetrics.accuracy || svmMetrics.accuracy_score || 89.7) * 1).toFixed(1) + '%';
    document.getElementById('svm-precision').textContent = ((svmMetrics.precision || 89.1) * 1).toFixed(1) + '%';
    document.getElementById('svm-recall').textContent = ((svmMetrics.recall || 90.3) * 1).toFixed(1) + '%';
    document.getElementById('svm-f1').textContent = ((svmMetrics.f1_score || svmMetrics.f1 || 89.7) * 1).toFixed(1) + '%';
    document.getElementById('svm-time').textContent = (svmMetrics.avg_response_time || 156) + 'ms';
    
    // Update KNN row
    const knnMetrics = metrics.knn || {};
    document.getElementById('knn-accuracy').textContent = ((knnMetrics.accuracy || knnMetrics.accuracy_score || 87.3) * 1).toFixed(1) + '%';
    document.getElementById('knn-precision').textContent = ((knnMetrics.precision || 86.8) * 1).toFixed(1) + '%';
    document.getElementById('knn-recall').textContent = ((knnMetrics.recall || 87.9) * 1).toFixed(1) + '%';
    document.getElementById('knn-f1').textContent = ((knnMetrics.f1_score || knnMetrics.f1 || 87.3) * 1).toFixed(1) + '%';
    document.getElementById('knn-time').textContent = (knnMetrics.avg_response_time || 189) + 'ms';
}

// Update Stats Display
function updateStatsDisplay(statsData) {
    if (statsData.total_predictions !== undefined) {
        // Update any stats cards if they exist
        const predictionCountEl = document.querySelector('[id*="prediction"]');
        if (predictionCountEl) {
            predictionCountEl.textContent = statsData.total_predictions;
        }
    }
}

// Load Model Performance Data
async function loadModelPerformance() {
    console.log('Loading model performance data from backend...');
    const performanceContent = document.getElementById('performanceContent');
    
    try {
        // First, generate the plot
        const generateUrl = `${API_URL}/generate-performance-plot`;
        console.log('Generating plot at:', generateUrl);
        
        const generateResponse = await fetch(generateUrl);
        if (!generateResponse.ok) {
            throw new Error(`Failed to generate plot: HTTP ${generateResponse.status}`);
        }
        
        const generateData = await generateResponse.json();
        console.log('Plot generation response:', generateData);
        
        if (generateData.success && generateData.available) {
            // Plot was generated, now fetch the image
            console.log('Plot generated successfully, fetching image...');
            
            // Use the API endpoint to serve the image
            const imageUrl = `${API_URL}/api/performance-plot-image`;
            console.log('Fetching image from:', imageUrl);
            
            updateModelPerformanceDisplay(imageUrl);
        } else {
            const errorMsg = generateData?.message || 'Plot generation skipped';
            console.warn('Plot not available:', generateData);
            showPerformanceError(`Backend: ${errorMsg}`, performanceContent);
        }
    } catch (error) {
        console.error('Error loading model performance:', error.message);
        console.error('Error stack:', error.stack);
        showPerformanceError(`Error: ${error.message}`, performanceContent);
    }
}

// Update Model Performance Display
function updateModelPerformanceDisplay(plotUrl) {
    const performanceContent = document.getElementById('performanceContent');
    if (!performanceContent) {
        console.error('performanceContent element not found');
        return;
    }
    
    console.log('Updating performance display with URL:', plotUrl);
    
    // Show loading state
    performanceContent.innerHTML = '<div class="placeholder"><i class="fas fa-spinner fa-spin"></i><p>Loading chart...</p></div>';
    
    // Create image element
    const img = new Image();
    img.className = 'performance-plot-image';
    img.alt = 'Model Training Performance';
    
    // Handle successful load
    img.onload = function() {
        console.log('Image loaded successfully:', plotUrl);
        performanceContent.innerHTML = ''; // Clear loading state
        performanceContent.appendChild(img); // Add image
        console.log('Image appended to DOM');
    };
    
    // Handle load error
    img.onerror = function(error) {
        console.error('Image failed to load from:', plotUrl);
        performanceContent.innerHTML = `
            <div class="placeholder">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Failed to load performance chart</p>
                <p style="font-size: 12px; color: #a0aec0;">Trying URL: ${plotUrl}</p>
            </div>`;
    };
    
    // Add timeout to catch cases where image doesn't load
    const loadTimeout = setTimeout(() => {
        if (performanceContent.querySelector('.fa-spinner')) {
            console.warn('Image loading timeout for:', plotUrl);
            // Try direct HTML insertion with timestamp to bypass cache
            const timestamp = new Date().getTime();
            const urlWithCache = `${plotUrl}?t=${timestamp}`;
            performanceContent.innerHTML = `<img src="${urlWithCache}" alt="Model Training Performance" class="performance-plot-image" style="width: 100%; height: auto; border-radius: 8px;">`;
        }
    }, 3000);
    
    // Set image source to trigger load
    img.src = plotUrl;
    console.log('Image loading started for:', plotUrl);
}

// Show error message in performance content
function showPerformanceError(message, container) {
    if (container) {
        container.innerHTML = `<div class="placeholder"><i class="fas fa-exclamation-circle"></i><p>${message}</p></div>`;
    }
}

// Show Toast Notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        info: 'info-circle'
    };

    toast.innerHTML = `
        <i class="fas fa-${icons[type]}"></i>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}
