
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const success = document.getElementById('success');

// Drag and drop functionality
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
        handleFile(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    hideMessages();
    showLoading();

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showError(data.error);
            } else {
                showResults(data);
            }
        })
        .catch(error => {
            hideLoading();
            showError('An error occurred while processing the image.');
            console.error('Error:', error);
        });
}

function showResults(data) {
    // Display the uploaded image
    document.getElementById('resultImage').src = 'data:image/png;base64,' + data.image;

    // Display prediction
    document.getElementById('predictedClass').textContent = data.predicted_class;
    document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    // Display all predictions
    const allPredictionsDiv = document.getElementById('allPredictions');
    allPredictionsDiv.innerHTML = '';

    Object.entries(data.all_predictions).forEach(([className, confidence]) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';

        item.innerHTML = `
                    <span>${className}</span>
                    <div class="prediction-bar">
                        <div class="prediction-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                    <span>${(confidence * 100).toFixed(1)}%</span>
                `;

        allPredictionsDiv.appendChild(item);
    });

    results.style.display = 'block';
}

function trainModel() {
    hideMessages();
    showLoading();

    fetch('/train', {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showError(data.error);
            } else {
                showSuccess(`${data.message} Final accuracy: ${(data.final_accuracy * 100).toFixed(1)}%`);
            }
        })
        .catch(error => {
            hideLoading();
            showError('An error occurred while training the model.');
            console.error('Error:', error);
        });
}

function showLoading() {
    loading.style.display = 'block';
}

function hideLoading() {
    loading.style.display = 'none';
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
}

function showSuccess(message) {
    success.textContent = message;
    success.style.display = 'block';
}

function hideMessages() {
    error.style.display = 'none';
    success.style.display = 'none';
    results.style.display = 'none';
}
