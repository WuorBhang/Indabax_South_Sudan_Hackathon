
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Define class names based on your dataset
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Global variable to store the model
model = None

def create_model():
    """Create and compile the EfficientNetB0 model"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_model():
    """Load or create the model"""
    global model
    try:
        # Try to load existing model
        model = tf.keras.models.load_model('weather_model.h5')
        print("Model loaded successfully!")
    except:
        print("No existing model found. Creating new model...")
        model = create_model()
        print("New model created. You'll need to train it with your dataset.")

def preprocess_image(img):
    """Preprocess image for prediction"""
    # Resize image to 224x224 (EfficientNetB0 input size)
    img = img.resize((224, 224))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_weather(img):
    """Predict weather from image"""
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Get all predictions for display
    all_predictions = {}
    for i, class_name in enumerate(CLASS_NAMES):
        all_predictions[class_name] = float(predictions[0][i])
    
    return predicted_class, confidence, all_predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and process the image
        img = Image.open(io.BytesIO(file.read()))
        
        # Make prediction
        predicted_class, confidence, all_predictions = predict_weather(img)
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'image': img_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with the weather dataset"""
    try:
        dataset_path = "indabax_south_sudan_intermediate"
        train_dir = os.path.join(dataset_path, "weather_dataset")
        test_dir = os.path.join(dataset_path, "test")
        
        if not os.path.exists(train_dir):
            return jsonify({'error': 'Training dataset not found'}), 400
        
        # Create data generators
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset="training",
            seed=123
        )
        
        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset="validation",
            seed=123
        )
        
        # Apply data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        # Prepare datasets
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Train the model
        global model
        model = create_model()
        
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=2)
            ]
        )
        
        # Save the trained model
        model.save('weather_model.h5')
        
        return jsonify({
            'message': 'Model trained successfully!',
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
