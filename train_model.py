
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import matplotlib.pyplot as plt

def create_model(num_classes):
    """Create EfficientNetB0 model for weather classification"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_weather_model():
    """Train the weather classification model"""
    # Dataset paths
    dataset_path = "indabax_south_sudan_intermediate"
    train_dir = os.path.join(dataset_path, "weather_dataset")
    test_dir = os.path.join(dataset_path, "test")
    
    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        print("Please make sure your dataset is in the correct location.")
        return
    
    # Create datasets
    train_dataset = image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset="training",
        seed=123
    )
    
    validation_dataset = image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset="validation",
        seed=123
    )
    
    # Get class names
    class_names = train_dataset.class_names
    print(f"Found classes: {class_names}")
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    # Apply augmentation and optimization
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create model
    model = create_model(len(class_names))
    print(f"Model created with {len(class_names)} classes")
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=3,
            factor=0.5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_weather_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('weather_model.h5')
    print("Model saved as 'weather_model.h5'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Test on test dataset if available
    if os.path.exists(test_dir):
        test_dataset = image_dataset_from_directory(
            test_dir,
            image_size=(224, 224),
            batch_size=32,
            shuffle=False
        )
        
        test_loss, test_accuracy = model.evaluate(test_dataset)
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_weather_model()
