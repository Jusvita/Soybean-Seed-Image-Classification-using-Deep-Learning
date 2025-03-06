import tensorflow as tf  # Core TensorFlow library for deep learning models

# Pre-trained ResNet50 model and preprocessing function
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Layers commonly used in CNN models for feature extraction and classification
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

# Callbacks to monitor and improve training performance
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from tensorflow.keras.models import load_model  # Load a saved model
from tensorflow.keras.optimizers import Adam  # Adam optimizer for training
from tensorflow.keras.preprocessing import image  # Image processing utilities
from tensorflow.keras.models import Model  # Base class for defining Keras models
from tensorflow.keras.regularizers import l2  # L2 regularization to prevent overfitting

# Evaluation metrics from scikit-learn
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt  # Visualization library for plotting graphs
import seaborn as sns  # Statistical data visualization library
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Numerical computing library
import time  # Time tracking for performance measurement
import os  # Operating system utilities for file and directory management

# Path to the dataset folder
image_dir = '/content/drive/MyDrive/..../Sebelum_Augmentasi/70_30'
train_dir = os.path.join(image_dir, 'train')
val_dir = os.path.join(image_dir, 'val')
test_dir = '/content/drive/MyDrive/...../data_test'

# Dataset parameters
image_size = (224, 224)  # Target image size
batch_size = 32  # Number of images per batch

# Create training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    image_size=image_size,
    batch_size=batch_size
)

# Create validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=val_dir,
    image_size=image_size,
    batch_size=batch_size
)

# Create test dataset
test_dir = tf.keras.utils.image_dataset_from_directory(
    directory=test_dir,
    image_size=image_size,
    batch_size=batch_size
)

def load_image_classification_model(input_shape):
    # Load the pre-trained ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model weights to prevent updating during training

    # Custom layers for fine-tuning
    x = base_model.output  # Get the output of the base model
    x = GlobalAveragePooling2D()(x)  # Apply global average pooling to reduce dimensions
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # Fully connected layer with L2 regularization
    x = BatchNormalization()(x)  # Apply batch normalization to improve training stability
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # Another fully connected layer
    x = BatchNormalization()(x)  # Apply batch normalization
    x = Dropout(0.5)(x)  # Add another dropout layer
    predictions = Dense(len(class_names), activation='softmax')(x)  # Output layer with softmax activation for multi-class classification

    # Combine the base model and custom layers to form the complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model  # Return the constructed model

# Load the model
input_shape = (*image_size, 3)  # Input shape should match image size and 3 color channels (RGB)
Resnet50_model = load_image_classification_model(input_shape)

Resnet50_model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Adam optimizer with a small learning rate for fine-tuning
    loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification with integer labels
    metrics=['accuracy']  # Track accuracy as the evaluation metric
)

# Train the model and include validation data
history = Resnet50_model.fit(
    train_70_30,
    epochs=100,
    validation_data=val_70_30,
    callbacks=[csv_logger, reduce_lr, model_checkpoint]
)

# Create callbacks
csv_logger = CSVLogger(os.path.join(results_dir, '70_30.csv'), append=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Callback to save the best model (monitor val_accuracy)
checkpoint_path = os.path.join(results_dir, 'best70_30.keras')
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',  # Monitor validation accuracy
    mode = 'max',
    save_best_only=True,     # Save only when validation accuracy improves
    save_weights_only=False, # Save the entire model
    verbose=1                # Display save information
)

# Train the model and include validation data
history = Resnet50_model.fit(
    train_70_30,
    epochs=100,
    validation_data=val_70_30,
    callbacks=[csv_logger, reduce_lr, model_checkpoint]
)

# Evaluate the model using the test dataset
test_loss, test_accuracy = Resnet50_model.evaluate(test_dataset, verbose=1)

# Print the evaluation results: test loss and accuracy
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Function to get true labels and predicted labels for the test dataset
def get_true_and_predicted_labels(model, dataset):
    true_labels = []
    predicted_labels = []

    # Iterate through the dataset and get true labels and predicted labels
    for images, labels in dataset:
        # Predict the class for a batch of images
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)

        predicted_labels.extend(predicted_classes)  # Adding predicted labels

    return true_labels, predicted_labels

# Get true labels and predicted labels from the test dataset
true_labels, predicted_labels = get_true_and_predicted_labels(best_model, test_dataset)

# 1. Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
