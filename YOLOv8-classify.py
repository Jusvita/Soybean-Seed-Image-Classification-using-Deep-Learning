import time
import yaml
import json  # reading and writing JSON (JavaScript Object Notation) data
import os  # returning, checking, creating, deleting, changing, and retrieving files or directories
import shutil  # copying, moving, deleting, and managing files and directories
import pandas as pd  # managing and analyzing data, especially in tabular format like DataFrame
import seaborn as sns  # creating more attractive and informative data visualizations, built on top of Matplotlib
import numpy as np
import requests  # performing HTTP requests, allowing us to fetch data from the web, such as HTML or other files
from bs4 import BeautifulSoup  # parsing HTML or XML documents, commonly used in web scraping to extract data from web pages
from PIL import Image  # outputting images for each class
import matplotlib.pyplot as plt  # initializing plots
import torch.nn.functional as F

# Define paths for train and validation datasets
dataset_path = '/content/drive/MyDrive/...../Sebelum_Augmentasi/70_30'
train_dir, val_dir = os.path.join(dataset_path, 'train'), os.path.join(dataset_path, 'val')

def analyze_dataset(directory):
    """
    Analyze dataset to count total files, unique dimensions, and color modes.
    """
    total_files, dimensions, color_modes = 0, set(), set()

    for cls in os.listdir(directory):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            for f in os.listdir(cls_path):
                file_path = os.path.join(cls_path, f)
                try:
                    with Image.open(file_path) as img:
                        total_files += 1
                        dimensions.add(img.size)
                        color_modes.add(img.mode)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return total_files, dimensions, color_modes

# Analyze train and validation datasets
train_total, train_dims, train_modes = analyze_dataset(train_dir)
val_total, val_dims, val_modes = analyze_dataset(val_dir)

# Get class names from the train dataset
classes = [cls for cls in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cls))]

!pip install --ultralytics -q  # Install the Ultralytics package silently

from ultralytics import YOLO  # Import the YOLO class from the Ultralytics package
model = YOLO('yolov8m-cls.pt')  # Load the YOLOv8 classification model

# Training function
def train_and_save_results():
    start_time = time.time()
    print("Training model...")

    # Train model
    results = model.train(data=dataset_path,
                          epochs=100,
                          imgsz=224,
                          name='70_30',
                          batch=32,
                          optimizer='Adam',     # Use Adam optimizer
                          lr0=0.0001,
                          plots=True,  # Menyimpan plot learning rate dan loss
                          device=0,
                          verbose=True
                         )
# Check validation result
metrics = model.val()
print(metrics)

#Evaluation
# Loop through each class in the test folder
for class_name in os.listdir(val_data_path):
    class_folder = os.path.join(val_data_path, class_name)

    if os.path.isdir(class_folder):
        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)

            # Process only image files (.jpg or .png)
            if filename.endswith(('.jpg', '.png')):
                # Predict the class of the image using the model
                results = model.predict(file_path, imgsz=224, conf=0.25)

                # Get probabilities from the prediction results
                predicted_class_index = np.argmax(probs)  # Index of the class with the highest probability
                predicted_class = results[0].names[predicted_class_index]  # Class name

                # Store true labels and predicted labels
                y_true.append(class_name)  # True label
                y_pred.append(predicted_class)  # Predicted label

# Generate classification report
report = classification_report(y_true, y_pred, target_names=os.listdir(val_data_path))
accuracy = accuracy_score(y_true, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=os.listdir(val_data_path))
