# Soybean-Seed-Image-Classification-using-Deep-Learning
This repository contains code and datasets for my thesis (Jusvita from Syarif Hidayatullah Islamic State University Jakarta) on soybean seed image classification using deep learning. It compares the performance of YOLOv8 and ResNet-50 in classifying five seed types: broken, intact, spotted, immature, and skin-damaged, while also evaluating the impact of color augmentation on model accuracy.

## **Dataset**
1. Dataset Structure
![Screenshot 2025-03-06 093940](https://github.com/user-attachments/assets/1382ab27-2892-4890-a78b-706537dbd036)

   The dataset consists of 5,000 soybean seed images classified into five categories:
   - Broken
   - Intact
   - Spotted
   - Immature
   - Skin-Damaged 

3. Data Splitting

   The dataset is split into 80:20 for training and testing.
   The training data is further divided into three different ratios: 70:30, 80:20, and 90:10 to evaluate the impact of training and validation data sizes on model accuracy.
   
4. Data Augmentation

   To enhance model generalization and reduce overfitting, color augmentation is applied to the training data, including:
   - Brightness
   - Contrast
   - Saturation
   - Hue
   
   After augmentation, the number of images in the training set increases to 20,000, while the test set remains at 5,000, with the same validation split ratios.
   This dataset is used to compare the performance of YOLOv8 and ResNet-50 in soybean seed image classification and analyze the effect of augmentation on model accuracy.

## **Models & Methods Used**
- YOLOv8: YOLOv8 is the latest version of the YOLO (You Only Look Once) model released by Ultralytics. YOLOv8 is used for object detection, segmentation, and image classification in computer vision applications.
- ResNet50: ResNet50 is a deep learning architecture based on residual learning that addresses the vanishing gradient problem and performance degradation.

## **Hyperparameters**
![image](https://github.com/user-attachments/assets/cdd5cf0f-7b90-41ca-b0b9-c97c9ced1142)

## **Evaluation & Results**
1. The study results demonstrate that deep learning models using YOLOv8 and ResNet50 architectures successfully classify soybean seeds. The YOLOv8 model, with a 70:30 data split after augmentation, achieved the highest performance with 94% accuracy. It also outperformed ResNet50 and other ratios in terms of precision, recall, and F1-score. Additionally, YOLOv8 exhibited superior validation accuracy and lower validation loss, indicating greater model stability and reduced overfitting risk. To further illustrate the model’s effectiveness, the classification report, confusion matrix, and accuracy-loss graphs of the YOLOv8 model with a 70:30 ratio after augmentation are provided.
![image](https://github.com/user-attachments/assets/034f421f-14c6-4b3c-b30f-2c593686d858)
![image](https://github.com/user-attachments/assets/092c23e6-ced0-4127-8d14-3b38743df1b0)
These are the accuracy-loss graphs
![image](https://github.com/user-attachments/assets/1dd0746a-3c88-48d4-a801-6dcf5fdf65cf) ![image](https://github.com/user-attachments/assets/d7ebf8d7-e3db-45c2-bbad-e7edf8eefd07)

2. Color manipulation in digital images generally contributed positively to enhancing classification accuracy for both models. For the YOLOv8 model with a 70:30 ratio, validation accuracy improved from 94.9% to 99.93% after color augmentation, while for the ResNet50 model, accuracy increased from 90.08% to 98.83%. These findings indicate that color manipulation enhances the model’s ability to recognize visual features of soybean seeds more effectively, ultimately improving classification performance.

If you want to know more about this research, you can visit my thesis report or contact me to jusvitajusvita@gmail.com and my instagram @jv_jusvitaa

Hope it helps!






