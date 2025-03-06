# Soybean-Seed-Image-Classification-using-Deep-Learning
This repository contains code and datasets for a thesis on soybean seed image classification using deep learning. It compares the performance of YOLOv8 and ResNet-50 in classifying five seed types: broken, intact, spotted, immature, and skin-damaged, while also evaluating the impact of color augmentation on model accuracy.

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









