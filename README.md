# DeepFake Detection

This project focuses on the detection of DeepFakes, defined as a binary classification problem. The project is divided into two parts:

## 1. Traditional Machine Learning `traditional_classifers.py`

The first part of the project involves the application of traditional machine learning models for image binary classification. The models used include:

- K-Means Clustering
- K-Nearest Neighbors (KNN)
- Kernel Ridge Regression
- Random Forest Regressor
- Support Vector Machine (SVM) Regressor
- Linear Support Vector Classifier (SVC)
- Stochastic Gradient Descent (SGD) Classifier
- Bagging Classifier with Decision Tree base estimator

Each model has its own strengths and characteristics, providing a comprehensive exploration of different approaches to achieve accurate and robust binary image classification.

## 2. Deep Learning: `nn_classifier.py`, `helper.py`

The second part of the project involves performing classification using neural networks, specifically Convolutional Neural Networks (CNNs) and transfer learning with pre-trained networks.

### Preprocessing

Before image processing, several preprocessing steps are performed:

- **Image Resizing**: Images are resized from 300x300 to 28x28 to reduce computational expense.
- **Grayscale Conversion**: Color images are converted to grayscale to standardize the data and reduce noise.
- **Image Mirroring**: As part of data augmentation, images are horizontally flipped to effectively double the dataset size.
- **Contrast Enhancement**: This is performed to emphasize differences between light and dark areas in the images.

### Modelling

The `efficientnet_b4` model initializes and trains a CNN model using EfficientNet-B4. The model is fine-tuned for a binary classification task. If a saved model state exists, it is loaded and evaluated on a validation dataset. Otherwise, the model is trained using a cross-entropy loss function and Adam optimizer. The training loop accumulates gradients over multiple batches, updates the model's parameters, and computes the loss and accuracy metrics. Early stopping is implemented to prevent overfitting. After training, the model's state dictionary is saved for future use.
