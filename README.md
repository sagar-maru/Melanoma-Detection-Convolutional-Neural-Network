# Melanoma Detection Using Custom Convolutional Neural Network
> A multiclass classification model to detect melanoma and other skin diseases using a custom CNN built in TensorFlow.

---

## Table of Contents
* [General Information](#general-information)
* [Problem Statement](#problem-statement)
* [Dataset Details](#dataset-details)
* [Project Pipeline](#project-pipeline)
* [Technologies Used](#technologies-used)
* [Model Architecture and Approach](#model-architecture-and-approach)
* [Handling Challenges](#handling-challenges)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

---

## General Information
Skin cancer is one of the most common forms of cancer, with melanoma being the most lethal subtype. Early detection can save lives, but manual diagnosis is time-consuming and subject to human error. This project aims to build a machine learning solution that assists dermatologists in accurately detecting melanoma and other skin conditions from images using a custom convolutional neural network (CNN).

By leveraging a dataset of skin lesion images, the model predicts whether the lesion belongs to one of nine classes of oncological diseases. This solution aims to reduce diagnostic time and enhance the accuracy of skin cancer detection.

---

## Problem Statement
Melanoma accounts for 75% of skin cancer deaths despite representing a small proportion of skin cancers. Early and accurate detection is critical to improve survival rates. The aim of this project is to:

- Build a custom CNN model to classify skin lesion images into nine categories accurately.
- Address challenges like class imbalance and potential overfitting in the training process.
- Implement data augmentation strategies to improve the robustness and generalization of the model.

---

## Dataset Details
The dataset used in this project was sourced from the **International Skin Imaging Collaboration (ISIC)**. It contains 2,357 labeled images of skin lesions, categorized into the following nine diseases:

1. Actinic keratosis  
2. Basal cell carcinoma  
3. Dermatofibroma  
4. Melanoma  
5. Nevus  
6. Pigmented benign keratosis  
7. Seborrheic keratosis  
8. Squamous cell carcinoma  
9. Vascular lesion  

### Key Notes:
- The dataset is slightly imbalanced, with more images of melanoma and moles compared to other classes.
- All images were resized to a uniform dimension of **180x180 pixels** for training.

---

## Project Pipeline
The project follows these structured steps:

### 1. Data Understanding and Preparation
- Read and preprocess the dataset.
- Divide the dataset into training and validation sets.
- Resize all images to **180x180 pixels** and batch them with a size of 32.

### 2. Dataset Visualization
- Visualize one representative image from each of the nine classes to understand the dataset distribution.

### 3. Initial Model Building and Training
- Build a custom CNN architecture to classify images into nine classes.
- Normalize pixel values to the range [0, 1] by rescaling images.
- Use an appropriate optimizer (e.g., Adam) and loss function (e.g., sparse categorical crossentropy).
- Train the model for **~20 epochs** and analyze results for overfitting/underfitting.

### 4. Data Augmentation
- Apply data augmentation techniques (e.g., rotation, flipping, zooming) to improve model generalization and address overfitting.

### 5. Training with Augmented Data
- Re-train the model on the augmented dataset for **~20 epochs**.
- Analyze the results to check whether overfitting/underfitting has been resolved.

### 6. Handling Class Imbalance
- Use the **Augmentor library** to address the class imbalance in the training dataset.
- Augment data for underrepresented classes to ensure balanced representation.

### 7. Training on Balanced Data
- Train the model on the class-balanced dataset for **~30 epochs**.
- Evaluate the final model performance to ensure robustness and accuracy.

---

## Model Architecture and Approach
### Model Overview:
The custom CNN architecture includes:
- Convolutional layers with ReLU activation for feature extraction.
- MaxPooling layers to reduce spatial dimensions and prevent overfitting.
- Dropout layers to add regularization.
- Fully connected dense layers for classification into nine classes.

### Key Hyperparameters:
- **Optimizer**: Adam (learning rate = 0.001)  
- **Loss Function**: Sparse categorical crossentropy  
- **Batch Size**: 32  
- **Epochs**: Initial (20), Augmented Data (20), Balanced Data (30)

---

## Handling Challenges
### 1. Overfitting:
- Regularization using dropout layers in the CNN.
- Data augmentation techniques to create variations in the training data.
  
### 2. Class Imbalance:
- Identified dominant and underrepresented classes in the dataset.
- Used the **Augmentor library** to increase samples for minority classes.

### 3. Model Performance:
- Experimented with different CNN architectures and hyperparameters.
- Evaluated model accuracy and loss metrics after each training phase.

---

## Technologies Used
- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Pandas**
- **Augmentor**
- **Google Colab** - For GPU runtime during model training

---

## Conclusions
1. The custom CNN achieved high accuracy in classifying skin lesions into nine classes.  
2. Data augmentation significantly reduced overfitting and improved generalization.  
3. Addressing class imbalance using the Augmentor library further enhanced the model's robustness.  
4. The final model demonstrates the potential to assist dermatologists in diagnosing melanoma and other skin conditions effectively.

---

## Acknowledgements
- This project was inspired by the **International Skin Imaging Collaboration (ISIC)** initiative.  
- References:
  - [ISIC Archive](https://www.isic-archive.com/)
  - TensorFlow and Keras documentation.  
- Special thanks to [Google Colab](https://colab.research.google.com/) for providing free GPU resources for training.

---

## Contact
Created by **[Sagar Maru](https://github.com/sagar-maru)**  
Feel free to reach out for collaborations or queries!

--- 
