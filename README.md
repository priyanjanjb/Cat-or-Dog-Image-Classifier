# 🐱🐶 Cats vs Dogs Image Classification using Deep Learning & Transfer Learning

This repository presents two different approaches to solving the **Cats vs Dogs Image Classification** problem using **Deep Learning**:

1. 🧠 **Custom CNN Model** – Built from scratch using Convolutional Neural Networks (CNNs).  
2. ⚡ **Transfer Learning Model (Xception)** – Leveraging a pre-trained model trained on the ImageNet dataset for better accuracy and faster convergence.

Both models are implemented using **TensorFlow** and **Keras**, and trained on the **Cats and Dogs for Classification dataset** from Kaggle.

---

## 📂 Repository Structure

Cats-vs-Dogs-Classification/
│
├── 🐾 custom_cnn_model.py # Model built from scratch using CNN
├── ⚡ transfer_learning_xception.py # Model using pretrained Xception network
├── 📁 dataset/ # Dataset (from Kaggle)
│ ├── training_set/
│ └── test_set/
└── README.md # Project documentation


---

## 🧠 Project Overview

This project demonstrates how to classify images of **cats and dogs** using deep learning.  
It includes:
- Data preprocessing and image resizing  
- Model building (CNN and Transfer Learning)  
- Training and validation  
- Model evaluation and performance visualization  
- Testing on new images  

---

## 🧩 Dataset Details

- **Source:** [Kaggle – Cats and Dogs for Classification](https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification)
- **Classes:** `cats`, `dogs`
- **Image Size:** 128 × 128 pixels  
- **Training Samples:** 7200  
- **Validation Samples:** 800  
- **Testing Samples:** 2000  
- **Split Ratio:** 80/10/10  

---

## ⚙️ Installation

Before running the scripts, install the required dependencies:

!pip install tensorflow numpy matplotlib pandas opendatasets opencv-python

##1. Custom CNN Model (custom_cnn_model.py)
🧱 Architecture
Layer Type	Parameters
Conv2D	32 filters, 3×3 kernel, ReLU
MaxPooling2D	2×2 pool size
Conv2D	32 filters, 3×3 kernel, ReLU
MaxPooling2D	2×2 pool size
Flatten	—
Dense	128 neurons, ReLU
Dense	1 neuron, Sigmoid

Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy

##Model Summary
Total Parameters: 1,475,521  
Trainable Parameters: 1,475,521

| Epoch | Training Accuracy | Validation Accuracy | Loss |
| ----- | ----------------- | ------------------- | ---- |
| 1     | 85.2%             | 83.5%               | 0.44 |
| 2     | 89.8%             | 88.4%               | 0.31 |
| 3     | 91.6%             | 90.3%               | 0.28 |

Testing the Model
img = image.load_img('/content/cat_or_dog_1.jpg', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
result = cnn.predict(img_array)

if result[0][0] == 1:
    print("Dog")
else:
    print("Cat")

## 2. Transfer Learning Model – Xception (transfer_learning_xception.py)

This model uses Transfer Learning by importing the Xception base model pretrained on ImageNet, freezing its convolutional layers, and adding custom dense layers for classification.

🧱 Architecture
Layer Type	Description
Xception	Pretrained base (ImageNet weights, no top layer)
Flatten	Convert feature maps to vector
Dense (128)	ReLU
Dense (128)	ReLU
Dense (32)	ReLU
Dense (1)	Sigmoid

Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy

##Training Performance (Sample)
|Epoch |Training Accuracy|Validation Accuracy  |Loss    |
|------|-----------------|---------------------|--------|
|1	   |96.7%	         |96.2%	               |0.078   |
|2	   |97.2%	         |95.7%	               |0.073   |
|3	   |97.6%	         |95.3%	               |0.058   |

Training Time: ~0.46 hours (on GPU)

##Model Evaluation (Transfer Learning)
Metric	Score
Precision	0.93
Recall	0.97
Accuracy	0.95
🔍 Testing with New Images
import cv2
img = cv2.imread('/content/images.jpeg')
resized = tf.image.resize(img, (128, 128)) / 255.0
y_hat = model.predict(np.expand_dims(resized, 0))

if y_hat > 0.5:
    print("Dog")
else:
    print("Cat")

##Training Visualization
plt.plot(history.history['accuracy'], color='teal', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='Val Accuracy')
plt.legend()
plt.show()

💡 Key Insights

Transfer learning (Xception) achieved faster convergence and higher accuracy than the custom CNN.

Custom CNN performs well but requires more epochs to generalize effectively.

Image normalization and correct augmentation are crucial for stability.

##Developed By

Priyanjan Perera
🎓 Deep Learning Enthusiast | Software Developer
💻 Tools Used: TensorFlow, Keras, NumPy, Pandas, Matplotlib, OpenCV
📧 priyanjanjb@gmail.com
