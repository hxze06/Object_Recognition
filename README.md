# 🧠 Object Recognition with CNN using TensorFlow

This project demonstrates a simple but effective **Convolutional Neural Network (CNN)** model built using **TensorFlow** and **Keras** to classify objects from the **CIFAR-10 dataset**. The model is trained to recognize 10 different object categories and achieves solid accuracy using a custom architecture.

---

## 📚 Dataset

We use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which contains 60,000 32x32 color images in 10 different classes:

- 🚀 Airplane  
- 🚗 Car  
- 🐦 Bird  
- 🐱 Cat  
- 🦌 Deer  
- 🐶 Dog  
- 🐸 Frog  
- 🐴 Horse  
- 🚢 Ship  
- 🚛 Truck

---

## 🧠 Model Architecture

- 3 Convolutional layers with ReLU activation
- MaxPooling layers after first two conv blocks
- Dense layer with 64 units
- Final Dense output layer with 10 units (one per class)

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install tensorflow numpy

2. Train the model

python train_model.py

This will:

Train the CNN on CIFAR-10 for 10 epochs

Save the model as object_model.h5

Save class names to class_names.npy

🧪 Files in this Repo
File	Description
train_model.py	CNN training script
object_model.h5	Trained Keras model
class_names.npy	List of class labels used in prediction
README.md	Project documentation

📌 Requirements
Python 3.11

TensorFlow 2.19

NumPy

📦 Future Add-ons (optional ideas)
Deploy as a web app with Flask or Streamlit

Add prediction script for custom images

Improve accuracy using Data Augmentation or Transfer Learning

👨‍💻 Author
GitHub: @hxze06

🔖 License
This project is open-source and free to use under the MIT License.