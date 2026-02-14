# 🧠 Customer Churn Prediction using Artificial Neural Networks (Deep Learning)

<p align="center">
  <img src="https://img.shields.io/badge/Python-Deep%20Learning-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-Neural%20Network-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/ANN-Churn%20Prediction-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-red?style=for-the-badge&logo=streamlit">
</p>

<p align="center">
🚀 An end-to-end Deep Learning project that predicts customer churn using Artificial Neural Networks and deploys an interactive AI dashboard with Streamlit.
</p>

---

## 📌 Table of Contents

* 🧠 Project Overview
* ✨ Features
* ⚙️ Tech Stack
* 📂 Project Structure
* 🔄 Deep Learning Pipeline
* 🤖 Model Architecture
* 📊 Dataset Information
* 🚀 Streamlit Application
* ▶️ Getting Started
* 🧩 Skills Demonstrated
* 🔮 Future Improvements
* 🧑‍💻 Author

---

## 🧠 Project Overview

Customer churn prediction is a critical business analytics problem.
This project builds a **Deep Learning ANN model** that learns customer behaviour patterns and predicts the probability of churn.

The project demonstrates an end-to-end workflow:

* Data preprocessing & feature engineering
* Neural network training
* Model evaluation
* Deployment using Streamlit

This repository reflects strong foundations in **Deep Learning, Machine Learning engineering, and AI application development**.

---

## ✨ Features

* 🤖 Artificial Neural Network built using TensorFlow/Keras
* 📊 Customer churn probability prediction
* ⚡ Feature scaling & encoding pipeline
* 🌐 Interactive Streamlit dashboard UI
* 🧪 Hyperparameter tuning experiments
* 📈 Model training logs & evaluation

---

## ⚙️ Tech Stack

| Technology             | Usage                  |
| ---------------------- | ---------------------- |
| Python                 | Core Programming       |
| TensorFlow / Keras     | ANN Model              |
| Pandas & NumPy         | Data Processing        |
| Scikit-Learn           | Scaling & Encoding     |
| Streamlit              | Interactive Dashboard  |
| Matplotlib / Notebooks | Experiments & Training |

---

## 📂 Project Structure

```
ANN Project
│
├── app.py                     # Streamlit ANN Dashboard
├── model.h5                   # Trained Neural Network
├── scaler.pkl                 # Feature Scaler
├── label_encoder_gender.pkl   # Label Encoder
├── onehot_encoder_geo.pkl     # One-Hot Encoder
├── experiments.ipynb          # Model Experiments
├── hyperparametertuningann.ipynb
├── prediction.ipynb
├── Churn_Modelling.csv        # Dataset
├── logs/                      # Training Logs
└── requirements.txt
```

---

## 🔄 Deep Learning Pipeline

```
Raw Dataset
   ↓
Data Cleaning
   ↓
Label Encoding + One-Hot Encoding
   ↓
Feature Scaling
   ↓
Artificial Neural Network
   ↓
Probability Output
   ↓
Streamlit Dashboard
```

Key Steps:

* Encoding categorical variables
* Scaling numerical features
* Training ANN with dense layers
* Predicting churn probability

---

## 🤖 Model Architecture

Artificial Neural Network built with:

* Dense hidden layers
* ReLU activation
* Sigmoid output layer
* Binary classification objective

Model learns complex non-linear relationships between:

* Customer demographics
* Financial features
* Account behaviour

---

## 📊 Dataset Information

The dataset contains customer banking information including:

* Geography
* Age
* Balance
* Credit Score
* Products owned
* Activity status

Target:

```
Exited → Customer churn (0 or 1)
```

Goal:

Predict whether a customer is likely to leave the service.

---

## 🚀 Streamlit Application

The repository includes a modern AI dashboard where users can:

* Enter customer details
* Run ANN predictions
* Visualize churn probability instantly

Run locally:

```
streamlit run app.py
```

---

## ▶️ Getting Started

Clone repository:

```
git clone https://github.com/Vashishtha05/Customer-Churn-Prediction-Artificial-Neural-Network.git
```

Install dependencies:

```
pip install -r requirements.txt
```

Run application:

```
streamlit run app.py
```

---

## 🧩 Skills Demonstrated

* Deep Learning with Artificial Neural Networks
* Feature Engineering & Encoding
* Model Training & Hyperparameter Tuning
* End-to-End ML Pipeline Design
* Interactive AI Dashboard Development
* Applied Business Analytics

---

## 🔮 Future Improvements

* Transformer-based churn modeling
* Explainable AI (SHAP / LIME)
* Real-time API deployment

---

## 🧑‍💻 Author

**Vashishtha Verma**

* Machine Learning & Deep Learning Engineer
* Strong background in DSA and Software Development

---

