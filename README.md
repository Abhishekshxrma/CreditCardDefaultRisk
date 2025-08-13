# Credit Card Default Risk Prediction

This repository contains my **learning project** on predicting credit card default risk.  
I built this project by following and adapting an existing public repository, with the goal of understanding the **data analysis workflow**, **feature engineering**, and **machine learning model building** process.

## 📌 Project Overview
The aim of this project is to predict whether a customer will default on their credit card payment next month based on historical payment and demographic data.

**Key objectives:**
- Explore and preprocess the dataset.
- Perform **EDA** (Exploratory Data Analysis) to identify trends and patterns.
- Engineer features for better model performance.
- Build and evaluate machine learning models, including a neural network.

## 📂 Dataset
The dataset used in this project is the **[UCI Credit Card Default dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)**.

**Features include:**
- Demographics (age, gender, education, etc.)
- Credit limit information
- Past payment history
- Bill statements
- Payment amounts

**Target Variable:**
- `default.payment.next.month` → 1 if customer defaults, 0 otherwise.

## ⚙️ Technologies Used
- **Python**
- **Jupyter Notebook**
- Libraries:
  - `pandas`, `numpy` → Data manipulation
  - `matplotlib`, `seaborn` → Data visualization
  - `scikit-learn` → Machine learning models & evaluation
  - `tensorflow.keras` → Deep learning (Sequential model)

## 🚀 Workflow
1. **Data Loading & Understanding**
2. **Data Cleaning**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Engineering**
5. **Model Training & Evaluation**
6. **Model Comparison**

## 🤖 Models Used

### Deep Learning Model (Keras Sequential)
A neural network built using the **Sequential API** with the following architecture:
- Dense layer: 128 neurons, ReLU activation
- Dropout: 0.3
- Dense layer: 64 neurons, ReLU activation
- Dropout: 0.3
- Dense layer: 32 neurons, ReLU activation
- Dropout: 0.3
- Output layer: 1 neuron, Sigmoid activation

**Compilation settings:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: AUC

## 📊 Results
The project compares the performance of traditional ML models and the neural network.  
The evaluation metrics used are **accuracy**, **precision**, **recall**, **F1-score**, and **AUC**.
