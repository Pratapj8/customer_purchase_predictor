# 🛒 Customer Purchase Predictor

A machine learning-based application that predicts whether a customer is likely to make a purchase based on their behavior and demographic data.

## 🚀 Overview

The **Customer Purchase Predictor** project aims to help businesses understand customer purchase behavior and make data-driven decisions. By analyzing historical customer data, the model predicts the likelihood of future purchases, allowing for better-targeted marketing and improved sales strategies.

## 🔍 Features

- Data preprocessing and feature engineering
- Machine learning model is used for prediction - Logistic Regression
- Evaluation metrics (Accuracy, Precision, Recall, ROC-AUC)
- User-friendly interface (Streamlit)

## 🧰 Tech Stack

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib
- **Model Deployment (Optional):** Streamlit 
- **Version Control:** Git

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/customer-purchase-predictor.git
   cd customer-purchase-predictor
2. Set Up Your Environment by creating a virtual environment
   ```bash
   python -m venv venv
3. Activate the environment

  ### On Windows:
     venv\Scripts\activate #or
     .\venv\Scripts\activate 
  ### On macOS/Linux:
     ```bash
     source venv/bin/activate

4. Install Required Packages
     ```bash
     pip install streamlit numpy #or
     pip install -r requirements.txt
  
5.📁 Create Your Project Folder (if not done yet)
  -  customer_purchase_predictor/
  -  ├── app/
  -  │   ├── model_utils.py
  -  │   └── predict_app.py
  -  ├── model/
  -  │   ├── theta.npy
  -  │   └── X_mean_std.npy
  -  ├── requirements.txt
6. Run Your Streamlit App (make sure you're in the project root):
     ```bash
     cd customer_purchase_predictor
     streamlit run app/predict_app.py
                                                            
