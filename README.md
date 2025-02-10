# Customer Churn Detection

## Introduction
Customer churn detection helps identify customers who are likely to stop using a service. This project uses machine learning to predict customer churn based on historical data.

## Project Overview
This Jupyter Notebook analyzes customer behavior and builds a predictive model using various machine learning algorithms.

## Dataset
The dataset used is **Churn_Modelling.csv**, which includes:
- **Numerical Features**: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary
- **Categorical Features**: Geography, Gender, HasCrCard, IsActiveMember
- **Target Variable**: Exited (1 = Churned, 0 = Retained)

## Workflow
1. **Data Preprocessing**
   - Loading the dataset
   - Encoding categorical variables
   - Standardizing numerical features

2. **Exploratory Data Analysis (EDA)**
   - Visualizing distributions of key features
   - Analyzing correlations
   - Checking for class imbalance and applying oversampling (SMOTE)

3. **Model Training & Evaluation**
   - Training models: K-Nearest Neighbors (KNN), Naive Bayes, Support Vector Machine (SVM), Decision Tree (DT)
   - Evaluating models using accuracy, precision, recall, F1-score, and ROC-AUC

4. **Results & Insights**
   - Comparing model performance and choosing the best model

## Requirements
Before running the notebook, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

