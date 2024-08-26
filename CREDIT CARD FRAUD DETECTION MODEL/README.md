# Credit Card Fraud Detection

This repository contains a Jupyter Notebook that demonstrates the process of detecting fraudulent credit card transactions using machine learning techniques. The project leverages Python's data science libraries to build a predictive model capable of identifying fraudulent transactions with high accuracy.

## Overview

Credit card fraud detection is a common application of machine learning in the finance industry. The goal of this project is to create a model that can predict whether a given transaction is fraudulent or legitimate based on a dataset of credit card transactions.

## Dataset

The dataset used in this project contains credit card transactions made by European cardholders in September 2013. The dataset has been anonymized and contains only numerical input variables. The key features of the dataset include:

- **Time**: The seconds elapsed between this transaction and the first transaction in the dataset.
- **V1 to V28**: The principal components obtained using PCA (Principal Component Analysis).
- **Amount**: The transaction amount.
- **Class**: The label for the transaction, where 1 indicates fraud and 0 indicates a legitimate transaction.

The link  for the DATASET of following project: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Structure

- **Credit_Card_Fraud_Detection.ipynb**: The main Jupyter Notebook file that contains the complete code for data preprocessing, model training, evaluation, and results visualization.
- **README.md**: This file provides an overview of the project.

## Dependencies

To run the notebook, you'll need to install the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn 
```

## Methodology

1. **Data Preprocessing**: 
   - The dataset is preprocessed by normalizing the features to ensure that all variables contribute equally to the analysis.
   - Class imbalance, a common issue in fraud detection, is addressed using techniques like undersampling or oversampling to balance the number of fraudulent and non-fraudulent transactions.

2. **Model Selection**: 
   - Various machine learning models are evaluated, including:
     - **Logistic Regression**: A linear model that predicts the probability of fraud.
     - **Random Forest**: An ensemble model that creates multiple decision trees and merges them to get a more accurate and stable prediction.
     - **Support Vector Machines (SVM)**: A model that finds the hyperplane that best separates fraudulent and non-fraudulent transactions.
   - Hyperparameter tuning is performed to optimize the models for the best performance.

3. **Model Evaluation**:
   - The models are evaluated using several performance metrics:
     - **Accuracy**: The percentage of correctly classified transactions.
     - **Precision**: The proportion of true positive transactions among all transactions classified as fraud.
     - **Recall**: The proportion of true positive transactions detected out of all actual fraudulent transactions.
     - **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.
     - **ROC-AUC**: The area under the receiver operating characteristic curve, which measures the model's ability to distinguish between classes.

4. **Results**: 
   - The best-performing model is selected based on its ability to correctly identify fraudulent transactions while minimizing false positives. The results demonstrate the effectiveness of the chosen model in detecting fraud with high accuracy and reliability.

## Results

The final model achieved the following results:

- **Accuracy**: [Insert Accuracy]
- **Precision**: [Insert Precision]
- **Recall**: [Insert Recall]
- **F1-Score**: [Insert F1-Score]
- **ROC-AUC**: [Insert ROC-AUC]

These results demonstrate the model's effectiveness in detecting fraudulent transactions, balancing the trade-off between catching fraud and avoiding false positives.

## Usage

To run the notebook, clone this repository and execute the `Credit_Card_Fraud_Detection.ipynb` file in Jupyter Notebook or JupyterLab:

```bash
git clone https://github.com/Conccurer/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
jupyter notebook Credit_Card_Fraud_Detection.ipynb

