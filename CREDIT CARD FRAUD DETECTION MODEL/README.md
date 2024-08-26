The link  for the DATASET of following project: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
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

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

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
