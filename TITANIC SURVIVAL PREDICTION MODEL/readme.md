# Titanic Survival Prediction Model

This repository contains a Jupyter Notebook implementing a machine learning model to predict passenger survival on the Titanic. Using Python and the `scikit-learn` library, the model is built and trained on the famous Titanic dataset.

## Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster based on features such as:

- Passenger Class (`Pclass`)
- Sex
- Age
- Number of siblings/spouses aboard (`SibSp`)
- Number of parents/children aboard (`Parch`)
- Fare
- Embarked port

The project follows the typical data science process:
1. Data Loading and Exploration
2. Data Preprocessing
3. Model Building and Training
4. Model Evaluation

## Dataset

The dataset used in this project is the Titanic dataset, which is publicly available on [Kaggle](https://www.kaggle.com/c/titanic/data). It contains 891 rows and 12 columns with information about the passengers, such as age, gender, class, and survival status.

### Data Features:
- `PassengerId`: Unique identifier for each passenger
- `Survived`: Target variable (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Passenger's name
- `Sex`: Passenger's gender
- `Age`: Passenger's age
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Installation

To run this notebook, you need to install the required dependencies. You can install them using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Titanic_Survival_Prediction_Model.git
    ```
2. Navigate to the directory:
    ```bash
    cd Titanic_Survival_Prediction_Model
    ```
3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Titanic_Survival_Prediction_Model.ipynb
    ```

## Model

The model used in this project is a **Logistic Regression** classifier. Logistic regression is a popular choice for binary classification problems, such as predicting survival (0 or 1). The data is split into training and testing sets to evaluate the performance of the model.

### Evaluation Metrics
The model is evaluated using **accuracy** as the primary metric. Additional metrics such as precision, recall, and the confusion matrix could also be considered to assess model performance.

## Results

The final model achieves an accuracy of `78.77%` on the test set. This result can be improved further with hyperparameter tuning, feature engineering, and experimenting with other machine learning algorithms such as Random Forests or Support Vector Machines.

## Conclusion

This project serves as a foundational exercise in predictive modeling using Python. By working through the Titanic dataset, we gained experience with the essential steps in building a machine learning pipeline, including data preprocessing, model training, and evaluation.

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your improvements or suggestions.

