# Movie Rating Prediction

This repository contains a Jupyter Notebook that demonstrates the process of predicting movie ratings using machine learning techniques. The project leverages Python's data science libraries to build a predictive model capable of estimating a movie's rating based on various features from the Indian movie industry.

## Overview

The goal of this project is to create a model that can accurately predict the rating of a movie based on a variety of features. Predicting movie ratings is essential for recommendation systems, widely used by streaming platforms and content providers.

## Dataset

The dataset used in this project is `IMDb_Movies_India.csv`, which includes information about Indian movies, with the following columns:

- **Name**: The name of the movie.
- **Year**: The year the movie was released.
- **Duration**: The duration of the movie in minutes.
- **Genre**: The genre(s) of the movie (e.g., Action, Drama).
- **Rating**: The IMDb rating of the movie, which is the target variable.
- **Votes**: The number of votes the movie has received on IMDb.
- **Director**: The director of the movie.
- **Actor1**: The first main actor in the movie.
- **Actor2**: The second main actor in the movie.
- **Actor3**: The third main actor in the movie.

The dataset is sourced from IMDb and focuses on Indian movies.

## Project Structure

- **Movie_Rating_Prediction.ipynb**: The main Jupyter Notebook file that contains the complete code for data preprocessing, feature engineering, model training, evaluation, and results visualization.
- **IMDb_Movies_India.csv**: The dataset containing information about Indian movies used for training and testing the model.
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
   - Handle missing values by either imputing them or removing the rows/columns.
   - Convert categorical variables (e.g., Genre, Director, Actor1) into numerical representations using techniques like one-hot encoding or label encoding.
   - Normalize numerical features such as `Duration` and `Votes` to ensure that all variables contribute equally to the model.

2. **Feature Engineering**:
   - Create new features that might help the model, such as the number of genres or the number of top-billed actors.
   - Analyze feature importance to select the most relevant features for the model.

3. **Model Selection**:
   - Various machine learning models are evaluated, including:
     - **Linear Regression**: A basic model that predicts the rating based on a linear relationship between features.
     - **Random Forest Regressor**: An ensemble model that builds multiple decision trees and merges them for better predictions.
     - **Gradient Boosting Regressor**: A model that builds an ensemble of trees sequentially, where each new tree corrects the errors of the previous ones.

4. **Model Evaluation**:
   - The models are evaluated using metrics like:
     - **Mean Absolute Error (MAE)**: The average absolute difference between the predicted and actual ratings.
     - **Mean Squared Error (MSE)**: The average of the squared differences between the predicted and actual ratings.
     - **R-squared**: The proportion of variance in the dependent variable that is predictable from the independent variables.

5. **Results**:
   - The best-performing model is selected based on its ability to predict movie ratings accurately, minimizing the error between predicted and actual ratings.

## Results

The final model achieved the following results:

- **Mean Absolute Error (MAE)**: [Insert MAE]
- **Mean Squared Error (MSE)**: [Insert MSE]
- **R-squared**: [Insert R-squared]

These results demonstrate the model's effectiveness in predicting movie ratings with a reasonable degree of accuracy.

## Usage

To run the notebook, clone this repository and execute the `Movie_Rating_Prediction.ipynb` file in Jupyter Notebook or JupyterLab:

```bash
git clone https://github.com/Conccurer/Movie-Rating-Prediction.git
cd Movie-Rating-Prediction
jupyter notebook Movie_Rating_Prediction.ipynb
```

## Conclusion

This project demonstrates how machine learning can be applied to the task of predicting movie ratings, specifically within the context of the Indian movie industry. The final model provides a strong foundation for building a recommendation system or enhancing existing models used by streaming platforms.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have any suggestions or improvements.
