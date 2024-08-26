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

