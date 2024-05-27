# GoodReads-Book-Rating-Prediction
This project applies neural networks to GoodReads data, predicting user engagement from review texts and interactions. It explores a spectrum of models, from deep neural networks to hybrids like CNN+LSTM.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Results](#results)

## Introduction
This project aims to predict book ratings on GoodReads based on the review text and other related features. Several machine learning and deep learning models have been implemented to explore the best possible architecture and technique for this prediction task.

## Setup
### Dependencies:
- Python 3.7+
- TensorFlow 2.x
- pandas
- matplotlib
- scikit-learn

### Installation:
```bash pip install pandas matplotlib scikit-learn tensorflow```

## Dataset Location:
The dataset can be downloaded from [Kaggle: CS9856 Spotify Classification Problem 2023](https://www.kaggle.com/competitions/cs9856-spotify-classification-problem-2023/data).


## Data Preprocessing
1. Data is loaded from CSV files into pandas dataframes.
2. Missing values in `review_id` are handled.
3. Data is split into training and validation sets.
4. The review text is vectorized using the TF-IDF technique, with a feature limit set to 50.
5. Numerical features like `n_votes` and `n_comments` are combined with the text vectors to form the final dataset.

## Models Implemented
- **Linear Regression**: A baseline model to understand the relationship between the features and the target variable.
- **3-Layer Neural Network**: A simple feedforward neural network with 3 layers.
  
### Deep Neural Networks:
Various architectures were explored:
  - 3-layer architecture with different activation functions like selu, relu, and elu.
  - 5-layer deep architecture using the elu activation function.

### Complex Neural Networks:
  - **CNN**: Convolutional Neural Network model to detect patterns in sequences.
  - **LSTM**: Long Short-Term Memory network to capture long-term dependencies in the data.
  - **CNN+LSTM**: A combination of CNN and LSTM architectures was also explored.

## Results
- **Linear Regression** achieved an accuracy of 37%.
- The **3-layer NN model** showed promising results with a validation Mean Absolute Error (MAE) of 0.87.
- **Deep NN models** provided insights into the significance of depth and activation functions.
- Complex models like **CNN, LSTM**, and the hybrid of both demonstrated the capability of capturing patterns in sequences and long-term dependencies respectively.
- While **LSTM model** yielded the best accuracy among all the tested architectures, achieving approximately 50% accuracy.



