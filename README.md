# Facial Verification System

```
Facial detection & verification w/ machine learning
```


## Brief

I employ **scikit-learn** to apply facial recognition & verification machine learning algorithms to a collection of JPEG images of people collected over the internet, from the **Labeled Faces in the Wild** dataset.

### Directory Structure

- **notebooks/**
  - `preprocessing.ipynb` 
    - Data is loaded into numPy arrays
    - Split into training & testing data

  - `data_exploration.ipynb` 
    - TODO
    - TODO

  - `model_training.ipynb` Machine Learning Model Training
    - TODO
    - TODO

- `README.md`


## Mathematical Theory

### StandardScaler Module

Standardise features by removing the mean & scaling them to have unit variance. This is a process uses the theory of z-score and is very common to many ML algorithms.

#### What?

- **Standardisation**: The process of transforming data in such a way that it has a standard scale. Standardised data has a normal distribution (spread of values) with a mean of zero & a standard deviation of one.

- **Removing the Mean**: The mean value of a feature is subtracted from each data point in that feature. Centers data around zero. 

- **Unit Variance**: Variance measures the spread of data points around the mean. For the condition of unit variance, the variance of each feature is equal to one. This helps ML algorithms to interpret & compare the data.

- **Z-Score**: Represents the number of standard deviations a data point is from the mean of the feature. For a data point with value _x_, $$z = (x - μ) / σ$$ where _z_ = Z-Score, _μ_ = mean, _σ_ = standard devation.

### Principal Component Analysis (PCA)


  
  