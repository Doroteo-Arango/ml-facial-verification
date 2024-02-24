# Facial Verification System

```
Facial detection & verification w/ machine learning
```


## Brief

I employ **scikit-learn** to apply facial recognition & verification machine learning algorithms to a collection of JPEG images of people collected over the internet, from the **Labeled Faces in the Wild** dataset.

The type of ML algorithm is **support vector classification (SVC)** with **principal component analysis (PCA)** for dimensionality reduction. This model achieves an accuracy of 87%.

### Directory Structure

- **notebooks/**
  - `SVC_PCA_ML.ipynb` 

- **results/**
  - `quantitative.png`
  - `qualitative1.png`
  - `qualitative2.png`

- `README.md`


## Mathematical Theory

### Data Preprocessing

#### sklearn.preprocessing.StandardScaler

Standardise features by removing the mean & scaling them to have unit variance. This is a process uses the theory of z-score and is very common to many ML algorithms.

- **Standardisation**: The process of transforming data in such a way that it has a standard scale. Standardised data has a normal distribution (spread of values) with a mean of zero & a standard deviation of one.

- **Removing the Mean**: The mean value of a feature is subtracted from each data point in that feature. Centers data around zero. 

- **Unit Variance**: Variance measures the spread of data points around the mean. For the condition of unit variance, the variance of each feature is equal to one. This helps ML algorithms to interpret & compare the data.

- **Z-Score**: Represents the number of standard deviations a data point is from the mean of the feature. For a data point with value _x_, $$z = (x - μ) / σ$$ where _z_ = Z-Score, _μ_ = mean, _σ_ = standard devation.

### Exploratory Data Analysis

#### sklearn.decomposition.PCA

Principal Component Analysis (PCA) for Dimensionality Reduction involves summarising larger data sets into smaller sets for easier visualisation & analysis. 

- **Principal Component Analysis**: Linear dimensionality reduction using singular value decomposition (SVD). This is done by finding a linear combination of new features in a lower dimension space via PCA. PCA is a popular technique for reducing data dimensionality while preserving feature integrity. For facial verification, it improves the ML algorithm's ability to extract meaningful features during image processing & analysis. 

- **n_components**: Number of principal components to keep after dimensionality reduction. It dictates the amount of variance retained in the data post-reduction.

- **svd_solver**: Singular value decomposition (SVD) is the technique used to compute principal components. The SVD algorithm chosen for facial recognition is randomised due to its suitability for large datasets where _n_components_ is much smaller than the total number of features.

- **whiten**: True means that the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. The result of this is that the principal components have unit variance after transformation.

### ML Model Training

#### scipy.stats.loguniform

A continuous log uniform probability distribution. The parameters for this process are _C_ & _gamma_.

- **Log Uniform Probability Distribution**: The logarithm of the variable is uniformly distributed  over a specific range. Probability density function is as follows: $$f(x, a, b) = 1 / x log(b/a)$$
where _a_ = lower bound, _b_ = upper bound, _a <= x <= b_, _b > a > 0_.

  This is particularly useful when dealing with parameters that are naturally positive and have a wide range of possible values, such as regularization parameters in machine learning models.

- **Regularisation Strength _C_**: Regularisation prevents overfitting in ML models. _C_ represents the penalty for misclassification during training. A larger value of _C_ encourages a smaller margin, vice versa. It is important to choose an acceptable value for the regularisation strength to ensure a good trade-off between minimising classification error & maximising margin for classification. 

- **Kernel Coefficient _gamma_**: The kernel coefficient determines how input data is mapped into a higher-dimensional space when separated linearly, during the SVM algorithm. It controls the influence of individual training samples on the decision boundary. A smaller _gamma_ implies smoother decision boundaries, meaning each training example has a wider influence on how the ML model works.

#### sklearn.model_selection.RandomizedSearchCV

Performs hyperparameter tuning for ML models using randomised search. 

- **Hyperparameters**: Parameters of an ML model that are defined prior to training. They control aspects of the learning process. Regularisation strength & kernel coefficient are examples. 

- **Randomised Search**: Randomised search is an approach to hyperparameter tuning that randomly samples hyperparameter combinations from specified distributions and evaluates their performance. 


#### sklearn.svm.SVC

Implements C-support vector machine (SVM) algorithm for classification.


## Model Evaluation & Conclusion

### Quantitative Testing

From **sklearn.metrics**, **ConfusionMatrixDisplay** & **classification_report** are used to generate quantitative analysis tables.

|                  | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Ariel Sharon     | 0.62      | 0.62   | 0.62     | 13      |
| Colin Powell     | 0.74      | 0.90   | 0.81     | 60      |
| Donald Rumsfeld  | 0.83      | 0.74   | 0.78     | 27      |
| George W Bush    | 0.93      | 0.95   | 0.94     | 146     |
| Gerhard Schroeder| 0.91      | 0.80   | 0.85     | 25      |
| Hugo Chavez      | 0.89      | 0.53   | 0.67     | 15      |
| Tony Blair       | 0.94      | 0.83   | 0.88     | 36      |
| **Accuracy**     |           |        | **0.87** | **322** |
| **Macro Avg**    | 0.84      | 0.77   | 0.79     | 322     |
| **Weighted Avg** | 0.87      | 0.87   | 0.86     | 322     |


- **F1-Score**: Represents the mean of precision & recall.

- **Precision**: Ratio of true positives compared to all positive predictions. Accuracy of positive predictions.

- **Recall**: Ratio of true positives compared to all actual positives. Relates to how many instances of a person's identification are successful.

- **Support**: Occurences of each person in the test data set.

- **Accuracy**: Overall measure of model accuracy, taking into account average F1-Score vs total person counts.

- **Macro Average**: Weighted average of precision, recall & F1-Score, ignoring support.

- **Weighted Average**: Macro average, including support.

### Qualitative Testing

Eigenfaces are used as the basis for analysing facial imagery. The predicted faces gallery showcases the application of the ML model in classification, based upon eigenfacial data.

Eigenfaces are the principal components obtained through PCA applied to this data set. In simple terms, they capture the essential facial features & variations present in a person's face to predict who's who. They represent the most important variations in the data set ordered from most important w/ largest variation to least.

### Reliability, Efficiency, Accuracy

The model maintains an accuracy of 87% when predicting the class label (person identity) correctly.

Analysing each classes' performance individually is essential for concluding this ML model's strength. For classes with larger support values, it is clear to see that the F1-Score is much higher. This is an obvious revelation, as usually, a larger test sample size correlates to a more reliable result.

| Class           | Images | F1-Score |
|-----------------|---------------|----------|
| George Bush     | 46            | 0.94     |
| Hugo Chavez     | 15            | 0.67     |

High-efficiency in data processing can be implied from the fast prediciton time of sub-100ms.
