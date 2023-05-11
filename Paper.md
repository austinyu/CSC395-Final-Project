[TOC]



# Introduction



# Materials and Methods

## Proposed Framework.

### Preprocessing

- Language Detection to remove non-English news 

Our dataset included news written in a language other than English. Considering the volume of the dataset and ease in the stemming, we removed all the datasets written in non-English by using language detection from langdetect. 

- Removing entries with NA
After performing the language detection, all the entries with at least one missing value were removed. 

- Stemming 

- Samplinng 

### Models

#### Logistic regression

Logistic regression is a supervised classification method that uses the Bernoulli distribution and the sigmoid function to predict the binary discrete values such as true/false and yes/no. In this work, we trained the logistic regression model with default parameters. 


#### Linear SVM



#### K-nearest neighbors

K Nearest Neighbours (KNN) is a non-parametric classification algorithm that has been extensively used in many scientific fields due to its simplicity and effectiveness. The algorithm operates by aggregating the outcomes of the k nearest neighbors from the training dataset to classify a new observation. In KNN, the selection of the optimal value of k is a critical step as it directly impacts the trade-off between bias and variance. Specifically, a small value of k leads to high variance and low bias, while a large value of k results in high bias and low variance. To determine the optimal value of k, the cross-validation method is usually employed, which involves splitting the dataset into training and validation sets to assess the performance of the model on new data. Therefore, the selection of the appropriate value of k in KNN is a crucial step in achieving high predictive accuracy and robustness of the model.

#### Voting classiﬁer



#### Bagging classiﬁer



#### RNN



## Benchmark Algorithms.



## Datasets



## Performance Metrics.

### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP+TN+FP+FN}
$$

### Recall

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

### Precision 

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

### F1-score 

$$
\text{F1} = 2 \frac{\text{Precision}\times \text{Recall}}{\text{Precision}+ \text{Recall}}
$$

# Results and Discussion





# Conclusion 