[TOC]



# Introduction



# Materials and Methods

## Proposed Framework.

### Preprocessing

- Language Detection to remove non-English news 

Our dataset included news written in a language other than English. Considering the volume of the dataset and ease in the stemming, we removed all the datasets written in non-English by using language detection from langdetect. 

- Removing entries with NA

After performing the language detection, all the entries with at least one missing value were removed. At this stage, the entry with atypical format of title was also removed. 

- Stemming 

We implemented stemming to our dataset as it removes prefixes and suffixes from a word, and thus thereby reudces words to their root form. In our data preprocessing, Porter Stememr was employed, which reduces words like: "running" and "runs" to "run," and "runner" to "runner." By transforming words into a common base form, we are able to increase simplicity and decrease volume. Yet, due to the length of dataset and size of each entry, stemming was only performed within the title variable.


- Samplinng 

For the purpose of evaluating models rigorously, we created multiple test datasets. To do so, among the 5000 randomly chosen samples from the original dataset, we assigned an index for each of the entries; for instance, the index of the first 1000 entries was 0, and the index of the following 1000 entries was 1. Then, after indexing the data, entries with the same index were grouped. In our model, index 0 refers to the training dataset and indices 1, 2, 3, and 4 refer to testing sets 1, 2, 3, and 4 respectively. 

### Models

#### Logistic regression

Logistic regression is a supervised classification method that uses the Bernoulli distribution and the sigmoid function to predict the binary discrete values such as true/false and yes/no. In this work, we trained the logistic regression model with default parameters. 

#### Linear SVM

Linear SVM (Support Vector Machine) is a machine learning algorithm that separates data points into different classes by finding the best hyperplane (decision boundary) that maximizes the margin between the classes. It achieves this by identifying support vectors, which are the data points closest to the decision boundary. Linear SVM aims to find the optimal hyperplane that maximally separates the classes while minimizing classification errors. It maps the input data into a higher-dimensional space using a linear function and employs a margin-based loss function to optimize the placement of the decision boundary. The resulting model can be used for predicting the class of new, unseen data points based on their features.

#### K-nearest neighbors

K Nearest Neighbours (KNN) is a non-parametric classification algorithm with simplicity and effectiveness. The algorithm operates by aggregating the outcomes of the k nearest neighbors from the training dataset to classify a new observation. In KNN, the selection of the optimal value of k is a critical step as it directly impacts the trade-off between bias and variance. Specifically, a small value of k leads to high variance and low bias, while a large value of k results in high bias and low variance. To determine the optimal value of k, the cross-validation method is usually employed, which involves splitting the dataset into training and validation sets to assess the performance of the model on new data. Therefore, the selection of the appropriate value of k in KNN is a crucial step in achieving high predictive accuracy and robustness of the model.

#### Voting classiﬁer



#### Bagging classiﬁer



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