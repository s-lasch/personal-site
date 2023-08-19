---
math: true
---

+++
author = "Steven Lasch"
title = "kNN Classification With Cross-Validation"
date = "2023-05-26"
tags = [
    "ml",
    "classification metrics",
    "knn",
    "projects"
]
+++

This article goes into detail about the implementation of cross-validation for $k$-NN classifiers, ties, and touches on confusion matrices. Links to external references are provided, and it is also recommended to use other sources to cross-validate *my* work with that of others! 😉


### **What is Cross-Validation**

Cross-validation is a technique used to assess the performance and generalization ability of machine learning models. It involves splitting the data into multiple subsets to train and evaluate the model multiple times. Some benefits to cross-validation is that it helps estimate how well a model will perform on unseen data and can be used to compare different models. It can also be used to evaluate the model's performance and determine the optimal value of $k$ for a $k$-NN classification model.

When thinking about a $k$NN classification model, or **$k$-classifier** for short, there is the obvious problem about a tie between one or more classes. 


### **Dealing With Ties**

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/f728c1a593c4ab810308a748f91f0a7c968325ff/images/knn_tie_example.svg" 
           alt="Example of a tie in a kNN model."/>
<p align="center"><em>Fig 1. Example of a tie (image by author)</em></p>
</p>

For a $k$-classifier, ties are determined as in Fig. 1. The circle represents an area in which all data points *within the circle* count as a neighbor. In this case, the new point is equidistant from the square and the X, and since our circle defines 2 neighbors, we have a tie. This tells us that it would be a wise decision to choose a fixed value for $k$, say $k=3$ so that we can reduce the chances of a tie.

This raises a problem since our classifier selects the class label with the minimum distance from our new point, yet in this case, the distances are equal.

Functions such as `confusion_matrix()` from the `sklearn` library break these ties randomly. This is largely to prevent bias in the models. Additionally, this is why counts of truth values may be inconsistent. Obviously, the example above is a very over-simplified one, but breaking ties with limited bias is important when making predictions. Model bias is one area that is constantly under scrutiny, especially with text-based models such as GPT and BERT, and also with classification models.


### **Implementing Cross-Validation**

For this example, we will be using the [Howells dataset](https://web.utk.edu/~auerbach/HOWL.htm), available [here](https://www.statsmachine.net/databases/STAT_139/Howells.csv). It is dataset that contains craniometric measurements taken from over 2,500 human crania from 28 populations between 1965 and 1980 by Dr. William W. Howells. This example will attempt to classify either BERG males/females or NORSE males/females based on several features of the dataset. 

``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# read in the data
url = "https://www.statsmachine.net/databases/STAT_139/Howells.csv"
Howells = pd.read_csv(url)
```

Now that we have the dataset in memory, we can begin the filtering process. 
, we will filter the data to contain only people from `NORSE` or `BERG` populations. These are the locations, and the `PopSex` column splits each into male or female, hence the `F` and `M` after the population name. 

The `'GOL', 'NOL', 'BNL', 'BBH', 'XCB'` features measure different dimensions of the human cranium. The description for specific features can be found [here](https://www.rdocumentation.org/packages/TestDimorph/versions/0.4.1/topics/Howells) and more [here](https://www.sciencedirect.com/topics/medicine-and-dentistry/craniometry). These columns, excluding `PopSex`, will become our training data. The `crosstab()` function is for visualization purposes, and the same result can be obtained alternatively by using the `value_counts()` function.

``` python
# filter the data
HBNMF = Howells.query("Pop == 'NORSE' or Pop == 'BERG'").sort_values(['Pop', 'Sex'])[['HID', 'Sex', 'Pop', 'PopSex', 'GOL', 'NOL', 'BNL', 'BBH', 'XCB']]

# choose which measurements to use in classification
train = HBNMF[['GOL', 'NOL', 'BNL', 'BBH', 'XCB']].values

# choose which group labels to use in classification
trl = HBNMF['PopSex'].values

# table showing the counts of each unique value in the desired column
ct = pd.crosstab(index=HBNMF['PopSex'], columns='count').T
ct
```
``` text
PopSex      BERGF BERGM NORSEF NORSEM
col_0				
count       53    56    55     55
```

Now we can begin with the $k$ part of the $k$-classifier. Essentially, the value $k$ represents the number of neighbors for our model to account for when it is trying to classify a new datapoint. As alluded to earlier, and from [this post](https://stats.stackexchange.com/questions/43388/different-use-of-neighbors-in-knn-classification-algorithm), we don't necessarily need to choose a fixed value for $k$. 

A similar result can be obtained by drawing a circle with a given radius, $r$, such that all instances within the resulting circle count as a neighbor. For this example, we will be using the former approach, by finding an optimal number of neighbors that minimizes false positives and false negatives.

``` python
# 1 nearest neighbor
knn = KNeighborsClassifier(n_neighbors=1)

# perform cross-validation and predict labels
kcres1 = cross_val_predict(knn, train, trl, cv=5, method='predict_proba')

# convert probabilities to predicted class labels
kcres1 = np.argmax(kcres1, axis=1)

# print confusion matrix
print("Confusion Matrix (1 neighbor):\n", pd.crosstab(kcres1, HBNMF['PopSex']), sep="")
```
``` text
Confusion Matrix (1 neighbor):
PopSex  BERGF  BERGM  NORSEF  NORSEM
row_0                               
0          30     10       7       1
1          11     33       8       7
2          10      5      32      12
3           2      8       8      35
```

Because we are cross-validating, let's pick another value for $k$, say $k=3$. Here we have the following:

``` python
# 3 nearest neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# perform cross-validation and predict labels
kcres3 = cross_val_predict(knn, train, trl, cv=3, method='predict_proba')

# convert probabilities to predicted class labels
kcres3 = np.argmax(kcres3, axis=1)

# print confusion matrix
print("Confusion Matrix (3 neighbors):\n", pd.crosstab(kcres3, HBNMF['PopSex']), sep="")
```
``` text
Confusion Matrix (3 neighbors):
PopSex  BERGF  BERGM  NORSEF  NORSEM
row_0                               
0          30      9      12       1
1           9     34       2       5
2          14      5      33       9
3           0      8       8      40
```

We'll do this two more times, when $k=7$ and when $k=9$, and compare the resulting tables.

``` text
Confusion Matrix (7 neighbors):
PopSex  BERGF  BERGM  NORSEF  NORSEM
row_0                               
0          39     12       7       0
1           5     32       1       5
2           9      4      40      10
3           0      8       7      40
```
``` text
Confusion Matrix (9 neighbors):
PopSex  BERGF  BERGM  NORSEF  NORSEM
row_0                               
0          39      9       9       0
1           4     33       2       6
2          10      6      38      11
3           0      8       6      38
```

Doing some simple calculations, it seems that the optimal value of $k$ is 7. Do note that we want to choose odd values for $k$ so that we avoid ties.


### **The Resulting Matrix**

The outputs above are called **confusion matrices**. They represent the truth values for each value of $k$. The true class labels are represented by the `row_0` column. Each row in the table corresponds to a specific true class label, and the counts in each cell indicate the number of data points that were predicted to and/or actually belong to that class label. 

These truth values give us a means to evaluate our values for $k$. Refer to [Confusion Matrices](https://s-lasch.github.io/2023/05/22/Confusion-Matrices.html) for more information on confusion matrices, and [Classification Evaluation](https://s-lasch.github.io/2023/05/22/Classification-Evaluation.html) to learn about the evaluation metrics that we can use from a confusion matrix. 
