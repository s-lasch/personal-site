+++
author = "Steven Lasch"
title = "Confusion Matrices"
date = "2023-05-22"
description = "Calculate the precision, accuracy, recall, and f-score of a classification problem using a confusion matrix."
tags = [
    "ml",
    "knn",
    "classification metrics"
]
mathjax = true
+++

Obviously, our goal is to model our data as accurately as possible. For classification problems such as logistic regression and kNN, this is where **confusion matrices** are useful. They will make calcuating the precision, recall, accuracy, and other metrics possible, and it is not difficult to fully understand how they work.


### **Confusion Matrices**
A **confusion matrix** is the primary way to evaluate the results of a classification model. If the previous explanations did not suffice, refer to my previous articles on [Binomial](https://s-lasch.github.io/2023/03/16/Binomial-Logistic-Regression.html) and [Multinomial](https://s-lasch.github.io/2023/05/01/Multinominal-Logistic-Regression.html) logistic regression models for more information about classification models. 

In both binary and multi-class confusion matrices, the diagonal values represent true positives and true negatives, while the non-diagonal values represent false positives or false negatives depending on their positions. The rows represent the predicted values, and the columns represent the truth values, which are the original values.


### **Truth Value Reference**

To make things more clear in the sections to come, below are the different truth values along with their meanings:
* **True Positive (TP):** The model correctly predicts a true value.
* **True Negative (TN):** The model correctly predicts a false value.
* **False Positive (FP):** The model incorrectly predicts a true value when it should have been false.
* **False Negative (FN):** The model incorrectly predicts a false value when it should have been true.


### **Binary Confusion**

In a binary confusion matrix, we are only dealing with two classes. By comparing the values across different confusion matrices, you can assess if there is a reduction in misclassifications. This works for all classification models, including logistic regression, or kNN models. Fig. 1 below demonstrates what a binary confusion matrix looks like, as well as how to interpret the values inside.

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/main/images/binary__confusion_matrix.png" 
           alt="Binary Confusion Matrix"/>
<p align="center"><em>Fig 1. Binary confusion matrix (image by author)</em></p>
</p>

Fig. 2 uses the results of a pregnancy test to further drive home a binary confusion table, as well as understanding the truth values TP, TN, FP, and FN. 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/main/images/truth_table_example.png" 
           alt="Pregnancy confusion matrix example."/>
<p align="center"><em>Fig 2. Example of binary confusion matrix with pregnancy results (image by author)</em></p>
</p>


### **Multi-class Confusion**

Multi-class confusion matrices get increasingly complex with the more classes we add. In our example, we will be using four classes, and a kNN algorithm for classification. Fig. 3 represents this type of confusion matrix. 

Typically, the first step in interpreting a confusion matrix should start with identifying positive and negative classes. For our case, we are attempting to classify unseen data as belonging to one of four classes: `BERGF`, `BERGM`, `NORSEM`, and `NORSEF`. Since it would make no sense to favor one population over another, we will need to calculate accuracy, precision, recall, etc. for each class.

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/main/images/multi_class_confusion_matrix.png" alt="Multi-class Confusion Matrix" 
           width="50%"/>
<p align="center"><em>Fig 3. Multi-class confusion matrix (image by author)</em></p>
</p>

This classification table, (we'll call it $M$), provides a decent amount of information: 
* Judging by the diagonal values, the model correctly predicted 130 correct samples out of the 219 total. This means that the **overall accuracy is 59.3%**, a terrible score.
* The model confused NORSE females with BERG females and NORSE males. This is evident with the values $M_{3,1} = 10$ (row 3 column 1) and $M_{3,4} = 12$. 
* To determine the extent to which the model should be optimized, we need to use these values to calculate some metrics. 

See the [Classification Evaluation](https://s-lasch.github.io/2023/05/22/Classification-Evaluation.html) for a detailed explanation of how we can utilize truth values to assess the evaluation of a classification model.
