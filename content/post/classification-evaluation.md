+++
author = "Steven Lasch"
title = "Classification Evaluation Metrics"
date = "2023-05-22"
description = "Displays the math behind some of the most common evaluation metrics fora  classification problem in machine learhing."
tags = [
    "ml",
    "classification metrics",
    "knn"
]
mathjax = true
+++

Below are some of the most common evaluation metrics used for classification models. For a binary classifier, this is very straight-forward, but for a multi-class classifier, we need some more intuition about a confusion matrix. For a quick reminder, we will be using the following matrices from my previous article about [Confusion Matrices](https://s-lasch.github.io/2023/05/22/Confusion-Matrices.html):

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/main/images/binary__confusion_matrix.png" 
           alt="Binary Confusion Matrix"/>
<p align="center"><em>Fig 1. Binary confusion matrix (image by author)</em></p>
</p>

<br>
<br>

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/main/images/multi_class_confusion_matrix.png" alt="Multi-class Confusion Matrix" 
           width="50%"/>
<p align="center"><em>Fig 3. Multi-class confusion matrix (image by author)</em></p>
</p>


### **Precision**

Precision measures how precise or accurate the positive predictions of a classifier are. It is the ratio of true positives to the sum of true positives and false positives. Precision indicates how many of the instances *predicted* as positive are *actually positive.*

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/3bceb851cbbbde4cb5660a4ca31e25864ebfd049/images/precision.svg" />
</p>

Let's calculate precision for the `BERGF` class. We need the true positives, \( M_{1,1} \), and the false positives, \( M_{1,2} + M_{1,3} + M_{1,4} \)
<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/2cd26b6067f845aad6e9c9fa5fa0c1912a07d438/images/bergf_prec.svg" />
</p>

As a rule of thumb, you should optimize your model for precision when you want to decrease the number of false positives.


### **Recall**

Recall, also known as sensitivity or true positive rate, measures the ability of the classifier to correctly identify positive instances. It is the ratio of true positives to the sum of true positives and false negatives. Recall indicates the proportion of actual positive instances that are correctly predicted. 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/3bceb851cbbbde4cb5660a4ca31e25864ebfd049/images/recall.svg" />
</p>

Let's calculate recall for the `BERGM` class. We need the true positives, \( M_{2,2} \), and the false negatives, \( M_{1,2} + M_{3,2} + M_{4,2} \)
<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/2cd26b6067f845aad6e9c9fa5fa0c1912a07d438/images/bergm_rec.svg" />
</p>

You should optimize your model for recall if you want to decrease the number of false negatives.


### **Accuracy**

Accuracy measures of how often the classifier correctly predicts the class of an instance. It is the ratio of the correctly classified instances to the total number of instances in the dataset. Higher accuracy indicates a better overall performance of the classifier. In the Multi-class Confusion section, we calculated the model's overall accuracy by adding all the correct predictions and dividing it by the number of total samples. 


### **F1-score**

Another common metric is the F1-score, which rates the successfulness of a classification model. It is the [harmonic mean](https://www.investopedia.com/terms/h/harmonicaverage.asp) of both the precision and recall for a given class. The benefit of a harmonic mean rather than a simple average is that it punishes extreme values. If a classification model has a precision of 1.0 and a recall of 0.0, has a simple average of 0.5, and an F1-score of 0. Below is the formula to calculate the F1-score.

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/2cb57db2189794a4b1ff610a7ffedbb45701de89/images/general_f1_score.svg" />
</p>

For a multi-class classifier, instead of calculating an overall F1-score, we must calculate the F1-score for each class individually, in a one-versus-rest fashion. So, we just need to apply the formulas for precision, recall, and F1-score for each class.

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/2cb57db2189794a4b1ff610a7ffedbb45701de89/images/multiclass_f1_score.svg" />
</p>

The following equations demonstrates how we can calculate the F1-scores for each class in the multi-class confusion matrix from above. 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/main/images/multi_class_confusion_matrix.png" width="50%"
           alt="Multi-class Confusion Matrix"/>
</p>

First we need to find the precision and recall for the `BERGF` class, then apply the above equation to find the F1-score:

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/d9fdd5dbf0328e874a38afa11b35ecdb3f2709aa/images/f1_bergf.svg" />
</p>

<br>

Then, we calculate precision and recall for the remaining classes:

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/d9fdd5dbf0328e874a38afa11b35ecdb3f2709aa/images/prec_and_recall_rest.svg" />
</p>

<br>

Finally, we apply the F1-score equation to each class:

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/d9fdd5dbf0328e874a38afa11b35ecdb3f2709aa/images/f1_score_rest.svg" />
</p>
