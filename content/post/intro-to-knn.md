---
math: true
---

+++
author = "Steven Lasch"
title = "Introduction to k-NN"
date = "2023-05-09"
tags = [
    "ml",
    "knn",
]
series = ["K-Nearest Neighbors"]
aliases = ["knn"]
+++

> **NOTE:** *all data and figures in this post are made using* `numpy` *and* `matplotlib`. 

### **Overview**

$k$-NN is a simple yet effective **supervised learning** algorithm that is used for classification and regression tasks. It is a **non-parametric** method, which means it makes predictions based on the similarity of the input data points to their nearest neighbors (NN).

The working principle behind $k$-NN is that similar data points belong to the same class. It assumes that if a particular data point is close to other data points in a feature space, it likely belongs to the same class as those neighbors.

$k$-NN uses a distance metric to determine the similarity between two data points. There are several, such as Euclidean distance—perhaps the most common—which measures the straight-line distance between two points in a multidimensional space. More information on these distance metrics and their uses is yet to come.

### **What is $k$?**

As previously mentioned, $k$-NN attempts to find the nearest neighbors of new data points by determining its closeness to other data points. To do this, our model needs to know *how many neighbors* around our data point it should account for. This is $k$.

$k$ is an important parameter that we need to define before we can make predictions. This is where $k$ optimization comes into play. Essentially, we need to find an optimal value for $k$–-one that isn’t too small nor too large. A small value for $k$ will result in the model becoming more sensitive to local variations, while a too large value for $k$ will smooth out the decision boundaries, but it could overlook finer patterns in the data.

### **Choosing $k$**

#### **Fixed-Value $k$**

This method utilizes **cross-validation** to find the optimal value of $k$. By selecting an integer value for $k$, a [confusion matrix](https://s-lasch.github.io/2023/05/22/Confusion-Matrices.html) is constructed for each feature, and [evaluation metrics](https://s-lasch.github.io/2023/05/22/Classification-Evaluation.html) are calculated to determine the number of false positives and negatives, as well as true positives and negatives. Iterating this process for different values of $k$ allows us to identify the optimal value.

By considering multiple values of $k$ and evaluating their corresponding confusion matrices, we can assess the trade-off between other performance metrics. The goal is to select a value of $k$ that strikes a balance between capturing sufficient local patterns in the data while avoiding overfitting or underfitting.

#### **Alternative Method**

An alternative method to implicitly select a value for $k$ is by utilizing a circle with a specified radius, $r$. In this approach, the number of neighbors is determined by counting all the data points that fall within the circle. Additionally, the data point of interest is placed at the center of the circle.

By employing this technique, we indirectly establish the number of neighbors based on the density of points within the defined radius. Intuitively, a larger radius will capture a larger density of points for its neighborhood, while a smaller radius includes less points.

### **Voronoi Diagrams**

A Voronoi Diagram is

> *…used to define and delineate proximal regions around individual data points by using polygonal boundaries.*
>
> – [*sciencedirect.com*](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjYns2h0av_AhU-lYkEHTf0AMAQFnoECCoQAw&url=https%3A%2F%2Fwww.sciencedirect.com%2Ftopics%2Fearth-and-planetary-sciences%2Fvoronoi-diagram%23%3A~%3Atext%3DThiessen%2520polygon%2520maps%252C%2520which%2520are%2Cpoints%2520by%2520using%2520polygonal%2520boundaries.&usg=AOvVaw333h_4NjhxXbSupF4Jt32j)

These diagrams have a [wide variety of uses](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjYns2h0av_AhU-lYkEHTf0AMAQFnoECCYQAw&url=https%3A%2F%2Fwww.irishtimes.com%2Fnews%2Fscience%2Fhow-voronoi-diagrams-help-us-understand-our-world-1.2947681%23%3A~%3Atext%3DVoronoi%2520diagrams%2520have%2520applications%2520in%2Cbased%2520on%2520exploratory%2520drill%2520holes.&usg=AOvVaw2y07KO-_HNPpHJu22OOvuW) in science, as well as in biological structures and engineering. They can be used to identify nearest airports in case of diversions, or aid mining engineers in mineral identification from drill holes. 

These diagrams represent a form of classification similar to $k$-NN. Each line represents the half-way point between any two given points. These lines intersect at vertices, which are equidistant from three points instead of two. 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/56ce30163c3accf918ce38f85f1243ad18316a32/images/voronoi_tessellation_example_transparent.svg" alt="Voronoi tessellation plain" 
           width="75%"/>
</p>

Let’s add some more color to this plot. We can really see how this resembles a $k$-NN model if we color each polygon to match the point inside. 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/56ce30163c3accf918ce38f85f1243ad18316a32/images/voronoi_tessellation_colored_transparent.svg" alt="Voronoi tessellation colored" 
           width="75%"/>
</p>

This isn’t exactly like a $k$-NN problem, though it is remarkably similar. You can see one area where this approach is very prone to overfitting—the landlocked green polygon surrounded by blue. Even though this point is actually green, it is fine-tuned to the special quirks of this randomly generated dataset.

Models that become fine-tuned to the dataset are dangerous because they can have a very high performance at training time, yet fail abysmally during inference. It is likely that the green point in the blue area is just an outlier, and so by classifying a *whole region* of outliers, we may severely limit the accuracy of our model during inference by choosing this model. 
