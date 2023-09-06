+++
author = "Steven Lasch"
title = "Cartesian product is everywhere"
date = "2023-08-25"
description = "This article covers a topic in discrete mathematics that often goes unnoticed in data science and data visualization: the Cartesian product."
tags = [
    "discrete mathematics",
    "math",
    "applied math",
    "data viz",
    "plotly"
]
mathjax = true
plotly = true
+++

This article covers a topic in discrete mathematics that often goes unnoticed in data science and data visualization. Here I will attempt to show the relevance of the **cartesian product** in what we do as data scientists. 

> **NOTE:** The author assumes basic understanding of set notation, and a grasp of Python basics such as list comprehension and basic data structures such as lists and sets.

## What is a Cartesian Product?

The cartesian product of two sets, $A$ and $B$, is denoted $A \times B$, and is defined as such: $$A \times B = { (a,b) \ | \ a \in A, \ b\in B }$$ In plain English, it is the set of all possible ordered pairs whose first component comes from $A$, and whose second component comes from $B$[^1].

###  Example

Here’s how we can apply that formula using Python[^2]:

```python
# define two sets
A, B = set([1,2,3]), set([1,2,3])

# apply cartesian product formula
cartesian_product = [(a,b) for a in A for b in B]
```

Which produces the following output:

```text
[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
```

## Applications

I have only listed a few of the applications of the cartesian product, though there are many.

### SQL Table Joins

Many examples of cartesian product exist in data science. One common example is an SQL **cross join**[^3].  Given two tables, a cross join applies the same formula as above to SQL tables. 

```sql
-- TABLE 1
-- create the first table
CREATE TABLE Table1 (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
);

-- insert data into the first table
INSERT INTO Table1 (name, age) VALUES ('Alice', 25);
INSERT INTO Table1 (name, age) VALUES ('Bob', 30);
INSERT INTO Table1 (name, age) VALUES ('Charlie', 28);

-- ----------------------------------------------------------------

-- TABLE 2 
-- create the second table
CREATE TABLE Table2 (
    id INTEGER PRIMARY KEY,
    product TEXT,
    price REAL
);

-- insert data into the second table
INSERT INTO Table2 (product, price) VALUES ('Phone', 599.99);
INSERT INTO Table2 (product, price) VALUES ('Laptop', 1099.99);
INSERT INTO Table2 (product, price) VALUES ('Tablet', 349.99);
```

After creating the table, we can run the following code to apply a cross join:

```sql
SELECT Table1.id, Table2.product
FROM Table1
CROSS JOIN Table2;
```

Which results in:

| **name** | **product** |
| -------- | ----------- |
| Alice    | Phone       |
| Alice    | Laptop      |
| Alice    | Tablet      |
| Bob      | Phone       |
| Bob      | Laptop      |
| Bob      | Tablet      |
| Charlie  | Phone       |
| Charlie  | Laptop      |
| Charlie  | Tablet      |

### Cartesian Plane

What if our sets are larger than just three elements, say an infinite amount? In this scenario, we can use the set of all real integers, $\mathbb{R}$, and have $\mathbb{R} \times \mathbb{R}$, (often denoted as $\mathbb{R}^3$). That is, the cartesian product of the real number line with itself ***is*** the **cartesian plane**[^4].

$$\mathbb{R} \times \mathbb{R} = \{(x,y) \ | \ x,y \in \mathbb{R}\}$$

You will find that any two numbers plugged into this general formula corresponds to an ordered pair of real numbers.

### $3$-Dimensional Plotting

We can also use the cartesian product to generate what is referred to as a **meshgrid** in 3-Dimensional plotting. In fact, Numpy's [`meshgrid()`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#:~:text=returns%20a%20meshgrid%20with%20Cartesian%20indexing) function unsurprisingly uses the cartesian product for its calculations. This involves creating a sort of blanket of points on a graph with 3 axes: $(x,y,z)$. 

In this scenario, we are dealing with $\mathbb{R} \times \mathbb{R} \times \mathbb{R} = \mathbb{R}^3$, as there are three number lines expanding indefinitely.

For this example, we will create a cube of planes. We first must start out by defining our sets. We will go for a 10x10x10 cube:

```python
# define three sets
X, Y, Z = [set(range(1, 11))] * 3

# apply cartesian product formula
cartesian_product = [(x,y,z) for x in X for y in Y for z in Z]
```

Giving us:

```text
[(1, 1, 1),
 (1, 1, 2),
 (1, 1, 3),
 (1, 1, 4),
 (1, 1, 5),
 (1, 1, 6),
 (1, 1, 7),
 (1, 1, 8),
 (1, 1, 9),
 (1, 1, 10),
 (1, 2, 1),
 (1, 2, 2),
 (1, 2, 3),
 ...
 (10, 10, 8),
 (10, 10, 9),
 (10, 10, 10)]
```

Now to create our plot, we will use `plotly.graph_objs` to display the cube, allowing for interactivity. 

```python
import plotly.graph_objs as go

# create the figure
fig = go.Figure(data=[go.Scatter3d(x=[i[0] for i in cartesian_product], # access the 'x' values
                                   y=[i[1] for i in cartesian_product], # access the 'y' values
                                   z=[i[2] for i in cartesian_product], # access the 'z' values
                                   mode='markers',
                                   marker=dict(size=5, opacity=0.8)
                                   )
                     ]
                )

fig.show()
```

<iframe title="Scatter Cube" src="https://rawcdn.githack.com/s-lasch/personal-site/6ca58f39f2bea009898daadc7d72eb43255c2c78/images/scatter_cube.html" style="height: 500px; width:100%; border: none;" ></iframe>
{{/*<plotly json="images/scatter_cube.json" height="400px">*/}}

If we want to create a plane, we just need to keep one of the axes constant, say $z$:

```python
import plotly.graph_objs as go
import numpy as np

# define three sets
X, Y, Z = set(range(1, 11)), set(range(1, 11)), np.ones_like(range(1,11) # keep z constant at z = 1

# apply cartesian product formula
cartesian_product = [(x,y,z) for x in X for y in Y for z in Z]

# create the figure
fig = go.Figure(data=[go.Scatter3d(x=[i[0] for i in cartesian_product],
                                   y=[i[1] for i in cartesian_product],
                                   z=[i[2] for i in cartesian_product],
                                   mode='markers',
                                   marker=dict(size=5, opacity=0.8)
                                   )
                     ]
                )

fig.show()
```

<iframe title="Scatter Plane" src="https://rawcdn.githack.com/s-lasch/personal-site/6ca58f39f2bea009898daadc7d72eb43255c2c78/images/scatter_plane.html" style="height: 500px; width:100%; border: none;" ></iframe>
{{/*<plotly json="https://rawcdn.githack.com/s-lasch/personal-site/cb5143d8f01e0e8bd4af940fc4e0d0207d5c43c1/images/scatter_cube.html" height="400px">*/}}

## References

[^1]: [ADS Cartesian Products and Power Sets (discretemath.org)](https://discretemath.org/ads/s-cartesian_Products_and_Power_Sets.html)
[^2]: I use Google Colab notebooks to run Python, though any environment will suffice. 
[^3]: [SQL CROSS JOIN Explained By a Practical Example (sqltutorial.org)](https://www.sqltutorial.org/sql-cross-join/)
[^4]: [Cartesian Product of Sets – The Math Doctors](https://www.themathdoctors.org/cartesian-product-of-sets/)
