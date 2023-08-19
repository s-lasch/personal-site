---
title: "MonkeyType for data science"
date: 2023-06-01T10:22:10+07:00
images: [
    https://source.unsplash.com/0gkw_9fy0eQ/,
]
tags: ["data viz", "plotly", "projects"]
toc: false
---

# Background
I have always had a knack for typing quickly. During college, I came across a website called [MonkeyType](https://monkeytype.com), that describes itself as "a minimalistic and customizable typing test." This is indeed accurate, and I would go farther than that by saying it is the ***best*** typing test out there on the internet. Before you ask, no, I am not sponsored by MonkeyType to say that---I genuinely think this is the best typing test on the internet for a variety of reasons:

## **It's Fun**

Do not forget about the "customizable" part of their definition... They have about a hundred features, languages, typing modes, and varying levels of word-set difficulties. There are many features such as Live WPM, Quick Restart, and others that have become essential tools for my typing tests. 

There are others that are more aimed for typing improvement, such as ones that will end the typing test if a certain minimum WPM/accuracy threshold is crossed. Using these tools works wonders for tracking improvement and knowing which letters or strokes need improving.  


## **It's Satisfying**

Unlike most typing test websites, MonkeyType does not feel like you're playing a children's game, nor annoy you with keyboard clicks or just plain terrible CSS design if you're really picky. For those who want an adult typing test experience, MonkeyType has you covered in that it is both aesthetically pleasing and highly ergonomical.


## **It's Transformative**

As someone who has been using MonkeyType for over a year now, I can say that I have seen a steady improvement in my typing score. Growing up, I remember starting out with websites typing individual letters, followed by home-row, top-row, and bottom-row keys. These websites would have me type random gibberish at times, and I felt like I wasn't getting much experience typing a variety of words. 

I have always typed fast, typically around 90 WPM, but I plateaued for many years due to inadequate typing test websites. Using MonkeyType for the past year has helped me to start improving my speed again, and doing so in a very efficient way that isn't monotonous. After a year of off-and-on test taking, I can now achieve an average WPM of around 106 for a 1 minute test.  


## **The Best Part**

One of the most interesting parts of this website---at least to me as a data scientist---is the fact that I can download my alltime results as a `csv` file. With this and a little bit of Python, I can see my typing statistics in a whole new light. 

Below, I have used the following code to generate a report of the last year of my typing results. I used the `pandas-profiling` package to accomplish this. The following code cell was executed using [Google Colaboratory](https://colab.research.google.com).

``` python
# download data
!wget https://raw.githubusercontent.com/s-lasch/typing-tests/main/results.csv

# install pandas-profiling
!pip install pandas-profiling


# read in the data
import pandas as pd

df = pd.read_csv("results.csv", delimiter="|")


# create the report
from pandas_profiling import ProfileReport

ProfileReport(df, title="MonkeyType Report").to_notebook_iframe()
```

This generates an HTML file that contains the report. Here is the report that it generates:

<iframe
    title="MonkeyType Report"
    src="https://rawcdn.githack.com/s-lasch/s-lasch.github.io/5ffe0d494eef1bfd5cd8b03efc324448311ccddc/_posts/MonkeyType/monkey_type_report.html"
    style="width: 100%; height: 800px; border: none;"
  ></iframe>

  <br>

# **Basic Data Visualization**

Below is a simple dashboard of the statistics found in the `results.csv` file, located [in my personal repo](https://github.com/s-lasch/typing-tests), which contains the MonkeyType data, a module for plots, as well as a module for automatic data extraction using `selenium`. 

The plots were created using [plotly.express](https://plotly.com/python/plotly-express/) in Python, and are meant to be interactive. My repo for this data has the dash implementation to create an interactive dashboard. Feel free to clone it and test it out!😄

<div style="display: flex;">
  <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/3eed19519eb89af909e9da0667e5ac17ca828179/images/language_pie.svg" alt="Displays language proportions in typing tests" width="48%"/>
  <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/df59605e37393e688bfe45536cd519107eece619/images/pie_chart.svg" alt="Displays typing test mode proportions in typing tests" width="48%"/>
</div>


<div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 10px;">
  <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/3eed19519eb89af909e9da0667e5ac17ca828179/images/wpm.svg" alt="Displays statistics on words per minute (wpm) broken down by each mode" width="170%"/>
  <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/3eed19519eb89af909e9da0667e5ac17ca828179/images/accuracy.svg" alt="Displays accuracy statistics broken down by each mode" width="170%"/>
  <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/3eed19519eb89af909e9da0667e5ac17ca828179/images/raw_wpm.svg" alt="Displays raw words per minute broken down by each mode" width="170%"/>
  <img src="https://raw.githubusercontent.com/s-lasch/s-lasch.github.io/3eed19519eb89af909e9da0667e5ac17ca828179/images/consistency.svg" alt="Displays typing speed consistency broken down by each mode" width="170%"/>
</div>
