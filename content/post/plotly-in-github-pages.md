---
title: "Using Plotly in Github Pages"
date: 2023-08-23T10:14:00-04:00
tags: ["plotly", "hugo", "github pages", "plotly github"]
images: [
    https://source.unsplash.com/jrh5lAq-mIs/,
]
toc: true
---

This article assumes that the reader has at least *some* familiarity with Python, as well as limited experience in using Markdown files for static websites.

## About

Github is good for many things, and serving static websites is no exception. The drawback is that files stored on a Github repo cannot be accessed in the same way that a standard <abbr title="Content Delivery Network">CDN</abbr> would. Thus, using a link to the file directly results in this:

![text-type-results](https://rawcdn.githack.com/s-lasch/personal-site/4f62260ebb048c5365a9e30ac2f3f40e609eea2b/images/text-type-results.png)

To get around this, we need a way to serve the file as a **content-type**, which we will delve into below[^1].

## Creating a Plot

As you may be familiar with already, Plotly allows us to download our graphs as a `.html` file.  We just save our figure using the `.write_html()` function. Below is a snippet of code to create a plot and download the HTML required to build it.

```python
import plotly.express as px # interactive data viz package

# create example figure
fig = px.line(x=[1,2,3,4,5], 
              y=[10,15,25,35,50], 
              title='Simple Line Plot')

# save figure as an HTML file
fig.write_image("simple_plot.html")
```

Now that you have the file, you can download it to your desktop and open it via a browser such as Chrome or Safari. Running the following line of code should give you the same result.

```python
fig.show()
```

## Hosting the File

Now that you have the html file, you can upload it to a Github repository. All you need to do after it has been uploaded is go to [raw.githack.com](https://githack.com) and paste in the URL to the file you uploaded. That’s it.

I would recommend using the production link, as it does encrypt some of your personal repository information, which is beneficial if you enjoy as much privacy as possible when hosting files.

## Displaying in Markdown

Markdown is a versatile language for websites, and it has added HTML rendering support. To display our graph, we can use an HTML `<iframe>`. 

Navigate to a blog post with the extension `.md` to add your plot. Paste the following code in the blog post:

```html
<iframe title="" 
        src=""
        style="width: 100%; 
               height: 500px; 
               border: none;">
</iframe>
```

Use the URL from githack as the `src` parameter, and adjust sizing and styling as needed. 

## Troubleshooting

A common error when using Hugo, a static site generator, is that HTML in a markdown file is not rendered by default[^2]. This is not the case in Jekyll, and it is most often an easy fix. It just involves adding an additional parameter to your site’s config file. 

### Hugo

For a site using a Hugo template, navigate to `/config.toml` and add the following parameter anywhere:

```toml
[markup.goldmark.renderer]
unsafe= true
```

## Result

Now, using the same code, we can clearly see that it works:

<iframe title="Crania Data Visualization"
		src="https://rawcdn.githack.com/s-lasch/s-lasch.github.io/5ac4853d6acfc41a3e45f33062fab314eee8eaf9/_includes/scatter_matrix.html"
		style="height: 820px; width:110%; border: none;"></iframe>

## References

[^1]: [How to display plotly graph on github pages? - 📊 Plotly Python - Plotly Community Forum](https://community.plotly.com/t/how-to-display-plotly-graph-on-github-pages/44398)
[^2]: [Raw HTML getting omitted in 0.60.0 - support - HUGO (gohugo.io)](https://discourse.gohugo.io/t/raw-html-getting-omitted-in-0-60-0/22032)

[🔼 back to top](#about)
