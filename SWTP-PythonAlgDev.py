# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# # Using Python for Algorithm Development (Presentation mode: press spacebar to advance)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Author: Andrew Holmgren<br>
# Contact: <email>adholmgren@gmail.com<email>  
# [github link](https://github.com/adholmgren/python_algdev)
# -

# If you have a Google account you can open the code in Google colab. Colab is an environment that will give free resources (you can get a GPU or a TPU for up to 12hrs). 
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adholmgren/python_algdev/blob/master/SWTP-PythonAlgDev.ipynb)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Meant as a guide for those who want to transition out of MATLAB, or just generally explore Python for scientific computing. Broken up as, firstly:
# 1. General Python use
#  * Jupyter(lab)
#     * Suggested extensions
#     * Markdown
#  * Python vs MATLAB
#  * Example libraries and capabilities

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Secondly:  
# 2. Speed up code and reduce computational bottlenecks
#   * Python-like code with outsourced optimization
#     * Python numpy
#     * Numba
#     * Cython
#   * How to integrate statically typed languages
#     * Fortran (f2py and fortran magic)
#     * C/C++ (CPython)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# * What I won't cover
#  * Python classes (dunder, instance method, class method, static method)
#  * Itertools (really cool)
#  * Imports (absolute, relative, practices)
#  * PEP8 (Python is white-space)

# + {"slideshow": {"slide_type": "slide"}, "toc-hr-collapsed": false, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ## From MATLAB to Python

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# This sections covers
#   1. Jupyter as editor
#   2. MATLAB to Python
#   3. Python capabilities (no toolboxes!)

# + {"slideshow": {"slide_type": "slide"}, "toc-hr-collapsed": false, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ### General Python/Jupyter use

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# This subsection covers general comments on using Python and Jupyter as an effective development environment.

# + {"slideshow": {"slide_type": "slide"}, "toc-hr-collapsed": true, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Jupyter(lab) how-to and favorite extensions

# + {"slideshow": {"slide_type": "subslide"}, "toc-hr-collapsed": false, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# **Jupyter** <br>
# "Project Jupyter exists to develop open-source software, open-standards, and services for interactive computing across dozens of programming languages."<br>
# Original language kernels: <br>
# * Ju(lia)
# * Pyt(hon)
# * [e]R

# + {"slideshow": {"slide_type": "subslide"}, "toc-hr-collapsed": false, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Kernels for many languages now: <br>
# * Ruby
# * C++
# * Fortran
# * Haskell
# * Rust
# * MATLAB
# * Brainf*&# (++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.)
# * and many more...

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Jupyter Extensions to try

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Jupyter extensions give more functionality to a notebook, they can easily be installed with conda.<br>
# ```conda install -c conda-forge jupyter_contrib_nbextensions``` <br>
# once installed the easiest way to enable extensions is with the [configurator](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator) <br>
# ```conda install -c conda-forge jupyter_nbextensions_configurator``` <br>

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Some personal favorites for jupyter notebook are:
# * Table of Contents (highly suggested for navigating this document)
# * Collaspable Headings
# * Codefolding
# * Execute Time (how long it takes a cell to execute)
# * Notify (sends a message to browser when cell done evaluating cell, really nice for neural networks)
# * Scratchpad (can try input without adding cells and messing up notebook organization)
# * Variable inspector (personally don't like this, but others do)
# * RISE (presentation/slideshow of notebook)

# + {"slideshow": {"slide_type": "skip"}, "toc-hr-collapsed": true, "hideCode": false, "hidePrompt": false, "heading_collapsed": true, "cell_type": "markdown"}
# ##### Markdown

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# Markdown is, arguably, an inconsequential part of the notebook if all you're looking for is to rapidly prototype and test some code. However, if you want to make comments, tell a story, embed formatted LaTeX equations, and other things to actually motivate your work so that someone other than you knows what's going on then it's a great tool that's easy to start using. Many of you may have already been exposed to markdown if you've been reading (or making) github pages, or maybe you're a fanatic redditor and already know all the tricks of the trade. 
#
# **tldr:** markdown is any easy way to do html and is easily supported by jupyter to make good looking documentation 

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# ###### Markdown reference and tips

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# This [github page](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) and this [site from the Markdown author](https://daringfireball.net/projects/markdown/basics) have good descriptions of how to use Markdown. The most common uses are listed below

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# ###### **Headers**

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# Headers are made with 
#     # Header 1
#     ## Header 2
#     ...
#     Heading 1
#     =========
#     Heading 2
#     ---------

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# ###### **Lists**

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# (dots ⋅⋅ indicate whitespace)
#     1. First ordered list item
#     2. Another item
#     ⋅⋅* Unordered sub-list. 
#     1. Actual numbers don't matter, just that it's a number
#     ⋅⋅1. Ordered sub-list
#     4. And another item.
#
#     ⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).
#
#     ⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
#     ⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
#     ⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)
#
#     * Unordered list can use asterisks
#     - Or minuses
#     + Or pluses

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# 1. First ordered list item
# 2. Another item
#   * Unordered sub-list.  
#
#
# 1. Actual numbers don't matter, just that it's a number  
#     1. Ordered sub-list  
# 4. And another item.
#
#     You can have properly indented paragraphs within list items. 
#    
#     Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).
#
#     To have a line break without a paragraph, you will need to use two trailing spaces (can use `<br>` of course).  
#     Note that this line is separate, but within the same paragraph.  
#     (This is contrary to the typical GFM (Git Flavored Markdown) line break behaviour, where trailing spaces are not required.)
#
#
#   * Unordered list can use asterisks
#   - Or minuses
#   + Or pluses

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# ###### **Code Highlighting**

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
#     Inline `code` has `back-ticks around` it.

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# Inline `code` has `back-ticks around` it.

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# Blocks of code are either fenced by lines with three back-ticks \`\`\`, or are indented with four spaces. I recommend only using the fenced code blocks -- they're easier and only they support syntax highlighting.

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
#     ```javascript
#     var s = "JavaScript syntax highlighting";
#     alert(s);
#     ```
#
#     ```python
#     s = "Python syntax highlighting"
#     print s
#     ```

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# ```javascript
# var s = "JavaScript syntax highlighting";
# alert(s);
# ```
#  
# ```python
# s = "Python syntax highlighting"
# print s
# ```

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# ###### **Emphasis**

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
#     Emphasis, aka italics, with *asterisks* or _underscores_.
#
#     Strong emphasis, aka bold, with **asterisks** or __underscores__.
#
#     Combined emphasis with **asterisks and _underscores_**.
#
#     Strikethrough uses two tildes. ~~Scratch this.~~

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# Emphasis, aka italics, with *asterisks* or _underscores_.
#
# Strong emphasis, aka bold, with double **asterisks** or __underscores__.
#
# Combined emphasis with **asterisks and _underscores_**.
#
# Strikethrough uses two tildes. ~~Scratch this.~~

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# ###### **Latex**

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
#     Inline latex is done with $e^{i\pi} + 1 = 0$

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# Inline latex is done with $e^{i\pi} + 1 = 0$

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
#     Expressions on their own line are surrounded by $$:
#     $$e^x=\sum_{i=0}^\infty \frac{1}{i!}x^i$$
#

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hidden": true, "cell_type": "markdown"}
# Expressions on their own line are surrounded by \$\$:
# $$e^x=\sum_{i=0}^\infty \frac{1}{i!}x^i$$
#

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Basic Jupyter hot keys

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# When editing a cell, press `esc` to leave and `return` to enter cell

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# `shift+return` evaluates cell and moves to next cell `ctrl+return` evaluates cell and stays at cell

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# `b` makes cell below `a` makes cell above `dd` deletes cell

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# When in command mode, i.e. not editing the cell, use `m` to toggle cell to markdown, `y` to toggle cell to code (associated with kernel), and `r` to make it raw code

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# **Ultra tip**: When in a function, use shift+Tab to show the docstring for the function (i.e. the help and info for the function).

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
np.reshape()

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Other helpful Jupyter tools

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ###### magic commands

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Jupyter has "[magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)" that perform a task at the cell level `%%` or at the line level `%`

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Timing, use `time` (does one-time timing) or `timeit` (statistics from many timings)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %%time
foo = [i*10+j+1 for i in range(10) for j in range(10)]  # "tally-count" to 100 in base-10
foo.append(3.141592)
print(foo)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
# %timeit small_list = [i for i in range(100)]

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %time large_list = [j for j in range(10000)]

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Quick profile, use `prun`

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %%prun
total = 0
N = 100000
for i in range(5):
    L = [j ^ (j >> i) for j in range(N)]
    total += sum(L)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# `%matplotlib notebook` will set all executions of matplotlib to have 'interactive' plots with the nbagg backend
#
# `%matplotlib inline` With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ###### debugging

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# If you're used to using a graphical debugger, such as MATLAB's editor or other IDE, then using pdb may be a big step in unfamiliarity. (If you're used to gdb then pdb is extremely similar and you'll be fine.) IBM has developed a graphical debugger called [pixiedust](https://medium.com/ibm-watson-data-lab/the-visual-python-debugger-for-jupyter-notebooks-youve-always-wanted-761713babc62) that you can use. You should be able to just use pip to install and then you just need to import as such
# ```python
# import pixiedust
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Let's try it on a simple function that finds the maximum value in a list

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import pixiedust  # must be own cell for some reason

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import random

def find_max(values):
    max_val = 0
    for val in values:
        if val > max_val:
            max_val = val
    return max_val


# + {"pixiedust": {"displayParams": {}}, "slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
# %%pixie_debugger
x = random.sample(range(100), 10)
m = find_max(x)
print(f'max value is {m}')

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Jupyter and version control

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Jupyter files (.ipynb) end up as large json files. As such, even the outputs get saved as pieces of json. For example, the section of the notebook for the previous code cell looks like
# ```json
# {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {},
#    "outputs": [],
#    "source": [
#     "import pixiedust\n",
#     "\n",
#     "import random\n",
#     "def find_max (values):\n",
#     "    max = 0\n",
#     "    for val in values:\n",
#     "        if val > max:\n",
#     "            max = val\n",
#     "    return max"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "metadata": {
#     "pixiedust": {
#      "displayParams": {}
#     }
#    },
#    "outputs": [],
#    "source": [
#     "%%pixie_debugger\n",
#     "x = random.sample(range(100), 10)\n",
#     "m = find_max(x)\n",
#     "print(m)"
#    ]
#   }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# * Any plain-text general merge is going to freak out. If you plan to work collaboratively with anyone, you should setup and use jupytext. 
# * In short, jupytext converts the json file to a more readable file (such as the "percent" format that VSCode, Hydrogen, and PyCharm IDEs are adopting). 
# * Turns out, once jupytext is setup, you can even just use the easier formated file in the notebook itself (set a pairing standard for the team to make sure everyone on same page &mdash; if you aren't going to couple the notebook with an VSCode etc. then I suggest the light format).

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# #### Top Python libraries to use

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# Most of these come with a base anaconda install

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# ##### General numerics
# [numpy](http://www.numpy.org/)    
# [scipy](https://www.scipy.org/)    
# [statsmodels](https://statsmodels.org)    
# [tqdm](https://github.com/tqdm/tqdm) makes a progress bar on an iterable

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# ##### Image processing
# [scikit-image](https://scikit-image.org/)  
# [opencv](https://opencv-python-tutroals.readthedocs.io/en/latest/)  
# [Pillow](https://pillow.readthedocs.io/en/stable/)

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# ##### Data 
# [pandas](https://pandas.pydata.org/) load an manipulate text/csv data  
# [scrapy](https://scrapy.org/) extract data from websites  
# [dask](http://docs.dask.org/en/latest/)  big data and task scheduling  
#

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# ##### Machine learning (including neural nets)
# [scikit-learn](https://scikit-learn.org/stable/)  good starting point for testing out a concept  
# [Pytorch](https://pytorch.org/)  Facebook's version of neural net  
# [Tensorflow/Keras](https://www.tensorflow.org/)  Google's version of neural net  
# [PyMC4](https://www.tensorflow.org/probability/install) (part of TF prob)  
# [GPFlow](https://github.com/GPflow/GPflow)  Gaussian process with TF backend .   
# [nltk](https://www.nltk.org/)  Natural Language Tool Kit  
# [SpaCy](https://spacy.io/)  More NLP  
# [PyFlux](https://pyflux.readthedocs.io/en/latest/)  Time series analysis  
#

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# ##### Plotting
# [matplotlib](https://matplotlib.org/) standard for basic plotting  
# [seaborn](https://seaborn.pydata.org/)  works on top of matplotlib  
# [plotly](https://plot.ly/python/)  great for animated or interactive plots  
# [bokeh](https://bokeh.pydata.org/en/latest/)  also does interactive (I'm not as familiar with it)

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ### Python vs MATLAB

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
#  1. Indexing and arrays
#  2. Multidimensional arrays
#  3. Loops
#  4. Beware of..

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Arrays and Indexing

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Build an array

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# * It's easy to build lists or arrays in Python. 
# * One significant difference between MATLAB and Python is that 1D Python lists and arrays are truly 1D, whereas in MATLAB a 1D array is always at least 2D with a singleton dimension (an artifact of it's original intention of being a Matrix Laboratory).

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# * Python vs. MATLAB: Make a list from 0..100 by 20 <br>
#
# MATLAB
# ```MATLAB
# x = 0:20:100;
# x = linspace(0, 100, 6);
# ```
#
# Python
# ```Python
# x = np.arange(0, 101, 20)  # using numpy array
# x = np.linspace(0, 100, 6)  # using a numpy linspace
# x = range(0, 101, 20)  # using python generator
# x = [a*20 for a in range(0, 6)]  # list comprehension
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
# create array from 0 to 100 in steps of 20
import numpy as np
x = np.arange(0, 101, 20)  # using numpy array
print(f'numpy arange: {x}')
x = np.linspace(0, 100, 6, dtype=np.int)  # using a numpy linspace
print(f'numpy linspace: {x}')
x = range(0, 101, 20)  # using python array
print(f'python range method: {x}')  # print doesn't list out elements because range is a generator
x = [a*20 for a in range(0, 6)]  # list comprehension (can make really succinct code, not here obviously)
print(f'list comprehension: {x}')

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Index an array

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# * Indexing is pretty similar, especially with [numpy indexing](https://docs.scipy.org/doc/numpy-1.14.1/reference/arrays.indexing.html), 
# * But.. Python is 0-based (first element is accessed with [0]) vs. MATLAB 1-based. 
# * And.. Last element in slice is exclusive. Slicing is more similar to e.g. C++ std library where the last slice of an iterable is non-inclusive.
# ```Python
# x[0:3] == x[0, 1, 2]
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# * Comparisons of some typical indexing
#
# MATLAB
# ```MATLAB
# x(1:2:3)
# x([1, 3])
# x(end:-1:end-2)  % access backwards from end
# x(3:end)  % go from 3rd element to last
# x(:)  % slice all, "that's what you do"
# ```
#
# Python
# ```Python
# x[0:3:2]  # start slice from 0, increment two at a time
# x[[0, 2]]  # numpy only: explicitly access element numbers
# x[-1:-4:-1]  # access backwards from end
# x[2:]  # go from 3rd element until end
# x[:]  # start at 0 until end implicit steps of 1
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
x = np.arange(0, 101, 20)
print(f'1st and 3rd: {x[0:3:2]}')
print(f'1st and 3rd (implicit): {x[:3:2]}')
print(f'access first and third directly: {x[[0, 2]]}')
print(f'count back last three: {x[-1:-4:-1]}')
print(f'3rd element to end: {x[2:]}')
print(f'all of x: {x[:]}')

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Test your knowledge. 

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ```python
# x = np.arange(0, 5)  # [0, 1, 2, 3, 4]
# print(x[:-1])
# ```
# Will the output be  
# a)
# ```python
# [0, 1, 2, 3, 4]
# ```  
# b)
# ```python
# [0, 1, 2, 3]
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
x = np.arange(0, 5)
print(x[:-1])

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# The answer is b!

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Multidimensional Arrays

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Multidimensional arrays work just how you'd expect them to, especially in numpy framework.

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
import numpy as np  # make sure numpy is imported
eye_matrix = np.eye(4)  # make 4x4 identity matrix
print(f'eye(4):\n{eye_matrix}')

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Use slicing operation to assign new numbers to matrix elements.  

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
eye_matrix[:-1, -1] = 2  # How will the matrix change?
print(f'slice assignment:\n{eye_matrix}')

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Another caveat, coming from the MATLAB world, is that python is row-major oriented instead of column-major oriented. For those unfamiliar with row-major vs. column-major, it comes into play with how arrays are arranged in memory. Since memory is not 2-dimensional, or n-dimensional, the arrays are actually stored "linearly" in memory. 
# <figure>
#   <img src="https://eli.thegreenplace.net/images/2015/row-major-2D.png" alt="Row major"/>
#   <figcaption>Row major (credit Eli Bendersky).</figcaption>
# </figure>

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# <figure>
#   <img src="https://eli.thegreenplace.net/images/2015/column-major-2D.png" alt="Col major"/>
#   <figcaption>Column major (credit Eli Bendersky).</figcaption>
# </figure>

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
import numpy as np
row_mat = np.arange(16, dtype=np.int).reshape((4, 4))
print(f'This is the row_mat matrix (default Python):\n{row_mat}\n'
      f'and this is the 5th element in memory (i.e. row_mat[4]):'
      f'\n{row_mat.flat[4]}')
col_mat = np.arange(16, dtype=np.int).reshape((4, 4), order='F')
ind_5 = col_mat[np.unravel_index(4, (4, 4), order='F')]
print(f'This is the col_mat matrix (MATLAB):\n{col_mat}\n'
      f'and this is the 5th element in row-view  (i.e. col_mat[4]):\n'
      f'{col_mat.flat[4]}\n'
      f'and this is 5th element in column order:\n{ind_5}') 

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Test your knowledge

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Swap rows or columns

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
row_mat = np.arange(16, dtype=np.int).reshape((4, 4)); print(f'row_mat:\n{row_mat}')
test_a = row_mat.copy()
test_a[[0, 2]] = test_a[[2, 0]]; print(f'test_a:\n{test_a}')
test_b = row_mat.copy()
test_b[:, [0, 2]] = test_b[:, [2, 0]]; print(f'test_b:\n{test_b}')

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Loops

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# In Python, certain objects are known to be "iterable" and an iterable object can be looped in a for loop. Coming from MATLAB world, if I want to loop over a matrix I would do something like this:
# ```MATLAB
# for j=1:size(my_matrix, 2)
#     for i=1:size(my_matrix, 1)
#         my_matrix(i,j)
#     end
# end
# ```
# I could do something similar in python...

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
import numpy as np
my_matrix = np.arange(4).reshape((2, 2))
print(f'my_matrix:\n{my_matrix}')
for i in range(my_matrix.shape[0]):
    for j in range(my_matrix.shape[1]):
        print(my_matrix[i, j])

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# There's nothing wrong with doing it this way. However, I want to point out that the matrix itself is iterable. There may be cases when this is cleaner. One caveat: don't try to modify the thing you are iterating (i.e. use the iterable itself for a nice clean reference, but use indexing if you want to modify the thing).

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
for row_array in my_matrix:  # iterate over rows [0, 1], [2, 3]
    for row_elem in row_array:  # iterate over row elements
        print(f'{row_elem} ')

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# If I wanted both value and index information I could, alternatively, enumerate the objects. Enumerating probably seems more indirect than just looping over indices, but when you get to more complicated objects (not just numbers in a matrix, a set of training images for example) you'll find a the object iterable and enumeration approach to be very convenient.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
my_matrix_now = my_matrix.copy()
for i, row_array in enumerate(my_matrix):
    for j, row_elem in enumerate(row_array):
        my_matrix_now[i, j] = row_elem**2
print(f'my_matrix_now:\n{my_matrix_now}')

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Another option: use numpy's nditer

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Be careful of...

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Some typical [gotchas](https://docs.python-guide.org/writing/gotchas/) are (I'll cover first one):
#   * Assigning does not copy. 
#   * Default function arguments to functions are mutable (should use None as a sentinel value if need other functionality).
#   * Late binding closures.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
a = np.arange(2, 5)  # a=[2, 3, 4]
b = a  # in MATLAB this makes a copy by default
a += 3  # now add some numbers to a
b += 1  # add 1 to b
# what is b? [3, 4, 5]?
print(b)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Same thing again, but this time add 1 right away

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
a = np.arange(2, 5)
b = a + 1
a += 3
print(b)

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# What's the difference? In the first case, object b is assigned to have the same identity as a. So anything done to b also applies to a. In the second case, numpy is doing implicit broadcasting (adding an array of 1s to match size of a) and saves the result as a new array.

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
a = np.arange(2, 5)
b = a  # in MATLAB this makes an array copy by default, but in python thiis just copies assignment
c = a + 1
print(a is b)
print(a is c)
# note: can now reassign a and b does not follow, not an attached reference
a = np.arange(2, 5)
print(a is b)

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Another note: if this were just a python list then you couldn't even do c = a + 1 without making your own class

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
p = list(range(2, 5))
c = p + 1  # native lists do not broadcast


# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# You'd have to make a class in order to add a value to a python list

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
class lame_numpy():
    def __init__(self):
        self.arr = []
    
    def brange(self, *args):
        if len(args) < 1:
            raise ValueError('Must be more than 1 argument')
        if len(args) < 2:
            self.arr = [i for i in range(args[0])]
        elif len(args) < 3:
            self.arr = list(range(args[0], args[1]))
        elif len(args) < 4:
            val_now = args[0]
            while (val_now < args[1]):
                val_now += args[2]
                self.arr.append(val_now)
        else:
            raise ValueError('Too many arguments')
        return self
    
    def __add__(self, value):
        for i in range(len(self.arr)):
            self.arr[i] += value
        return self
        
    def __str__(self):
        return f'{self.arr}'


# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
a = lame_numpy()
a.brange(2, 5)
print(a)
a = a + 3
print(a)

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ### Capabilities and examples of Python

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# This section shows some of the common use libraries out there for Python. Personally, I've yet to come across a MATLAB function that isn't included in a python library somewhere. Many times, the Python libraries do the task better -- **bold statement**.

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Scikit-image (skimage)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# skimage is a library that comes with anaconda, so there should be little setup required. If you make a new anaconda environment, beyond the base environment, you'll want to install it into the new environment but it is as easy as running the following from the command line (or anaconda prompt in Windows)
#
# `conda install -c conda-forge scikit-image`
#
# You can also use pip to install, but anaconda is the easiest and most reliable method.
#
# `pip install scikit-image`
# -

# The below is example is adapted from, and more for info see, the [documentation](http://scikit-image.org/docs/dev/auto_examples/filters/plot_cycle_spinning.html).

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Cyclic Wavelet Denoise of Cat Picture

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Use these modules

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Import cat picture and add noise

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# get image and add noise
original = img_as_float(data.chelsea()[100:250, 50:300])  # import and crop chelsea the cat
sigma = 0.155  # amount of noise
noisy = random_noise(original, var=sigma**2)  # add noise using imported skimage module
psnr_noisy = compare_psnr(original, noisy)  # psnr of noisy image

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
# plot images
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), sharex=False, sharey=False)
ax = ax.ravel()
ax[0].imshow(original); ax[0].axis('off')
ax[0].set_title('Original image')
ax[1].imshow(noisy); ax[1].axis('off')
ax[1].set_title('Noisy\nPSNR={:0.4g}'.format(psnr_noisy))

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# For denoising algorithm want to minimize this cost function
# $$\min_{x_{dn}} \left[\lVert x - x_{dn} \rVert_2^2 + \lambda\lVert Wx_{dn}\rVert_1\right]$$
# where $W$ is a linear operator, such as gradient or wavelet, $\lambda$ is regularization parameter, $x$ is the noisy measurement and $x_{dn}$ is the denoised estimate. The following example shows denoising using total variation, where $W$ is a gradient operator, and using wavelets, where $W$ is a discrete wavelet transform.

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Add in the skimage restoration module

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet, cycle_spin, denoise_tv_bregman
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr
from skimage.filters import sobel
from skimage.color import rgb2gray

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Get noisy image and denoise

# + {"slideshow": {"slide_type": "fragment"}, "code_folding": [], "hideCode": false, "hidePrompt": false}
# get image and add noise
original = img_as_float(data.chelsea()[100:250, 50:300])  # import and crop chelsea the cat
sigma = 0.155  # amount of noise
noisy = random_noise(original, var=sigma**2)  # add noise
psnr_noisy = compare_psnr(original, noisy)  # psnr of noisy image

# get (approximate) gradient of image
gray_img = rgb2gray(original)
sobel_edges = sobel(gray_img)

# denoise image with total variation
tv_denoise = denoise_tv_bregman(noisy, 1.)
psnr_tv = compare_psnr(original, tv_denoise)

# denoise image with wavelets and cycle shifts
denoise_kwargs = dict(multichannel=True, convert2ycbcr=True, wavelet='db1')
im_bayescs = cycle_spin(noisy, func=denoise_wavelet, max_shifts=5,
                            func_kw=denoise_kwargs, multichannel=True)
psnr_wv = compare_psnr(original, im_bayescs)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
# plot images
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 7), sharex=False, sharey=False); ax = ax.ravel()
ax[0].imshow(original); ax[0].axis('off'); ax[0].set_title('Original image')
ax[1].imshow(noisy); ax[1].axis('off'); ax[1].set_title('Noisy\nPSNR={:0.4g}'.format(psnr_noisy))
ax[2].imshow(sobel_edges, cmap='gray'); ax[2].axis('off'); ax[2].set_title('Image gradients')
ax[3].imshow(tv_denoise); ax[3].axis('off'); ax[3].set_title('TV denoise: PSNR={:0.4g}'.format(psnr_tv))
ax[4].imshow(im_bayescs); ax[4].axis('off'); ax[4].set_title("Denoised: {0}x{0} shifts PSNR={1:0.4g}".format(5, psnr_wv)); ax[5].axis('off')  # null plot

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Scikit-learn (sklearn)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# * Another library that has great features is sklearn, a library devoted (primarily) to machine learning.
# * Similar to skimage, sklearn comes with the base anaconda environment
# -

# The below example is adapted from, and for more info see, the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor).

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ##### Gaussian process regression

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Let's say you have some data points, and you want to be able to fit a function to them. There's lots of options. For example:
#   * Polynomial
#   * Piece-wise cubic spline
#   * Sinc
#
# For all these options, you could make a model and then minimize the error between the model and data points to "fit" the model. But, what if you want to be able to ascribe some uncertainty to that fit? A good way to do this is with a "Gaussian process."

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Import libraries

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Simulate data

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
def f(x):
    """The unknown function underlying the data."""
    return x * np.sin(x)

# Generate sample points
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Simulate Observations
y = f(X).ravel()

# use finer sampling for evaluating the btrue function
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Plot the data

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
plt.figure()
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Fit the data using Gaussian process (look how easy this is!)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction (give me my fit values)
y_pred, sigma = gp.predict(x, return_std=True)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Plot results

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]), np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.ylim(-10, 20)
plt.xlabel('$x$'); plt.ylabel('$f(x)$'); plt.legend(loc='upper left')

# + {"hideCode": false, "hidePrompt": false}
x = np.arange(5, 11)

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ## How to speed up code

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# The following sections will all try to solve the Poisson equation, and the timings between the different methods will be compared
# $$\nabla^2 \phi(x,y) = \rho(x,y)$$
# One of the easiest and most straightforward methods to solve the equation for $\phi$, numerically, is to use a central finite difference approximation i.e.
# $$\phi_{i-1, j} + \phi_{i, j-1} + \phi_{i+1, j} + \phi_{i, j+1} - 4\phi_{i,j} = h^2 \rho_{i,j}$$
# which is to say, take a point and approximate the derivative at the point as the difference of the points near it.

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# For this example, the boundaries will be defined (using dirichlet boundary conditions). To make it simple, the boundaries will just be constants (could be functions), make
# $$\begin{array}
# \phi(0, y) = 0 \\
# \phi(x, 0) = 0 \\
# \phi(1, y) = 100 \\
# \phi(x, 1) = 100 \\
# \rho(.5, .5) = -100/h^2
# \end{array}$$

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ### Write like Python for easy gains (and let other programs work it out)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# With Cython and Numba you can get performance gains with hardly any extra effort.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# numpy and n_iter will be used for all code in this section
import numpy as np
n_iter = 100  # set number of finite element iterations for all code below


# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Start with simple numpy

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Code up the finite difference approximation. Note: there's lots of better ways to implement the finite difference method than what's coded here (e.g. use fft or wavelet), but let's do the simple thing because this is about timings.

# + {"pixiedust": {"displayParams": {}}, "slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
def phi_iteration(rho, phi, h):
    sz = phi.shape[0]
    phi_now = phi.copy()
    # do above equation
    for j in range(1, sz-1):
        for i in range(1, sz-1):
            phi[i, j] = 0.25 * (-h**2 * rho[i,j] + phi_now[i-1, j] + phi_now[i, j-1] 
                                + phi_now[i+1, j] + phi_now[i, j+1])
    return phi


# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Now write a function that initializes the boundary values and the source term $\rho$, and then runs the finite difference iterations.

# + {"pixiedust": {"displayParams": {}}, "slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
def PDE_solve(niter):
    # define boundary values
    lower_bndy = 0; upper_bndy = 100
    sz = 101; h = 1. / (sz - 1)
    # make grid
    phi = np.zeros((sz, sz))
    # apply boundary values
    phi[-1, :] = upper_bndy; phi[:, -1] = upper_bndy
    phi[0, :] = lower_bndy; phi[:, 0] = lower_bndy
    # apply source term
    rho = np.zeros((sz, sz))
    rho[sz//2, sz//2] = -100/h**2
    # iterate PDE steps to get solution
    for _ in range(niter):
        phi = phi_iteration(rho, phi, h)
    return phi


# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Run timing on the function

# + {"pixiedust": {"displayParams": {}}, "slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %time phi_soln = PDE_solve(n_iter)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(phi_soln, origin='lower')

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# What is the bottleneck?

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %prun PDE_solve(n_iter)

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Cython

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
#  * "The Cython language is a superset of the Python language that additionally supports calling C functions and declaring C types on variables and class attributes."
#  * If you want to wrap C++, most of the time you should use Cython
#  * Extensive support for C++ classes (more elaborate classes and multi-inheritance starts breaking down)
#  * Solutions in Cython range from very Python-like (seen below) code to very C++-like code (consult docs)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# load Cython into notebook
# %load_ext Cython

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Cython can give some big gains in speed just by declaring variable types

# + {"code_folding": [], "slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %%cython
# declare numpy since it's implicitly wrapped in Cython
import numpy as np

# code very similar to before, but now variables are specifically typed
def phi_iteration_cython(double[:, ::1] rho, # type inputs
                          double[:, ::1] phi, 
                          float h):
    cdef int sz = phi.shape[0]  # type as int
    cdef int j, i  # type loop variables as int
    cdef double[:, ::1] phi_now = phi.copy()

    # exact same loop as before
    for j in range(1, sz-1):
        for i in range(1, sz-1):
            phi[i, j] = 0.25 * (-h**2 * rho[i,j] + phi_now[i-1, j] + phi_now[i, j-1] 
                                + phi_now[i+1, j] + phi_now[i, j+1])
    return phi

# solver is also similar, but now the numpy arrays are typed
def PDE_solve_cython(niter):
    # define boundary values
    lower_bndy = 0; upper_bndy = 100
    sz = 101; h = 1. / (sz - 1)
    # make grid
    phi = np.zeros((sz, sz), dtype=np.float64)  # this is now typed in np
    # apply boundary values
    phi[0, :] = lower_bndy; phi[:, 0] = lower_bndy
    phi[-1, :] = upper_bndy; phi[:, -1] = upper_bndy
    # apply source term
    rho = np.zeros((sz, sz), dtype=np.float64)  # this is now typed
    rho[sz//2, sz//2] = -100/h**2
    # iterate PDE steps to get solution
    for _ in range(niter):
        phi = phi_iteration_cython(rho, phi, h)
    return phi


# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Now run the timings and see if things look better

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %timeit -r 2 -n 100 phi_soln = PDE_solve_cython(n_iter)

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Now tell cython that I know what I'm doing and I promise not to go out of bounds and make segmentation faults.

# + {"slideshow": {"slide_type": "skip"}, "code_folding": [], "hideCode": false, "hidePrompt": false}
# %%cython
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def phi_iteration_cython2(double[:, ::1] rho, 
                          double[:, ::1] phi, 
                          float h):
    cdef int sz = phi.shape[0]  # type as int
    cdef int j, i  # type as int
    cdef double[:, ::1] phi_now = phi.copy()
    
    # same loop as before
    for j in range(1, sz-1):
        for i in range(1, sz-1):
            phi[i, j] = 0.25 * (-h**2 * rho[i,j] + phi_now[i-1, j] + phi_now[i, j-1] 
                                + phi_now[i+1, j] + phi_now[i, j+1])
    return phi

def PDE_solve_cython2(niter):
    # define boundary values
    lower_bndy = 0; upper_bndy = 100
    sz = 101; h = 1. / (sz - 1)
    # make grid
    phi = np.zeros((sz, sz), dtype=np.float)  # this is now typed
    # apply boundary values
    phi[0, :] = lower_bndy; phi[:, 0] = lower_bndy
    phi[-1, :] = upper_bndy; phi[:, -1] = upper_bndy
    # apply source term
    rho = np.zeros((sz, sz), dtype=np.float)  # this is now typed
    rho[sz//2, sz//2] = -100/h**2
    # iterate PDE steps to get solution
    for _ in range(niter):
        phi = phi_iteration_cython2(rho, phi, h)
    return phi


# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
# %timeit -r 2 phi_soln = PDE_solve_cython2(n_iter)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Cool.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ![Brent Rambo Approves](https://i.imgflip.com/ljj79.jpg)

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Numba

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Numba is JIT (just in time) compiling. Unfortunately, it really only works on simple things (scalars or arrays &mdash; don't be trying to numba your loaded up neural network graph/class), because it needs to be able to interpret the Python generated bytecode and turn it into machine code. 

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ![numba_flow](https://cdn-images-1.medium.com/max/800/1*9n6WpEXjuD2lBSlX2_pU0g.png)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Numba comes with anaconda, so there shouldn't be anything to install, just import and go.

# + {"code_folding": [], "slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
from numba import jit
phi_iteration_numba = jit(nopython=True)(phi_iteration)  # note this is the exact same function as the original slow one

# this is same function as before, but with phi_iteration_numba
def PDE_solve_numba(niter):
    # define boundary values
    lower_bndy = 0; upper_bndy = 100
    sz = 101; h = 1. / (sz - 1)
    # make grid
    phi = np.zeros((sz, sz))
    # apply boundary values
    phi[0, :] = lower_bndy; phi[:, 0] = lower_bndy
    phi[-1, :] = upper_bndy; phi[:, -1] = upper_bndy
    # apply source term
    rho = np.zeros((sz, sz))
    rho[sz//2, sz//2] = -100/h**2
    # iterate PDE steps to get solution
    for _ in range(niter):
        phi = phi_iteration_numba(rho, phi, h)
    return phi


# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# That was easy! Now, see how the numba implementation does in timing.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %timeit -r 2 -n 500 numba_soln = PDE_solve_numba(n_iter)

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "hideOutput": false, "cell_type": "markdown"}
# Can also help the type inference aspect of the just in time compiler

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
from numba import float64 as nb_float64

# eager interpretation, I'm giving it more information about the function for it to work with in interpreting
phi_iteration_numba2 = jit(nb_float64[:, :](nb_float64[:, :], nb_float64[:, :], nb_float64), 
                          nopython=True)(phi_iteration)

def PDE_solve_numba2(niter):
    # define boundary values
    lower_bndy = 0; upper_bndy = 100
    sz = 101; h = 1. / (sz - 1)
    # make grid
    phi = np.zeros((sz, sz))
    # apply boundary values
    phi[0, :] = lower_bndy; phi[:, 0] = lower_bndy
    phi[-1, :] = upper_bndy; phi[:, -1] = upper_bndy
    # apply source term
    rho = np.zeros((sz, sz))
    rho[sz//2, sz//2] = -100/h**2
    # iterate PDE steps to get solution
    for _ in range(niter):
        phi = phi_iteration_numba2(rho, phi, h)
    return phi


# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
# %timeit numba_soln = PDE_solve_numba2(n_iter)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false}
from IPython.display import Image
Image(url="https://media.giphy.com/media/jy0E1KmYYzm8g/giphy.gif", embed=True)

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Better numpy

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# I wasn't fair to numpy earlier, so what if I made my numpy code better to incorporate its vectorization?

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "code_folding": [12]}
def phi_iteration_np(rho, phi, h):
    sz = phi.shape[0]
    # do above equation
#     for j in range(1, sz-1):
#         for i in range(1, sz-1):
#             phi[i, j] = 0.25 * (-h**2 * rho[i,j] + phi[i-1, j] + phi[i, j-1] 
#                                 + phi[i+1, j] + phi[i, j+1])
    # this is now vectorized
    phi[1:-1, 1:-1] = 0.25 * (-h**2 * rho[1:-1, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, :-2] 
                                + phi[2:, 1:-1] + phi[1:-1, 2:])
    return phi

def PDE_solve_np(niter):
    # define boundary values
    lower_bndy = 0; upper_bndy = 100
    sz = 101; h = 1. / (sz - 1)
    # make grid
    phi = np.zeros((sz, sz))
    # apply boundary values
    phi[0, :] = lower_bndy; phi[:, 0] = lower_bndy
    phi[-1, :] = upper_bndy; phi[:, -1] = upper_bndy
    # apply source term
    rho = np.zeros((sz, sz))
    rho[sz//2, sz//2] = -100/h**2
    # iterate PDE steps to get solution
    for _ in range(niter):
        phi = phi_iteration_np(rho, phi, h)
    return phi


# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %timeit -r 2 phi_soln_np = PDE_solve_np(n_iter)

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ### Interface statically typed languages

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# What about interfacing statically typed languages?
#   * Technically, already did interface C++ with Cython
#   * Let's interface python with the pillars of coding (there's some blazing fast implementations in these languages):
#     * Fortran
#     * C++

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### Fortran

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# If you're coming from MATLAB, [fortran](http://www.fortran90.org/src/best-practices.html) is actually very easy to use. The indexing is 1 based by default (can change indexing to 0 based at time of declaration) and there's built-in broadcasting of arrays. Despite the common misconception, the language is not old and useless -- many libraries are calling fortran code in the background (e.g. anything using LAPACK, which is most things).

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Fortran magic compilation from within the notebook itself (basically a wrapper on another program called f2py). The library should be installed from the appropriate anaconda channel or through pip  
# `conda install -c conda-forge fortran-magic`  
# `pip install fortran-magic`
# -

# !pip install fortran-magic

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %load_ext fortranmagic

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Converting the iteration aspect of the finite difference method is pretty simple. The first few lines just declare the variables and whether they get passed in, out, or in/out. The loops are very straightforward.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %%fortran --opt='-O3'
subroutine phi_iteration_fortran(rho, phi, h, m, n)
    integer, intent(in) :: m, n
    real*8, intent(in) :: rho(m, n), h
    real*8, intent(inout) :: phi(m, n)
    real*8, dimension(m, n) :: phi_now
    integer :: i, j
    phi_now = phi
    ! do above equation
    do i = 2, m - 1
        do j = 2, n - 1
            phi(i, j) = 0.25 * (-h**2 * rho(i, j) + phi_now(i-1, j) + phi_now(i, j-1) &
                                & + phi_now(i+1, j) + phi_now(i, j+1))
        end do
    end do
end subroutine

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Now, similar to numpy broadcasting, can undo the for loops and just do the finite difference with broadcasting (which is built in to fortran)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %%fortran --opt='-O3'
subroutine phi_iteration_broadcast(rho, phi, h, m, n)
    integer, intent(in) :: m, n
    real*8, intent(in) :: rho(m, n), h
    real*8, intent(inout) :: phi(m, n)
    real*8, dimension(m, n) :: phi_now
    phi_now = phi
    ! do above equation
    phi(2:m-1, 2:n-1) = 0.25 * (-h**2 * rho(2:m-1, 2:n-1) + phi(1:m-2, 2:n-1) + phi(2:m-1, 1:n-2) &
                                & + phi(3:m, 2:n-1) + phi(2:m-1, 3:n)) 
end subroutine

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Fortran also has the ability to easily parallelize loops with openmp.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %%fortran --f90flags='-fopenmp' --extra='-lgomp' --opt='-O3'
subroutine phi_iteration_fortran_omp(rho, phi, h, m, n)
    integer, intent(in) :: m, n
    real*8, intent(in) :: rho(m, n), h
    real*8, intent(inout) :: phi(m, n)
    real*8, dimension(m, n) :: phi_now
    integer :: i, j
    phi_now = phi
    !$omp parallel do private(i, j) collapse(2)
    do i = 2, m - 1
        do j = 2, n - 1
            phi(i, j) = 0.25 * (-h**2 * rho(i, j) + phi_now(i-1, j) + phi_now(i, j-1) &
                                & + phi_now(i+1, j) + phi_now(i, j+1))
        end do
    end do
    !$omp end parallel do
end subroutine


# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# As before, make a function that will setup the iterations. Unlike before (working with numba and Cython) where there were nuances in type declarations that forced me to remake the "caller function" accordingly, just make it so I can call whatever function I want.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
def PDE_solve_fortran(niter, func):
    # define boundary values
    lower_bndy = 0; upper_bndy = 100
    sz = 101; h = 1. / (sz - 1)
    # make grid
    phi = np.zeros((sz, sz), order='F', dtype=np.float64)
    # apply boundary values
    phi[0, :] = lower_bndy; phi[:, 0] = lower_bndy
    phi[-1, :] = upper_bndy; phi[:, -1] = upper_bndy
    # apply source term
    rho = np.zeros((sz, sz), order='F', dtype=np.float64)
    rho[sz//2, sz//2] = -100/h**2
    # iterate PDE steps to get solution
    for _ in range(niter):
        func(rho, phi, h, sz, sz)
    return phi


# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Now test the timings. 

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %timeit -r 2 -n 100 phi_soln_f = PDE_solve_fortran(n_iter, phi_iteration_fortran)
# %timeit -r 2 -n 100 phi_soln_fb = PDE_solve_fortran(n_iter, phi_iteration_broadcast)
# %timeit -r 2 -n 100 phi_soln_fomp = PDE_solve_fortran(n_iter, phi_iteration_fortran_omp)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Side note, look how easy it is to pass in the functions (after all, they are just objects) compared to MATLAB where you'd have to make handles or some anonymous function that wraps up the calls.

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Just to show that the answer is the same as before, the improvement isn't because the code is wrong.

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import matplotlib.pyplot as plt
phi_soln_f = PDE_solve_fortran(n_iter, phi_iteration_fortran)
plt.figure()
plt.imshow(phi_soln_f, origin='lower')

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# #### C++

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Most major python libraries have some level of C++ code in the background (e.g. numpy, scipy, and even Python itself &mdash; not to mention the bytecode interpreter). As such, the interface between Python and C++ has lots of capabilities &mdash; but also lots of information to get lost in. Since this is about algorithm development in Python, not trying to build up Django from scratch, I'm just going to go into interfacing with numpy.  
#   * [Here's](https://docs.scipy.org/doc/numpy-1.16.1/reference/c-api.html) a more in-depth look at all the numpy C-API functionalities.  
#   * The full-on Python information for its C-API can be found [here](https://docs.python.org/3/c-api/index.html).

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# The basics:  
#   * Use a C-structure (usually named Py{Name}Object) that is binary- compatible with the PyObject structure itself but holds the additional information needed for that particular object 
#   * Pointers to PyTypeObject can safely be cast to PyObject pointers, whereas the inverse is safe only if the object is known to be an array. 

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# In this case, we're interested in the [PyArrayObject](https://docs.scipy.org/doc/numpy-1.16.1/reference/c-api.types-and-structures.html)
# ```C++
# typedef struct PyArrayObject 
# {
#     PyObject_HEAD  // formality
#     char *data;  // the data bytes
#     int nd;  // array dimensionality
#     npy_intp *dimensions;  // the shape of the dimensions
#     npy_intp *strides;  // the byte strides in each dimension
#     PyObject *base;  // manages memory if a "copy" of another array
#     PyArray_Descr *descr;  // struct for memory and data types (endian, bool, int, etc.)
#     int flags;  // Flags indicating how the memory pointed to by data (C-style, F-style, contiguous, etc.)
#     PyObject *weakreflist;
# } PyArrayObject;
# ```

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# Note: all the cpp code below needs more and better error checking, if you develop you should do error checking but they're omitted here in order to not clutter the concepts.

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# So, let's dive in and see how things work. The first piece of code to develop will be reading in a value from python
#
# ```C++
# PyObject*
# pde_solve_cpp(PyObject *self, PyObject *args)
# {
#     // read in n_iter
#     int n_iter;
#     if (!PyArg_ParseTuple(args, "i", &n_iter))
#     {
#         std::cerr << "Bad input parameters. Put in just n_iter." << std::endl;
#         return NULL;
#     }
#     ...
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# This first bit just says we will return a pointer to a PyObject*, which is really just to say that we're going to give Python all the bits that go into making a Python object. The first argument is a dummy argument and not used. The second argument contains all the input. In this case, it will contain the number of finite element iterations.
# ```C++
# PyObject*
# pde_solve_cpp(PyObject *self, PyObject *args)
# {
#     ...
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# The next piece parses the expected values from the passed arguments. In this case, we are expecting an integer, so read the args and put it into the n_iter memory reference. [This](https://docs.python.org/3/c-api/arg.html) goes over what do with more arguments, including keywords.
# ```C++
# PyObject*
# pde_solve_cpp(PyObject *self, PyObject *args)
# {
#     // read in n_iter
#     int n_iter;
#     if (!PyArg_ParseTuple(args, "i", &n_iter))
#     {
#         std::cerr << "Bad input parameters. Put in just n_iter." << std::endl;
#         return NULL;
#     }
# ...
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Now, just like in the Python version, initialize the variables and boundary values.
# ```C++
# PyObject*
# pde_solve_cpp(PyObject *self, PyObject *args)
# {
#     ...
#     // problem size
#     constexpr int sz = 101;
#     const int rows = sz;
#     const int cols = sz;
#     vector<double> phi(rows * cols, 0.0);  // initialize with 0s
#     // fill last row with 100.0 BC
#     std::fill(phi.begin() + (rows - 1)*cols,
#               phi.end(), 100.0);
#     // make last column 100.0 BC
#     for (int i = 0; i < rows; ++i)
#     {
#         phi[i * cols + cols - 1] = 100.0;
#     }
#     double h = 1 / (static_cast<double>(sz) - 1);  // step size
#     // initialize source term and apply BC
#     vector<double> rho(rows * cols, 0.0);
#     rho[rows/2 * cols + cols/2] = -100.0 / (h * h);
#     ...
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Then run the finite element iterations.
# ```C++
# pde_solve_cpp(PyObject *self, PyObject *args)
# {
#     ...
#     for(auto i = 0; i < n_iter; ++i)
#     {
#         phi_iteration_cpp(phi, rho, h, rows, cols);
#     }
#     ...
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Where the finite_element iteration is the same old thing, though using linear indexing instead of 2D array indexing. (In linear indexing arr[i, j] -> arr[i*cols + j], based on the row-major memory order shown earlier.)
# ```C++
# void
# phi_iteration_cpp(vector<double>& phi,
#                   const vector<double>& rho,
#                   const double h,
#                   const int rows,
#                   const int cols)
# {
#     vector<double> phi_now;
#     double finite_elem;
#     phi_now.reserve(phi.size());
#     std::copy(phi.begin(), phi.end(), phi_now.begin());
#
#     // do update equation
#     for(int i=1; i < rows - 1; ++i)
#     {
#         for(int j=1; j < cols - 1; ++j)
#         {
#             finite_elem =
#                    (-h*h * rho[i * cols + j] +
#                     phi_now[(i - 1) * cols + j] +
#                     phi_now[i * cols + (j - 1)] +
#                     phi_now[(i + 1) * cols + j] +
#                     phi_now[i * cols + (j + 1)]);
#             phi[i * cols + j] = 0.25 * finite_elem;
#         }
#     }
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Lastly, in the pde_solve_cpp function, we need to send the calculated phi vector off to numpy. Essentially, this makes an array telling numpy what dimensions to use (the rows and cols), then convert the phi vector into a Python object and return it. Pretty simple. (Will go into vector_to_2Dnparray function next.)
# ```C++
# PyObject*
# pde_solve_cpp(PyObject *self, PyObject *args)
# {
#     ...
#     // point to array for np to make Python object
#     npy_intp dims[2]{rows, cols};
#     PyObject* phi_np = vector_to_2Dnparray(phi, dims, NPY_DOUBLE);
#     return phi_np;
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Converting the vector is actually pretty easy. One thing to mention is that the vector data needs to be copied into the numpy array because C++ will deallocate the vector memory once the vector goes out of scope. If I had used arrays, I wouldn't have to copy, but I would have to tell numpy that it has the job of freeing the memory.
# ```C++
# template<typename T>
# static PyObject* vector_to_2Dnparray(const vector<T>& vec, npy_intp* dims, int type_num)
# {
#     // note assumes row-major order, can either handle column-major at numpy level or with different API calls
#     // not empty
#     if( !vec.empty() ){
#         PyObject* vec_array = PyArray_SimpleNew(2, dims, type_num);
#         T *vec_array_pointer = (T*) PyArray_DATA(vec_array);
#
#         std::copy(vec.begin(), vec.end(), vec_array_pointer);
#         return vec_array;
#
#     // no data at all
#     } else {
#         npy_intp dims[1] = {0};
#         return (PyObject*) PyArray_ZEROS(1, dims, type_num, 0);
#     }
#
# }
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# The main pieces to pay attention to are
# ```C++
#         PyObject* vec_array = PyArray_SimpleNew(2, dims, type_num);
#         T *vec_array_pointer = (T*) PyArray_DATA(vec_array);
# ```
# which uses the numpy API to allocate memory for a 2D array, with the passed in dims (rows, cols) and the defined type_num (double in this case).
# ```C++
#         std::copy(vec.begin(), vec.end(), vec_array_pointer);
#         return vec_array;
# ```
# This snippet then copies all the data in the vector to the numpy array and returns it.

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Lastly there's just some wrapping overhead to tell Python which C++ code to use and how I want to call the C++ code
# ```C++
# static PyMethodDef
# pde_solve_cpp_method[] =
# {
#     {
#         "run_iterations", pde_solve_cpp, METH_VARARGS, "PDE solve with cpp"
#     },
#     {NULL, NULL, 0, NULL}
# };
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Module definition
# ```C++
# static struct PyModuleDef pde_solve_cpp_def =
# {
#     PyModuleDef_HEAD_INIT,
#     "pde_solve_cpp",
#     "A Python extension module that calculates trace in C++ code.",
#     -1,
#     pde_solve_cpp_method
# };
# ```

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Lastly the PyInit function that completes it all.
# ```C++
# PyMODINIT_FUNC PyInit_pde_solve_cpp(void)
# {
#     Py_Initialize();
#     import_array();
#     return PyModule_Create(&pde_solve_cpp_def);
# }
# ```

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Write all the code snippets to a file so that it can be compiled.

# + {"slideshow": {"slide_type": "skip"}, "hideCode": false, "hidePrompt": false}
# %%file poissonPDE.cpp
#include <Python.h>
#include <numpy/arrayobject.h>
#include <algorithm>
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

template<typename T>
static PyObject* vector_to_2Dnparray(const vector<T>& vec, npy_intp* dims, int type_num)
{
    // note assumes row-major order, can either handle column-major at numpy level or with different API calls
    // not empty
    if( !vec.empty() ){
        PyObject* vec_array = PyArray_SimpleNew(2, dims, type_num);
        T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

        std::copy(vec.begin(), vec.end(), vec_array_pointer);
        return vec_array;

    // no data at all
    } else {
        npy_intp dims[1] = {0};
        return (PyObject*) PyArray_ZEROS(1, dims, type_num, 0);
    }

}

void
phi_iteration_cpp(vector<double>& phi,
                  const vector<double>& rho,
                  const double h,
                  const int rows,
                  const int cols)
{
    vector<double> phi_now;
    double finite_elem;
    phi_now.reserve(phi.size());
    std::copy(phi.begin(), phi.end(), phi_now.begin());

    // do update equation
    for(int i=1; i < rows - 1; ++i)
    {
        for(int j=1; j < cols - 1; ++j)
        {
            finite_elem =
                   (-h*h * rho[i * cols + j] +
                    phi_now[(i - 1) * cols + j] +
                    phi_now[i * cols + (j - 1)] +
                    phi_now[(i + 1) * cols + j] +
                    phi_now[i * cols + (j + 1)]);
            phi[i * cols + j] = 0.25 * finite_elem;
        }
    }
}

static PyObject*
pde_solve_cpp(PyObject *self, PyObject *args)
{
    // read in n_iter
    int n_iter;
    if (!PyArg_ParseTuple(args, "i", &n_iter))
    {
        std::cerr << "Bad input parameters. Put in just n_iter." << std::endl;
        return NULL;
    }

    // problem size
    constexpr int sz = 101;
    const int rows = sz;
    const int cols = sz;
    vector<double> phi(rows * cols, 0.0);  // initialize with 0s
    // fill last row with 100.0 BC
    std::fill(phi.begin() + (rows - 1)*cols,
              phi.end(), 100.0);
    // make last column 100.0 BC
    for (int i = 0; i < rows; ++i)
    {
        phi[i * cols + cols - 1] = 100.0;
    }
    double h = 1 / (static_cast<double>(sz) - 1);  // step size
    // initialize source term and apply BC
    vector<double> rho(rows * cols, 0.0);
    rho[rows/2 * cols + cols/2] = -100.0 / (h * h);
    for(auto i = 0; i < n_iter; ++i)
    {
        phi_iteration_cpp(phi, rho, h, rows, cols);
    }

    // point to array for np to make Python object
    npy_intp dims[2]{rows, cols};
    PyObject* phi_np = vector_to_2Dnparray(phi, dims, NPY_DOUBLE);
    return phi_np;
}

static PyMethodDef
pde_solve_cpp_method[] =
{
    {
        "run_iterations", pde_solve_cpp, METH_VARARGS, "PDE solve with cpp"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pde_solve_cpp_def =
{
    PyModuleDef_HEAD_INIT,
    "pde_solve_cpp",
    "A Python extension module that calculates trace in C++ code.",
    -1,
    pde_solve_cpp_method
};

PyMODINIT_FUNC PyInit_pde_solve_cpp(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&pde_solve_cpp_def);
}

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Easiest way to compile into a useable shared object is to make a setup script

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# %%file setup_cpp.py
from distutils.core import setup, Extension

def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.misc_util import get_info

    config = Configuration('',
                           parent_package,
                           top_path)
    config.add_extension('pde_solve_cpp',
                         sources=['poissonPDE.cpp'],
                         extra_compile_args=['-std=c++11'])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)


# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Then compile the code

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# !python setup_cpp.py build_ext --inplace

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Woohoo, finally at the point where the cpp function can be imported!

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import pde_solve_cpp
import numpy as np

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
# ...and time it!
# %timeit -r 2 -n 500 pde_solve_cpp.run_iterations(n_iter)

# + {"slideshow": {"slide_type": "subslide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Plot to make sure everything looks good

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
phi_soln_cpp = pde_solve_cpp.run_iterations(n_iter)

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false}
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(phi_soln_cpp, origin='lower')

# + {"slideshow": {"slide_type": "slide"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ## Conclusion

# + {"slideshow": {"slide_type": "fragment"}, "hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# Hopefully this has been able to convince you to take the plunge into Python. Things are really pretty simple (other than wrapping C++ with CPython and Numpy API). If you have questions, please feel free to contact me.
