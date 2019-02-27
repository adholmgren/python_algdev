# python_algdev
Based on a talk that I gave at Ball Aerospace. Goes over the basics of Jupyter, skimage/sklearn examples, optimizing code.

The main file is the Jupter notebook SWTP-PythonAlgDev.ipynb. Another file SWTP-PythonAlgDev.py exists, which is the percent formatted
version of the json notebook (see the [Editing the notebook](#editing-the-notebook) section for more details).

Note: certain parts of the optimization section of the notebook will write files to the repo.

# Generating presentation
File > Download as > Reveal.js slides (html)  
or in a terminal run
```
jupyter nbconvert --to slides SWTP-PythonAlgDev.ipynb --reveal-prefix=reveal.js --SlidesExporter.reveal_theme=sky --SlidesExporter.reveal_scroll=True --SlidesExporter.reveal_transition=zoom
```

# Editing the notebook
If you'd like to edit the notebook or contribute, please feel free. I ask that you only edit and submit the .py file using [Jupytext](https://github.com/mwouts/jupytext) to handle the conversion so that any merges are easier to manage.
