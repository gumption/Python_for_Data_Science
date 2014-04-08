# Python for Data Science

This short primer on [Python](http://www.python.org/) is designed to provide a rapid "on-ramp" for computer programmers who are already familiar with basic concepts and constructs in other programming languages to learn enough about Python to effectively use open-source and proprietary Python-based machine learning and data science tools.

The primer is spread across a collection of [IPython Notebooks](http://ipython.org/notebook.html), and the easiest way to use the primer is to [install IPython Notebook](http://ipython.org/install.html) on your computer. You can also [install Python](https://www.python.org/downloads/), and manually copy and paste the pieces of sample code into the Python interpreter, as the primer only makes use of the Python standard libraries.

There are three versions of the primer. Two versions contain the entire primer in a single notebook:

* Single IPython Notebook: [Python_for_Data_Science_all.ipynb](Python_for_Data_Science_all.ipynb)
* Single web page (HTML): [Python_for_Data_Science_all.html](Python_for_Data_Science_all.html)

The other version divides the primer into 5 separate notebooks:

* [Introduction](1_Introduction.ipynb)
* [Data Science: Basic Concepts](2_Data_Science_Basic_Concepts.ipynb)
* [Python: Basic Concepts](3_Python_Basic_Concepts.ipynb)
* [Using Python to Build and Use a Simple Decision Tree Classifier](4_Python_Simple_Decision_Tree.ipynb)
* [Next Steps](5_Next_Steps.ipynb)

There are several exercises included in the notebooks. Sample solutions to those exercises can be found in two Python source files:

* [`simple_ml.py`](simple_ml.py): a collection of simple machine learning utility functions
* [`SimpleDecisionTree.py`](SimpleDecisionTree.py): a Python class to encapsulate a simplified version of a popular machine learning model

There are also 2 data files, based on the [mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom) in the UCI Machine Learning Repository, used for coding examples, exploratory data analysis and building and evaluating decision trees in Python:

* [`agaricus-lepiota.data`](agaricus-lepiota.data): a machine-readable list of examples or instances of mushrooms, represented by a comma-separated list of attribute values
* [`agaricus-lepiota.attributes`](agaricus-lepiota.attributes): a machine-readable list of attribute names and possible attribute values and their abbreviations