# Python for Data Science

This short primer on [Python](http://www.python.org/) is designed to provide a rapid "on-ramp" for computer programmers who are already familiar with basic concepts and constructs in other programming languages to learn enough about Python to effectively use open-source and proprietary Python-based machine learning and data science tools.

The primer is spread across a collection of [IPython Notebooks](http://ipython.org/notebook.html), and the easiest way to use the primer is to [install IPython Notebook](http://ipython.org/install.html) on your computer. You can also [install Python](https://www.python.org/downloads/), and manually copy and paste the pieces of sample code into the Python interpreter, as the primer only makes use of the Python standard libraries.

There are four versions of the primer. Three versions contain the entire primer in a single notebook:

* Single IPython Notebook (cleared output cells): [Python_for_Data_Science_clean.ipynb](Python_for_Data_Science_clean.ipynb)
* Single IPython Notebook (filled output cells): [Python_for_Data_Science_clean.ipynb](Python_for_Data_Science_all.ipynb)
* Single web page (HTML): [Python_for_Data_Science_all.html](Python_for_Data_Science_all.html)

The other version divides the primer into 5 separate notebooks:

* [Introduction](1_Introduction.ipynb)
* [Data Science: Basic Concepts](2_Data_Science_Basic_Concepts.ipynb)
* [Python: Basic Concepts](3_Python_Basic_Concepts.ipynb)
* [Using Python to Build and Use a Simple Decision Tree Classifier](4_Python_Simple_Decision_Tree.ipynb)
* [Next Steps](5_Next_Steps.ipynb)

There are several exercises included in the notebooks. Sample solutions to those exercises can be found in two Python source files:

* [`simple_ml.py`](simple_ml.py): a collection of simple machine learning utility functions
* [`simple_decision_tree.py`](simple_decision_tree.py): a Python class to encapsulate a simplified version of a popular machine learning model

There are also 2 data files, based on the [mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom) in the UCI Machine Learning Repository, used for coding examples, exploratory data analysis and building and evaluating decision trees in Python:

* [`agaricus-lepiota.data`](agaricus-lepiota.data): a machine-readable list of examples or instances of mushrooms, represented by a comma-separated list of attribute values
* [`agaricus-lepiota.attributes`](agaricus-lepiota.attributes): a machine-readable list of attribute names and possible attribute values and their abbreviations

## Change Log

2015-07-21

* Updated 5 subnotebooks for Python 3 compatibility
* Changed file name of `SimpleDecisionTree.py` to `simple_decision_tree.py` (class name is unchanged)
* Reordered introduction of `defaultdict` and `Counter` containers
* Reordered the values returned by `classification_accuracy()`
* Added more examples of formatted printing via `str.format()`
* Various and sundry other minor changes to prepare for [tutorial](http://seattle.pydata.org/schedule/presentation/8/) at [PyData Seattle 2015](http://seattle.pydata.org/)

2015-02-23

* Added attribution for suggested changes to accommodate Python 3 to [Nick Coghlan](https://twitter.com/ncoghlan_dev)

2015-02-22

* Added `from __future__ import print_function, division` for Python 3 compatibility
* Updated `simple_ml.py` and `SimpleDecisionTree.py` to also use Python 3 `print_function` and `division`
* Replaced `xrange()` (Python 2 only) with `range()` (Python 2 or 3)
* Replaced `dict.iteritems()` (Python 2 only) with `dict.items()` (Python 2 or 3)
* Changed ["call by reference"](https://en.wikipedia.org/wiki/Evaluation_strategy#Call_by_reference) to ["call by sharing"](https://en.wikipedia.org/wiki/Evaluation_strategy#Call_by_sharing)
* Added `isinstance()` (and reference to duck typing) to section on `type()`
* Added variable for `delimiter` rather than hard-coding `'|'` character
* Cleaned up various cells