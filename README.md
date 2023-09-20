# Assessment: Machine Learning and Statistics Portfolio

by Sarah McNelis - G00398343

<br>

## Introduction

This repository contains my MLS Portfolio which consists of jupyter notebooks as part of my assessment for Machine Learning and Statistics (MLS) module for my Higher Diplomena in Computing in Data Analytics.

<br>

## Setup

### Install the following

1. Download and install [anaconda](https://docs.anaconda.com/anaconda/install/index.html).
2. Download and install [cmder](https://cmder.app/) if on windows.

<br>

### Running a jupyter notebook

1. Open [cmder](https://cmder.app/) terminal.
2. Run `jupyter lab` or `jupyter notebook`on the command line.
3. The notebook should automatically launch in your browser. 
4. If not, then you may need to temporarily disable security software as some internet security can interfere with jupyter. 
5. Or you can copy the http link in cmder and paste into the url box of your browser. 
6. Once jupyter is open in your browser select which notebook you want to view. 
7. Once the notebook is open click on `kernal` on the tool bar and then `restart and run all`. 
8. Now the notebook is ready. 

<br>

## What to expect

This repository is broken into two parts - A and B. 


### A) `anomaly-detection.ipynb` 
This is a project based on Anomaly Detection using Keras. 
The task for this project is to re-create the time-series anomaly detection from the offical keras website. This involves re-constructing and explaining the concepts of each function and the process of anomaly detection. 

This notebook is divided into 4 sections:

1. **Loading the data** - This consists of retrieving the training and testing datasets and setting them up.

2. **Preprocessing** - This section looks at setting up the training data and training the model.
 
3. **Neural Network** - In this part of the notebook I look at building the layers for the neural network foundation and then fitting the model. 

4. **Evaluation** - This section is made up of smaller elements. First the history of the training and validation lost must be retrieved. Then the training loss must be predicted in order to establish the threshold. Once this is done, the testing data can be set up and the testing loss for this predicted too. Finally, the anomaly detection is set up to determine if the testing loss is greater than the training loss. If this is true, an anomaly is found. 

<br>

### B) `exercises` folder
This folder is comprised of weekly exercises completed using jupyter notebooks. Each notebook contains tasks which are clearly marked and described within. 

The 3 notebooks are as follows:

1. `01-statistics-exercises.ipnb` - consists of 4 exercises based around scipy's version of Fisher's exact test relating the the Lady Tasting Tea problem. 

2. `02-models-exercises.ipnb` - contains 2 exercises which focuses on fitting a straight line using numpy polyfit, scipy optimize minimize and scipy optimize curve fit methods. 

3. `01-parameters-exercises.ipnb` - has 1 exercise which involves using numpy's polyfit to fit polynomials to two different datasets. 

<br>


## Badges


### NB viewer
You can view my MLS Portfolio on nbviewer by clicking on the following badge:

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/SarahMcN25/machine_statistics_assessment/tree/main/)


### Binder
You can view my MLS Portfolio on mybinder by clicking on the following badge:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SarahMcN25/machine_statistics_assessment/HEAD)

<br>


## Credits

- For both the weekly exercises and the anomaly-detection project I heavely relied on my lecturer's notes. You can access his notebooks [here](https://github.com/ianmcloughlin/2223-S1-machine-learn-stats/tree/main/notebooks). 

- I also sourced concepts and code from the below references list. 


<br>


## References


- https://stackoverflow.com/a/4941932 
- https://github.com/ianmcloughlin/2223-S1-machine-learn-stats/blob/main/notebooks/01-statistics.ipynb 
- https://www.geeksforgeeks.org/python-math-comb-method/
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
- https://towardsdatascience.com/fishers-exact-test-from-scratch-with-python-2b907f29e593
- https://www.statology.org/fishers-exact-test/#:~:text=Fisher%27s%20Exact%20Test%20is%20used,table%20is%20less%20than%205
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
- https://www.delftstack.com/api/scipy/scipy-scipy.stats.norm-method/
- https://en.wikipedia.org/wiki/Welch%27s_t-test
- https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
- https://numpy.org/doc/stable/reference/generated/numpy.arange.html
- https://www.sharpsightlabs.com/blog/numpy-absolute-value/ 
- https://sparkbyexamples.com/numpy/numpy-absolute-value/
- https://blog.finxter.com/python-abs/
- https://github.com/ianmcloughlin/2223-S1-machine-learn-stats/blob/main/notebooks/02-models.ipynb
- https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
- https://en.wikipedia.org/wiki/Polynomial 
- https://www.pythonpool.com/numpy-polyfit/#:~:text=The%20function%20NumPy.,Y%2C%20and%20the%20polynomial%20degree
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- https://realpython.com/python-scipy-cluster-optimize/
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
- https://machinelearningmastery.com/curve-fitting-with-python/#:~:text=Curve%20fitting%20is%20a%20type,examples%20of%20inputs%20to%20outputs.
- https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
- https://github.com/ianmcloughlin/2223-S1-machine-learn-stats/blob/main/notebooks/03-parameters.ipynb
- https://www.pythonpool.com/numpy-polyfit/#:~:text=The%20function%20NumPy.,Y%2C%20and%20the%20polynomial%20degree
- https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
- https://en.wikipedia.org/wiki/Curve_fitting
- https://keras.io/examples/timeseries/timeseries_anomaly_detection/
- https://github.com/ianmcloughlin/2223-S1-machine-learn-stats/blob/main/notebooks/04-learning.ipynb
- https://github.com/ianmcloughlin/2223-S1-machine-learn-stats/blob/main/notebooks/05-evaluation.ipynb
- https://www.tensorflow.org/learn
- https://keras.io/
- https://www.kaggle.com/datasets/boltzmannbrain/nab
- https://github.com/numenta/NAB
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
- https://stackoverflow.com/questions/43227058/why-is-python-pandas-dataframe-rounding-my-values
- https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#frequently-used-options
- https://medium.com/@krzysztofdrelczuk/time-series-anomaly-detection-with-python-example-a92ef262f09a
- https://numpy.org/doc/stable/reference/generated/numpy.stack.html
- https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
- https://www.ibm.com/cloud/learn/neural-networks
- https://www.tutorialspoint.com/what-is-convolution-in-signals-and-systems
- https://keras.io/api/layers/#:~:text=Layers%20are%20the%20basic%20building,variables%20(the%20layer%27s%20weights)
- https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
- https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3
- https://www.baeldung.com/cs/learning-curve-ml


<br>


# End