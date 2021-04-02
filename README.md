<!--- # MachineLearning --->
Included is a collection of my machine learning projects as well as my homework assignments for the online Machine Learning course offered by Stanford through coursera programmed in Matlab.  

# Table of Contents 
1. [MachineLearning](#MachineLearning)
2. [Projects](#Projects)
3. [machine-learning-ex](#machine-learning-ex)

## MachineLearning
Python modules that implement various machine learning models such as linear regression, logisitic regression, and neural network.  Adadelta.py is a modified gradient descent type algorithm (arXiv:1212.5701). The machine learning modules also contain examples that reproduce results from the Matlab coursera homework assignments. The relavent data files are included. Supervised.py combines the three mentioned modules. With the exception of neural networks, the regression modules have built in polynomial feature options and feature scaling. The neural network can support any number of hidden layers. All modules have regularization built in.  

## Notebooks 
Implementations of some of my personal machine learning modules in jupyter notebooks. The .py modules are located in the MachineLearning folder but they are directly coded inside the notebooks. 
The data is taken from my coursera homework. 

- <ins>LinearRegression.ipynb</ins>: Implements both linear and polynomial regularized regression. Includes plots of cost function history and a contour plot of training parameter history.
- <ins>LogisticRegression.ipynb</ins>: Implements linear and polynomial regularized regression. Includes plots of cost function history and the decision boundary of the binary classifiers. 
- <ins>NeuralNetwork.ipynb</ins>: Implements a neural network to detect handwritten data. 

## Projects

### bank
Jupyter notebook of a classification problem that loads data from http://archive.ics.uci.edu/ml/datasets/Bank+Marketing and follows https://towardsdatascience.com/data-handling-using-pandas-machine-learning-in-real-life-be76a697418c to predict if the client will subscribe (yes/no) a term deposit. The code uses my implemented logistic regression and neural network codes as well as sklearn. The relevant csv files are also included. 

### ApplianceEnergyPrediction
Jupyter notebook of a regression problem that loads data from
https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction. The data is take from 

  Luis M. Candanedo, Veronique Feldheim, Dominique Deramaix, Data driven prediction models of energy use of appliances in a low-energy house, Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97, ISSN 0378-7788. 
  
The models used were regularized linear regression and Gradient Boosting Machine. The linear regression included cross-validation and up to degrees 3 in polynomial features to achieve $R^2$ values of 0.405 and 0.356  for the training and test set respectively. Gradient Boosting included a 60,20,20 split in training, cross validation and test set. The cross validation was used to estimate the n_estimators argument of GradientBoostingRegression in sklearn with depth = 5.  The final $R^2$ values in the prediction of the energy used in Appliances were 0.986 and 0.509 on the training and test set respectively which are similar to the results achieved in the reference. 

### SQL 

Find objects in the Sloan Digital Sky Survey's Data Release 16 using sciserver's SQL module. Objects returned from the query have 3 classifications (star, galaxy, QSO). I use various machine learning classification methods to predict whether an object is a star, galaxy, or QSO based off of the z,ra,dec,dered_u,dered_g,dered_r,dered_i,dered_z features. I run cross validation to determine the final parameters that give the highest F1 score using Neural Network, Gradient Boosting Classification, and Support Vector Machine. The first part of the notebook is based off a course I was a TA for. 

## machine-learning-ex
Matlab homework assignments for the online certification course through Standford. Each homework assignment is labeled as ex(1-8). Various topics included are
- Linear Regression
- Logistic Regression
- Neural Networks
- Feature Scaling
- Gradient Descent
- K-means clustering
- Principle Componant Analysis (PCA)
More will be added as I complete the course. 

<!---
# Table of Contents
1. [Linear Regression ex1](#Linear-Regression-ex1)
2. [Logistic Regression ex2](#Logistic-Regression-ex2)

## Linear Regression ex1

Objectives

<pre>
 
                                   Part Name |     Score | Feedback
                                   --------- |     ----- | --------
                            Warm-up Exercise |  10 /  10 | Nice work!
           Computing Cost (for One Variable) |  40 /  40 | Nice work!
         Gradient Descent (for One Variable) |  50 /  50 | Nice work!
                       Feature Normalization |   0 /   0 | Nice work!
     Computing Cost (for Multiple Variables) |   0 /   0 | Nice work!
   Gradient Descent (for Multiple Variables) |   0 /   0 | Nice work!
                            Normal Equations |   0 /   0 | Nice work!
                                   --------------------------------
                                             | 100 / 100 | 

</pre>

Sample Plots

<p float="center">
 <img src="https://github.com/vagiedd/MachineLearning-Matlab/blob/main/ex1/A39B530C-8953-450D-9631-6401FF86647B.png" width="50%" height="50%">
 <img src="https://github.com/vagiedd/MachineLearning-Matlab/blob/main/ex1/BD674829-7A17-4EFF-9A70-D3FEA629ABC7.png" width="50%" height="50%">
 
</p>

<p align="center">
 <img src="https://github.com/vagiedd/MachineLearning-Matlab/blob/main/ex1/44F27504-4BD8-4F1D-9BE7-F3B72BEB24C7.png" width="50%" height="50%">
</p>

## Logistic Regression ex2

-->
