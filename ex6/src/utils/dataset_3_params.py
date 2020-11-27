import numpy as np
import scipy.optimize as optimize
from sklearn.svm import SVC
from . import svm_train, svm_predict, gaussian_kernel

def dataset_3_params(X, y, Xval, yval):
  """
    dataset_3_params returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel

    def dataset_3_params(X, y, Xval, yval) returns your choice of C and 
    sigma. You should complete this function to return the optimal C and 
    sigma based on a cross-validation set.
  """

  # You need to return the following variables correctly.
  C = 1;
  sigma = 0.3;

  # ====================== YOUR CODE HERE ======================
  # Instructions: Fill in this function to return the optimal C and sigma
  #               learning parameters found using the cross validation set.
  #               You can use svmPredict to predict the labels on the cross
  #               validation set. For example, 
  #                   predictions = svmPredict(model, Xval);
  #               will return the predictions on the cross validation set.
  #
  #  Note: You can compute the prediction error using 
  #        mean(double(predictions ~= yval))



  # =========================================================================

  return (C, sigma)