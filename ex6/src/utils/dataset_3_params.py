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

  best = 0;
  for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
      # model = svm_train(X, y, C, gaussian_kernel, 1e-3, 5, sigma);
      # predictions = np.array([svm_predict(model, Xval, sigma)]).T
      print(f"Trying SVM classification with: C = {C}, sigma = {sigma}", end="")
      svm = SVC(C=C, kernel=lambda x1, x2: gaussian_kernel(x1, x2, sigma), tol=1e-3, max_iter=10000)
      model = svm.fit(X, y.ravel())
      predictions = np.array([model.predict(Xval)]).T

      acc = np.mean(predictions == yval);
      print(f" Accuracy = {acc}\n")
      if acc > best:
        best = acc;
        C_best = C;
        sigma_best = sigma;

  C = C_best;
  sigma = sigma_best;

  # =========================================================================

  return (C, sigma)