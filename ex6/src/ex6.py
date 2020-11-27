import numpy as np
from sklearn.svm import SVC
from .utils import \
  load_data, \
  display_data, \
  svm_train, \
  visualize_boundary_linear, \
  visualize_boundary, \
  linear_kernel, \
  gaussian_kernel, \
  dataset_3_params

def ex6():
  """
  Machine Learning Online Class - Exercise 6 | Support Vector Machines

  Instructions
  ------------

  This file contains code that helps you get started on the
  exercise. You will need to complete the following functions:
  
     gaussian_kernel.py
     dataset_3_params.py
     process_email.py
     email_features.py
  
  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.
  """

  # =============== Part 1: Loading and Visualizing Data ================
  #  We start the exercise by first loading and visualizing the dataset. 
  #  The following code will load the dataset into your environment and plot
  #  the data.

  print("Loading and Visualizing Data ...")

  # Load from ex6data1: 
  # You will have X, y in your environment
  (X, y) = load_data.load_data1()

  # Plot training data
  display_data(X, y)

  # ==================== Part 2: Training Linear SVM ====================
  #  The following code will train a linear SVM on the dataset and plot the
  #  decision boundary learned.

  # Load from ex6data1: 
  # You will have X, y in your environment
  (X, y) = load_data.load_data1()

  print("\nTraining Linear SVM ...")

  # You should try to change the C value below and see how the decision
  # boundary varies (e.g., try C = 1000)
  C = 1

  # model = svm_train(X, y, C, linear_kernel, 1e-3, 20)
  svm = SVC(C=C, kernel="linear", tol=1e-3, max_iter=10000)
  model = svm.fit(X, y.ravel())
  visualize_boundary_linear(X, y, model)

  # =============== Part 3: Implementing Gaussian Kernel ===============
  #  You will now implement the Gaussian kernel to use
  #  with the SVM. You should complete the code in gaussian_kernel.py

  print("\nEvaluating the Gaussian Kernel ...")

  x1 = np.array([1, 2, 1])
  x2 = np.array([0, 4, -1])
  sigma = 2
  sim = gaussian_kernel(x1, x2, sigma)

  print("Gaussian Kernel between ", end="")
  print("x1 = [1, 2, 1], ", end="")
  print("x2 = [0, 4, -1], ", end="")
  print(f"sigma = {sigma}: {sim}")
  print("(for sigma = 2, this value should be about 0.324652)")

  # =============== Part 4: Visualizing Dataset 2 ================
  #  The following code will load the next dataset into your environment and 
  #  plot the data. 

  print("Loading and Visualizing Data ...")

  # Load from ex6data2: 
  # You will have X, y in your environment
  (X, y) = load_data.load_data2()

  # Plot training data
  display_data(X, y)

  # ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
  #  After you have implemented the kernel, we can now use it to train the 
  #  SVM classifier.

  print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

  # Load from ex6data2: 
  # You will have X, y in your environment
  (X, y) = load_data.load_data2()

  # SVM Parameters
  C = 1
  sigma = 0.1

  # We set the tolerance and max_passes lower here so that the code will run
  # faster. However, in practice, you will want to run the training to
  # convergence.
  # model = svm_train(X, y, C, gaussian_kernel, 1e-3, 5, sigma)
  svm = SVC(C=C, kernel=lambda x1, x2: gaussian_kernel(x1, x2, sigma), tol=1e-3, max_iter=10000)
  model = svm.fit(X, y.ravel())
  visualize_boundary(X, y, model, sigma)

  # =============== Part 6: Visualizing Dataset 3 ================
  #  The following code will load the next dataset into your environment and 
  #  plot the data. 

  print("Loading and Visualizing Data ...")

  # Load from ex6data3: 
  # You will have X, y in your environment
  (X, Xval, y, yval) = load_data.load_data3()

  # Plot training data
  display_data(X, y)

  # ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

  #  This is a different dataset that you can use to experiment with. Try
  #  different values of C and sigma here.

  # Load from ex6data3: 
  # You will have X, y in your environment
  (X, Xval, y, yval) = load_data.load_data3()

  # Try different SVM Parameters here
  [C, sigma] = dataset_3_params(X, y, Xval, yval)

  # Train the SVM
  # model = svm_train(X, y, C, gaussian_kernel, 1e-3, 5, sigma)
  svm = SVC(C=C, kernel=lambda x1, x2: gaussian_kernel(x1, x2, sigma), tol=1e-3, max_iter=10000)
  model = svm.fit(X, y.ravel())
  visualize_boundary(X, y, model, sigma)