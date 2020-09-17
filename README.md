# AA_assignment_1

Grade: 20/20

Objective

The goal of this assignment is to parametrize, fit and compare Naive Bayes and Support Vector Machine classifiers. The data set is inspired on the banknote authentication problem in the UCI machine learning repository, but the data was adapted for this assignment. You can download the data files from here:

TP1_train.tsv
    This is the training data set. Use this to optimize parameters and train the classifiers.
TP1_test.tsv
    This is the test set for estimating the true error and comparing the final classifiers.

You must implement your own Naïve Bayes classifier using Kernel Density Estimation for the probability distributions of the feature values. For this, you can use any code from the lectures, lecture notes and tutorials that you find useful. Also, use the KernelDensity class from sklearn.neighbors.kde for the density estimation.

You will need to find the optimum value for the bandwitdh parameter of the kernel density estimators you will use. Use the training set provided in the TP1_train.tsv for this.

The second classifier will be the Gaussian Naïve Bayes classifier in the sklearn.naive_bayes.GaussianNB class. You do not need to adjust parameters for this classifier

Finally, use a Support Vector Machine with a Gaussian radial basis function, available in the sklearn.svm.SVC class. Use a regularization factor C = 1 and optimize the gamma parameter with cross-validation on the training set.

Finally, compare the performance of the three classifiers, identify the best one and discuss if it is significantly better than the others.

The data are available on .tsv files where each line corresponds to a bank note and the five values, separated by commas, are, in order, the four features (variance, skewness and curtosis of Wavelet Transformed image and the entropy of the bank note image) and the class label, an integer with values 0 for real bank notes and 1 for fake bank notes.

In addition to the code, you must include two plots with the training and cross-validation errors for the optimization of the bandwidth parameter of your classifier (this plot should be named NB.png) and the gamma value of the SVM classifier (this plot should be named SVM.png). These plots should have a legend identifying which line corresponds to which error.

Furthermore, you must also answer a set of questions about this assignment. The question files are available here (choose one only). Right-click the links and save the chosen file locally. You must include the file with your answers in the zip file when submitting the assignment.

Optional: fine tune the SVM classifier

The SVM classifier you will use has two adjustable parameters, gamma and C. In this assignment we fixed parameter C at a value of 1 but you can try to improve the performance of this classifier by adjusting simultaneously these two parameters. This is an optional exercise, worth only 1/20 of the assignment grade: optimize both parameters, compare the optimized SVM with the previous classifiers and discuss whether this additional optimization was useful in this case.
