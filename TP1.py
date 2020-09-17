#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:33:44 2019

@author: André Bastos & Carolina Goldstein
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot as skplt
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
import math as m

# Fixing seed, None for no seed to be fixed
random_state = None

def standardize(Xs):
    """Transforms data to fit a normal distribution with 0 means and 1 standard deviation"""
    means = np.mean(Xs,axis=0)
    stdevs = np.std(Xs,axis=0)
    return (Xs-means)/stdevs,means,stdevs

def standardize_test(means,stdevs, Xs_test):
    """Transforms data to fit a normal distribution with 0 means and 1 standard deviation for the test set,
    using the mean and std of the training data
    """
    return (Xs_test-means)/stdevs

#---------------------------------------------------------------------------------------------------------#

def cross_validation_nb(Xs_train, Ys_train, feats):
    """Optimizes a parameter for the naive bayes classifier using the cross validation technic"""
    folds = 5
    kf = StratifiedKFold(n_splits=folds)
    #cross_error_list = []
    bws = []
    train_errors = []
    val_errors = []
    
    #For loop iterating every band-width value from 0.02 to 0.6 with a step of 0.02
    for bw in np.arange(0.02, 0.6, 0.02):
        
        summed_train_errors = 0
        summed_val_errors = 0
    

        #Stratified k folds
        for train_idx, valid_idx in kf.split(Ys_train, Ys_train):

            #Obtain the training and validation folds from the training set
            x_train_set = Xs_train[train_idx]
            x_val_set = Xs_train[valid_idx]

            y_train_set = Ys_train[train_idx]
            y_val_set = Ys_train[valid_idx]
        
            # Calculate naive bayes for this specific bandwidth
            nb = NaiveBayes(bw, feats)
            
            train_error, kde_list, prior_class0, prior_class1 = nb.fit(x_train_set, y_train_set)
            
            val_error,pred_val = nb.predict(x_val_set, y_val_set, kde_list, prior_class0, prior_class1)
    
            
            summed_train_errors += train_error
            summed_val_errors += val_error
       
            
        bws.append(bw)
        train_errors.append(summed_train_errors/folds)
        val_errors.append(summed_val_errors/folds)
    
    # Choose best bandwidth
    best_bandwidth = 0
    best_bw_val_error = 100
    for i in range(len(bws)):
        bw = bws[i]
        if val_errors[i] < best_bw_val_error:
            best_bandwidth = bw
            best_bw_val_error = val_errors[i]
    print("Best BW training")
    print(best_bandwidth)
    
    return (best_bandwidth, bws, train_errors, val_errors)

    
def run_naive_bayes(train_data, test_data):
    """Runs the naive bayes classifier"""
    print("Running Naive Bayes")
    feats = 4
    
    # Shuffle the data sets
    train_data = shuffle(train_data,random_state=random_state)
    
    # Divide them between data and labels
    Ys_train = train_data[:,-1]
    Xs_train_pre = train_data[:,:-1]
    
    Ys_test = test_data[:,-1]
    Xs_test = test_data[:,:-1]
    
    # Standardize the data
    Xs_train,means,stdevs = standardize(Xs_train_pre)
    
    Xs_test = standardize_test(means,stdevs, Xs_test)
    
    best_bandwidth, bws, train_errors, val_errors = cross_validation_nb(Xs_train, Ys_train, feats)
    
    fig, ax = plt.subplots()
    
    ax.plot(bws, val_errors, label = "Validation Error", color = "r")
    ax.plot(bws, train_errors, label = "Training Error", color = "b")
    ax.set_title('Naive Bayes Training and Validation Errors per Bandwidth')
    ax.legend()
    fig.savefig("NB.png")
    #quanto menor o bw mais overfitting porque considera menos pontos
    
    # Train the classifier with the best bandwidth
    nb = NaiveBayes(best_bandwidth, feats)
    train_error, kde_list, prior_class0, prior_class1 = nb.fit(Xs_train, Ys_train)
    
    # Predict the classes for the test set
    test_error,predicted_classes_test = nb.predict(Xs_test, Ys_test, kde_list, prior_class0, prior_class1)
    
    print(f"Test error: {test_error}")
    
    return predicted_classes_test,Ys_test
    
#---------------------------------------------------------------------------------------------------------#  

def run_gaussian_nb(train_data, test_data):
    """Runs the gaussian naive bayes classifier"""
    print("Running Gaussian NB")
    
    # Shuffle the data sets
    train_data = shuffle(train_data,random_state=random_state)
    
    # Divide them between data and labels
    Ys_train = train_data[:,-1]
    Xs_train_pre = train_data[:,:-1]
    
    Ys_test = test_data[:,-1]
    Xs_test = test_data[:,:-1]
    
    # Standardize the data
    Xs_train,means,stdevs = standardize(Xs_train_pre)
    
    Xs_test = standardize_test(means,stdevs, Xs_test)
    
    gnb = GaussianNB()
    gnb.fit(Xs_train, Ys_train)
    prediction = gnb.predict(Xs_test)
    test_error = 1 - gnb.score(Xs_test, Ys_test)
    
    skplt.metrics.plot_confusion_matrix(Ys_test, prediction, normalize=True)
    print(f"Test error: {test_error}")
    plt.show()
    
    return prediction,Ys_test
  
#---------------------------------------------------------------------------------------------------------#     

def calc_fold(gamma,X,Y, train_ix, val_ix, c = 1.0):
    """Return classification error for train and validation sets"""
    clf = SVC(gamma=gamma, C = c)
    clf.fit(X[train_ix,:], Y[train_ix])
    error_tr = 1-clf.score(X[train_ix,:],Y[train_ix])
    error_val = 1-clf.score(X[val_ix,:],Y[val_ix])
    return (error_tr,error_val)

def svm(Xs_train, Ys_train, Xs_test, Ys_test):
    """Uses cross validation to optimize the gamma parameter for SVM"""
    folds = 5
    kf = StratifiedKFold(n_splits=folds)
    gammas = []
    training_error=[]
    validation_error=[]
    best_tr_err=10
    best_va_err=10
    best_gamma=0.1
    for gamma in np.arange(0.2,6.0,0.2):
        tr_err = va_err = 0
        for tr_ix,va_ix in kf.split(Xs_train,Ys_train):
            r,v = calc_fold(gamma,Xs_train,Ys_train,tr_ix,va_ix)
            tr_err += r
            va_err += v
        training_error.append(tr_err/folds)
        validation_error.append(va_err/folds)
        gammas.append(gamma)
        if va_err < best_va_err:
            best_tr_err = tr_err
            best_va_err = va_err
            best_gamma = gamma
        #print(feats,':', tr_err/folds,va_err/folds)
    print(f"Best gamma: {best_gamma}")
    
    #training with the best gamma and producing the test error
    clf = SVC(gamma=best_gamma)
    clf.fit(Xs_train, Ys_train) 
    test_error=1-clf.score(Xs_test,Ys_test)
    prediction= clf.predict(Xs_test)
    print(f'Test error: {test_error}')
    
    return best_gamma, gammas, training_error, validation_error,prediction
    
def svm_optimized(Xs_train, Ys_train, Xs_test, Ys_test):
    """Uses cross validation to optimize the gamma and c parameter for SVM"""
    folds = 5
    kf = StratifiedKFold(n_splits=folds)
    gammas = []
    Cs = []
    training_error=[]
    validation_error=[]
    best_c=0.1
    best_gamma=0.1
    c = 0.1
    # Validation error using both parameters
    best_validation_error = 10
    
    while c <= 10000:
        # Validation error for the gamma optimization
        best_tr_err=10
        best_va_err=10
        gammas = []
    
        for gamma in np.arange(0.2,6.0,0.2):
        
            tr_err = va_err = 0
            for tr_ix,va_ix in kf.split(Xs_train,Ys_train):
                r,v = calc_fold(gamma,Xs_train,Ys_train,tr_ix,va_ix, c)
                tr_err += r
                va_err += v
            
            gammas.append(gamma)
            
            if va_err < best_va_err:
                best_tr_err = tr_err/folds
                best_va_err = va_err/folds
                best_gamma = gamma
        
        if best_va_err < best_validation_error:
            best_validation_error = best_va_err
            best_c = c
                
        training_error.append(best_tr_err)
        validation_error.append(best_va_err)
                
        Cs.append(c)
        c = c * 2
        
    print(f"Best Gamma: {best_gamma}, Best C: {best_c}")
    
    #training with the best gamma and producing the test error
    clf = SVC(gamma=best_gamma, C = best_c)
    clf.fit(Xs_train, Ys_train) 
    test_error=1-clf.score(Xs_test,Ys_test)
    prediction= clf.predict(Xs_test)
    print(f'Test error: {test_error}')
    
    return best_c, Cs, training_error, validation_error,prediction
    
def run_SVM(train_data, test_data, isOptimizingC = False):
    """Runs the svm classifier, for both gamma optimization and gamma and c optimization"""
    print("Running SVM")
    
    # Shuffle the data sets
    train_data = shuffle(train_data,random_state=random_state)
    
    # Divide them between data and labels
    Ys_train = train_data[:,-1]
    Xs_train_pre = train_data[:,:-1]
    
    Ys_test = test_data[:,-1]
    Xs_test = test_data[:,:-1]
    
    # Standardize the data
    Xs_train,means,stdevs = standardize(Xs_train_pre)
    
    Xs_test = standardize_test(means,stdevs, Xs_test)
    
    validation_error = []
    training_error = []
    parameters = []

    if isOptimizingC:
        # Optimize gamma and c SVM case
        best_c, Cs, training_error, validation_error, prediction = svm_optimized(Xs_train, Ys_train, Xs_test, Ys_test)
        parameters = Cs
    else:
        # Optimize gamma SVM case
        best_gamma, gammas, training_error, validation_error, prediction = svm(Xs_train, Ys_train, Xs_test, Ys_test)
        parameters = gammas
    
    fig, ax = plt.subplots()
    ax.plot(parameters,validation_error,label = "Validation Error", color = "r")
    ax.plot(parameters,training_error,label='Training error', color="b")
    
    if isOptimizingC:
        parameter = "C"
    else:
        parameter = "Gamma"
    ax.set_title(f'SVM Errors per {parameter}') 
    ax.legend()
    if not isOptimizingC:
        fig.savefig("SVM.png")
    plt.show()
    
    return prediction,Ys_test
 
#---------------------------------------------------------------------------------------------------------#    

#Get data for training and testing
train_data = np.loadtxt('TP1_train.tsv',delimiter='\t')
test_data = np.loadtxt('TP1_test.tsv',delimiter='\t')

#Run Naive Bayes
prediction_nb ,Ys_test = run_naive_bayes(train_data, test_data)

#Run GaussianNB
prediction_gnb ,Ys_test = run_gaussian_nb(train_data, test_data)

#Run SVM
prediction_svm ,Ys_test = run_SVM(train_data, test_data)

#Optional
#Run c, gamma optimization SVM
prediction_svm_c ,Ys_test = run_SVM(train_data, test_data, True)

#Compare classifiers
#Approximate normal test

def app_normal_test(prediction,true_values,classifier): 
    '''Returns a 95% confidence interval for the expected number of errors in the given classifier'''
    w=abs(prediction-true_values.ravel())
    X= np.sum(w)#number of misclassified examples
    N=len(true_values) #total size of the test set
    p0= X/N #probability of misclassification
    sigma=m.sqrt(N*p0*(1-p0))
    interval=[X-1.96*sigma,N*p0 ,X+1.96*sigma]
    print(f'95% confidence interval for {classifier}: {interval}')
    return(interval)

interval_SVM = app_normal_test(prediction_svm,Ys_test,'SVM')
interval_GNB = app_normal_test(prediction_gnb,Ys_test,'GNB')
interval_NB = app_normal_test(prediction_nb,Ys_test,'NB')
interval_SVM_C = app_normal_test(prediction_svm_c,Ys_test,'SVM C')



print('SVM',interval_SVM)
print('GNB',interval_GNB)
print('SVM_C',interval_SVM_C)
print('NB',interval_NB)

def comparison(interval_classifier1,interval_classifier2):
    """Receiving two confidence intervals this function chooses wether the classifiers are significantly different by checking if the intervals do intercept each other"""
    if (interval_classifier1[2]<=interval_classifier2[0] and interval_classifier1[0]<=interval_classifier2[0]) or (interval_classifier1[0]>=interval_classifier2[0] and interval_classifier1[0]>=interval_classifier2[2]):
        print("The classifiers are significantly different")
    else:
        print("The classifiers are NOT significantly different")
        

#Mcnemar
def mcnemar(classifier1,classifier2,Ys_te):
    '''Chooses wether the classifiers are significantly different by executing the mcnemar technic'''
    misclassified_1=abs(classifier1-Ys_te.ravel())
    misclassified_2=abs(classifier2-Ys_te.ravel())
    e01=0
    e10=0
    for i in range(len(Ys_te)):
        if (misclassified_1[i]==1 and misclassified_2[i]==0):
            e01+=1
        elif (misclassified_1[i]==0 and misclassified_2[i]==1):
            e10+=1  
    mcnemar = ((abs(e01-e10)-1)**2)/(e01+e10)
    if mcnemar>3.84:
        conclusion="The two classifiers perform significantly different"
    else:
        conclusion="The two classifiers do not perform significantly different"
    print(f"McNemar's test: {mcnemar}, Conclusion: {conclusion}")

print('Comparisons with Mcnemar')
print('SVM and SVM with optimized C\t')
mcnemar(prediction_svm,prediction_svm_c,Ys_test)
print('GNB and SVM\t')
mcnemar(prediction_gnb,prediction_svm,Ys_test)
print('NB and SVM\t')
mcnemar(prediction_nb,prediction_svm,Ys_test)
print('GNB and NB C\t')
mcnemar(prediction_gnb,prediction_nb,Ys_test)