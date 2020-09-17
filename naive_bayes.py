#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:22:50 2019

@author: andre
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity


class NaiveBayes:
    """
        Class responsible for implementing all the Naive Bayes classifier logic
    """
    

    def __init__(self, bw, feats):
        self.bw = bw
        self.feats = feats
        
    def get_prior(self, class_0, class_1, length_x):
        """Calculates the prior probability for each class"""
        
        #Obtain the prior probability of class 1
        prior_1 = np.log(len(class_1) / length_x)
        #Obtain the prior probability of class 0
        prior_0 = np.log(len(class_0) / length_x)
        
        return (prior_0, prior_1)

    def get_kdes(self, class_0, class_1):
        """Creates a list filled with KDE fitted for each feature"""

        #List that will contain all the different KDE, one for each feature, for all classes
        kde_list = []
        
        #Iterate through the features
        for feat in range (0, self.feats):
            feature_class1 = class_1[:, [feat]] #Get a specific feature of the set of class 1
            feature_class0 = class_0[:, [feat]] #Get a specific feature of the set of class 0

            kde_class1 = KernelDensity(kernel = "gaussian", bandwidth = self.bw)
            kde_class1.fit(feature_class1)  #Fit the kde of class 1 for feature "feat"

            kde_class0 = KernelDensity(kernel = "gaussian", bandwidth = self.bw)
            kde_class0.fit(feature_class0) #Fit the kde of class 0 for feature "feat"
            
            #In kde_list we store the KDE for feature "feat" for class 0 and for class 1
            kde_list.append((kde_class0, kde_class1))
        
        return kde_list
    
    def calculate_scores(self, Xs, kde_list):
        """Calculates the scores and returns them for each class and feature"""
        
        # Array of log_probs/score for every feature, per class
        probs_class_0 = []
        probs_class_1 = []
        
        #For feature_tuple in kde_list:
        for feature in range(self.feats):
            
            feature_column = Xs[:, [feature]]
            
            # Class 0 score
            probs_class_0.append(kde_list[feature][0].score_samples(feature_column))
            # Class 1 score
            probs_class_1.append(kde_list[feature][1].score_samples(feature_column))
        
        # Will return arrays with another arrays inside, every array inside of the wrapping array represents a feature
        return probs_class_0, probs_class_1

    def fit(self, Xs, Ys):
        """Trains the models and gets the errors for training data"""
        
        #Obtain every row that has the last column = 1
        class_1 = Xs[ Ys[:] == 1, : ]
        #Obtain every row that has the last column = 0
        class_0 = Xs[ Ys[:] == 0, : ]
        
        
        # Get the prior probability of each class 
        prior_class0, prior_class1 = self.get_prior(class_0, class_1, len(Xs))
        
        #Get the list with fitted KDE's for every feature
        kde_list = self.get_kdes(class_0, class_1)
        
        #Get the scores for each class in the training data and validation data
        probs_class_0, probs_class_1 = self.calculate_scores(Xs, kde_list)
        
        #Get the sum for every class, for training and validation
        summed_feat_class0 =  prior_class0 + np.sum(probs_class_0, axis = 0)
        summed_feat_class1 =  prior_class1 + np.sum(probs_class_1, axis = 0)
        
        #Array with per observation predicted class
        predicted_classes = (summed_feat_class1 >= summed_feat_class0).astype(int)
        
        #Get the errors by calculating the opposite probability of the scores
        train_error = 1 - accuracy_score(Ys, predicted_classes)
        
        return (train_error, kde_list, prior_class0, prior_class1)
        
    def predict(self, Xs_test, Ys_test, kde_list, prior_class0, prior_class1):
        """Gets the error and predicted classes for a test7validation case"""
        
        #Get the scores for each class in validation data
        probs_class_0_test, probs_class_1_test = self.calculate_scores(Xs_test, kde_list)
        
        summed_feat_class0_test =  prior_class0 + np.sum(probs_class_0_test, axis = 0)
        summed_feat_class1_test =  prior_class1 + np.sum(probs_class_1_test, axis = 0)
        
        # Array with per observation predicted class
        predicted_classes_test = (summed_feat_class1_test >= summed_feat_class0_test).astype(int)
        
        test_error = 1 - accuracy_score(Ys_test, predicted_classes_test)
    
        return test_error,predicted_classes_test






