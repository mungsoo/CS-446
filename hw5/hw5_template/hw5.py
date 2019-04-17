#!/bin/python
#Version 2.1

import numpy as np
import torch
from hw5_utils import *
from scipy.stats import norm
import matplotlib.pyplot as plt

################################# Problem 2 #################################
def k_means(X, k, max_iters = 1000000):
    """
    Implements Lloyd's algorithm.

    arguments:
    X -- n by d data matrix
    k -- integer, number of centers

    return:
    A matrix C of shape k by d with centers as its rows.
    """
    #Hint: You can use np.random.randn to initialize the centers randomly.
    #Hint: Implement auxiliary functions for recentering and for reassigning. Then repeat until no change.
    n, d = X.shape[0], X.shape[1]
    
    rand_center = np.arange(n)
    np.random.shuffle(rand_center)
    rand_center = rand_center[:k]
    C = X[rand_center]
    count = 0
    
    def recenter(X, A):
        
        C = np.empty((k, d))
        for i in range(k):
            if (A[:, i] == False).all():
                C[i] = X[np.random.randint(0, n-1)]
            else:
                C[i] = X[A[:, i] == True].mean(0)
        
        return C
        
    def reassign(X, C):
        
        A = np.zeros((n, k), dtype=np.bool)
        for i in range(n):
            dis = ((X[i] - C)**2).sum(1)
            A[i][np.argmin(dis)] = 1
        
        return A
    
    A = reassign(X, C)
    prev_A = np.zeros_like(A)
    
    while (A != prev_A).any():
        prev_A = A
        C = recenter(X, prev_A)
        A = reassign(X, C)
        count += 1
        
        if count >= max_iters:
            print("Too many iterations, check your code again!")
            break

    return C

def get_purity_score(X, Y, C):
    """
    Computes the purity score for each cluster.

    arguments:
    X -- n by d data matrix
    Y -- n by 1 label vector
    C -- k by d center matrix

    return:
    Fraction of points with label matching their cluster's majority label.
    """
    n = X.shape[0]
    k = C.shape[0]
    matches = 0.0
    
    clusters = [[] for i in range(k)]
    for i in range(n):
        dis = ((X[i] - C)**2).sum(1)
        clusters[np.argmin(dis)].append(i)

    for i in range(k):
        cluster_label = list(Y[clusters[i]])
        if cluster_label:
            
            majority = max(cluster_label, key=cluster_label.count)
            matches += cluster_label.count(majority)
        
    return matches / n
    

def classify_using_k_means(X, Y, k, l=1):
    """
    Classifies the datapoints learning features by k-means and classifying by logistic regression.

    arguments:
    X -- n by d data matrix
    Y -- n by 1 label vector
    k -- integer; number of components
    l -- integer; number of centers to take into account

    return:
    lr -- a logistic classifier
    C -- k by d matrix of centers
    
    assertions:
    l <= k
    """
    assert l <= k
    n = X.shape[0]
    C = k_means(X, k)
    def reassign(X, C):
    
        A = np.zeros((n, k), dtype=np.bool)
        for i in range(n):
            dis = ((X[i] - C)**2).sum(1)
            A[i][np.argsort(dis)[:l]] = 1
      
        return A
        
    A = reassign(X, C)
    lr = logistic_regression(A, Y) 
    return lr,  C

# Part c
X_train, X_test, Y_train, Y_test = load_iris_data()
scores = []
for k in range(2, len(X_train)):
    C = k_means(X_train, k)
    score = get_purity_score(X_train, Y_train, C)
    scores.append(score)
print(scores)
line_plot(scores, output_file="2c")

# Part f
# X_train, X_test, Y_train, Y_test = load_iris_data(0.8)
# def loss(Y, y):
    # return 1-list(Y == y).count(True) / len(Y)
# def reassign(X, C, l):
    # n = X.shape[0]
    # k = C.shape[0]
    # A = np.zeros((n, k), dtype=np.bool)
    # for i in range(n):
        # dis = ((X[i] - C)**2).sum(1)
        # A[i][np.argsort(dis)[:l]] = 1
      
    # return A

    

# train_loss = []
# test_loss = []
# for k in range(2, 21):
    # lr, C = classify_using_k_means(X_train, Y_train, k, l=1)
    
    # y_train = lr.predict(reassign(X_train, C, 1))
    # train_loss.append(loss(Y_train, y_train))
    # y_test = lr.predict(reassign(X_test, C, 1))
    # test_loss.append(loss(Y_test, y_test))
# print(train_loss)
# print(len(train_loss))
# print(test_loss)
# print(len(test_loss))  
# line_plot(train_loss, min_k=2, output_file="l1_train_l.pdf")
# line_plot(test_loss, min_k=2, output_file="l1_test_l.pdf")

# train_loss = []
# test_loss = []
# for k in range(3, 21):
    # lr, C = classify_using_k_means(X_train, Y_train, k, l=3)
    
    # y_train = lr.predict(reassign(X_train, C, 3))
    # train_loss.append(loss(Y_train, y_train))
    # y_test = lr.predict(reassign(X_test, C, 3))
    # test_loss.append(loss(Y_test, y_test))
# print(train_loss)
# print(len(train_loss))
# print(test_loss)
# print(len(test_loss))  
# line_plot(train_loss, min_k=3, output_file="l3_train_l.pdf")
# line_plot(test_loss, min_k=3, output_file="l3_test_l.pdf")

# Part g
# def reassign(X, C):
    # n = X.shape[0]
    # k = C.shape[0]
    # A = np.zeros((n, k), dtype=np.bool)
    # for i in range(n):
        # dis = ((X[i] - C)**2).sum(1)
        # A[i][np.argmin(dis)] = 1
      
    # return A
    
# X = np.empty((4, 50, 2))
# X[0] = np.sqrt(5) * np.random.randn(50, 2) + np.array([5, 5])
# X[1] = np.sqrt(5) * np.random.randn(50, 2) + np.array([5, -5])
# X[2] = np.sqrt(5) * np.random.randn(50, 2) + np.array([-5, 5])
# X[3] = np.sqrt(5) * np.random.randn(50, 2) + np.array([-5, -5])
# X = X.reshape((-1, 2))
# # plt.figure()
# # plt.scatter(X[0][:,0], X[0][:,1])
# # plt.scatter(X[1][:,0], X[1][:,1])
# # plt.scatter(X[2][:,0], X[2][:,1])
# # plt.scatter(X[3][:,0], X[3][:,1])
# # plt.show()
#
# C = k_means(X, 4)
# A = reassign(X, C)
# scatter_plot_2d_project(X[A[:,0],:], X[A[:,1],:], X[A[:,2],:], X[A[:,3],:], C)


################################# Problem 3 #################################
def gmm(X, k, epsilon=0.0000001):
    """
    Computes the maximum likelihood Gaussian mixture model using expectation maximization algorithm.

    argument:
    X -- n by d data matrix
    k -- integer; number of Gaussian components
    epsilon -- improvement lower bound

    return:
    mu -- k by d matrix with centers as rows
    variances -- k by d matrix of variances
    weights -- k by 1 vector of probabilities over the Gaussian components
    """
    
    n = X.shape[0]
    d = X.shape[1]
    
    weights = np.full(k, 1/k)
    variances = np.full((k, d), 1, dtype=np.float64)
    mu = k_means(X, k)
    
    def E(weights, mu, variances):
        R = np.empty((n, k))
        for i in range(n):
            for j in range(k):
                R[i][j] = np.linalg.det(np.diag(variances[j])) ** (-0.5) *\
                (2 * np.pi) ** (-k * 0.5) * \
                weights[j] * np.exp(-1/2 * (X[i] - mu[j]) @ \
                np.linalg.inv(np.diag(variances[j])) @ (X[i] - mu[j])).T
    
        for i in range(n):
            sum = R[i, :].sum()
            R[i] /= sum
        return R
    
    def M(R, weights, mu):
        n_variances = np.empty_like(variances)
        n_weights = np.empty_like(weights)
        n_mu = np.empty_like(mu)
        
        for i in range(k):
            n_variances[i] = (R[:, i].reshape((-1, 1)) * (X - mu[i])**2).sum(0) / (n * weights[i])
            n_mu[i] = (R[:, i].reshape((-1, 1)) * X).sum(0) / (n * weights[i])
            n_weights[i] = R[:, i].sum() / n
        
        n_variances[n_variances < epsilon] = epsilon
        n_weights[n_weights < epsilon] = epsilon
        return n_weights, n_mu, n_variances
        
    prev_weights = np.zeros_like(weights)
    prev_variances = np.zeros_like(variances)
    prev_mu = np.zeros_like(mu)
    while abs((prev_weights - weights).sum()) > 0.000001 or \
          abs((prev_variances - variances).sum()) > 0.000001 or\
          abs((prev_mu - mu).sum()) > 0.000001:
        #print(count)
        #count += 1
        # print(variances)
        R = E(weights, mu, variances)
        
        prev_weights = weights.copy()
        prev_variances = variances.copy()
        prev_mu = mu.copy()
        
        weights, mu, variances = M(R, weights, mu)
    
    return mu, variances, weights
        

def gmm_predict(x, mu, variances, weights):
    """
    Computes the posterior probability of x having been generated by each of the k Gaussian components.

    arguments:
    x -- a single data point
    mu -- k by d matrix of centers
    variances -- k by d matrix of variances
    weights -- k by 1 vector of probabilities over the Gaussian components

    return:
    a k-vector that is the probability distribution of x having been generated by each of the Gaussian components.
    """
    k = mu.shape[0]
    R = np.empty(k)
    for j in range(k):
        R[j] = np.linalg.det(np.diag(variances[j])) ** (-0.5) *\
        (2 * np.pi) ** (-k * 0.5) * \
        weights[j] * np.exp(-1/2 * (x - mu[j]) @ \
        np.linalg.inv(np.diag(variances[j])) @ (x - mu[j]).T)
    
    
    return R /R.sum()

def classify_using_gmm(X, Y, k):
    """
    Classifies the datapoints learning features by GMM and classifying by logistic regression.

    arguments:
    X -- n by d data matrix
    Y -- n by 1 label vector
    k -- integer; number of components

    return:
    lr -- a logistic classifier
    mu -- k by d matrix of centers
    variances -- k by d matrix of variances
    weights -- k-vector of component weights
    """
    
    mu, variances, weights = gmm(X, k)
    n = X.shape[0]
    d = X.shape[1]
    
    def E(weights, mu, variances):
        R = np.empty((n, k))
        for i in range(n):
            for j in range(k):
                R[i][j] = np.linalg.det(np.diag(variances[j])) ** (-0.5) *\
                (2 * np.pi) ** (-k * 0.5) * \
                weights[j] * np.exp(-1/2 * (X[i] - mu[j]) @ \
                np.linalg.inv(np.diag(variances[j])) @ (X[i] - mu[j])).T
    
        for i in range(n):
            sum = R[i, :].sum()
            R[i] /= sum
        return R
    R = E(weights, mu, variances)
    lr = logistic_regression(A, Y) 
    return lr, mu, variances, weights

#####################################################################
# Part b   

# X_train, X_test, Y_train, Y_test = load_iris_data()

def log_likelihood(X, mu, variances, weights):
    n = X.shape[0]
    k = mu.shape[0]
    
    R = np.empty((n, k))
    for i in range(n):
        for j in range(k):
            R[i][j] = np.linalg.det(np.diag(variances[j])) ** (-0.5) *\
            (2 * np.pi) ** (-k * 0.5) * \
            weights[j] * np.exp(-1/2 * (X[i] - mu[j]).T @ \
            np.linalg.inv(np.diag(variances[j])) @ (X[i] - mu[j]))
    
    R = R.sum(1)
    ll = R.prod()
    
    return ll
    
# log_l = []    
# for k in range(2, 11):
    # mu, variances, weights = gmm(X_train, k)
    # print(k)
    # # print(mu)
    # # print(variances)
    # # print(weights)
    # ll = log_likelihood(X_train, mu, variances, weights)
    # log_l.append(ll)
# print(log_l)
# line_plot(log_l, output_file="3b")    

#####################################################################
#Part e

# def E(weights, mu, variances):
    # R = np.empty((n, k))
    # for i in range(n):
        # for j in range(k):
            # R[i][j] = np.linalg.det(np.diag(variances[j])) ** (-0.5) *\
            # (2 * np.pi) ** (-k * 0.5) * \
            # weights[j] * np.exp(-1/2 * (X[i] - mu[j]) @ \
            # np.linalg.inv(np.diag(variances[j])) @ (X[i] - mu[j])).T
 
    # for i in range(n):
        # sum = R[i, :].sum()
        # R[i] /= sum
    # return R
    
# X = np.empty((4, 50, 2))
# X[0] = np.sqrt(5) * np.random.randn(50, 2) + np.array([5, 5])
# X[1] = np.sqrt(5) * np.random.randn(50, 2) + np.array([5, -5])
# X[2] = np.sqrt(5) * np.random.randn(50, 2) + np.array([-5, 5])
# X[3] = np.sqrt(5) * np.random.randn(50, 2) + np.array([-5, -5])
# X = X.reshape((-1, 2))
# mu, variances, weights = gmm(X, 4)
# n = X.shape[0]
# k = 4
# R = E(weights, mu, variances)
# A = np.zeros_like(R, dtype=np.bool)
# for i in range(n):
    # A[i][np.argmax(R[i])] = True
# gaussian_plot_2d_project(mu, variances, X[A[:,0],:], X[A[:,1],:], X[A[:,2],:], X[A[:,3],:])