import math

from numpy import *
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

def compute_error_for_given_points(w0, w1, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w1*x + w0)) ** 2
    return totalError / float(len(points))

def step_gradient(w0_current, w1_current, points, learning_rate):
    w0_gradient = 0
    w1_gradient = 0
    N = float(len(points))
  
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w0_gradient += -(2/N) * (y - (w1_current*x + w0_current)) 
        w1_gradient += -(2/N) * x * (y - (w1_current*x + w0_current))
    
    new_w0 = w0_current - (learning_rate * w0_gradient)
    new_w1 = w1_current - (learning_rate * w1_gradient)
    
    return [new_w0, new_w1]

def gradient_descent_runner(points, starting_w0, starting_w1, learning_rate, num_iterations):
    w0 = starting_w0
    w1 = starting_w1
    for i in range(num_iterations):
        w0, w1 = step_gradient(w0, w1, points, learning_rate)
    
    return [w0, w1]

def run():
    income = 'https://raw.githubusercontent.com/ceciliakemiac/DataScienceMaybeFiles/master/simple_linear_regression/data/income.csv'
    points = genfromtxt(income, delimiter=',')
    learning_rate = 0.0001
    
    #y = w1x + w0
    initial_w0 = 0
    initial_w1 = 0
    num_iterations = 1000
  
    print("Starting gradient descent at w0 = {0}, w1 = {1}, error = {2}".format(initial_w0, initial_w1, compute_error_for_given_points(initial_w0, initial_w1, points)))
    print("Running...")
    
    [w0, w1] = gradient_descent_runner(points, initial_w0, initial_w1, learning_rate, num_iterations)
        
    print("After {0} iterations w0 = {1}, w1 = {2}, error = {3}".format(num_iterations, w0, w1, compute_error_for_given_points(w0, w1, points)))
  
run()
