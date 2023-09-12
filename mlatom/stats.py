#!/usr/bin/python
'''
  !---------------------------------------------------------------------------! 
  ! stats.py: Statistics routines                                             ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 
'''

def mean(x):
    # Calculates average value
    value = 0.0
    N = len(x)
    for i in range(N):
        value += x[i]
    value /= N
    return value

def largest_positive_outlier(y_est, y_ref):
    # Finds largest positive outlier
    value = y_est[0] - y_ref[0]
    index = 0
    N = len(y_est)
    for i in range(N):
        temp = y_est[i] - y_ref[i]
        if temp > value:
            value = temp
            index = i
    return value, y_est[index], y_ref[index], index + 1

def largest_negative_outlier(y_est, y_ref):
    # Finds largest negative outlier
    value = y_est[0] - y_ref[0]
    index = 0
    N = len(y_est)
    for i in range(N):
        temp = y_est[i] - y_ref[i]
        if temp < value:
            value = temp
            index = i
    return value, y_est[index], y_ref[index], index + 1

def mae(y_est, y_ref):
    # Mean absolute error
    value = 0.0
    N = len(y_est)
    for i in range(N):
        value += abs(y_est[i] - y_ref[i])
    value /= N
    return value

def mse(y_est, y_ref):
    # Mean signed error
    value = 0.0
    N = len(y_est)
    for i in range(N):
        value += y_est[i] - y_ref[i]
    value /= N
    return value

def rmse(y_est, y_ref):
    # Root-mean-square error
    import math
    value = 0.0
    N = len(y_est)
    for i in range(N):
        value += (y_est[i] - y_ref[i]) ** 2
    value /= N
    value = math.sqrt(value)
    return value

def correlation_coefficient(x, y):
    # Pearson correlation coefficient
    # Note that squared correlation coefficient
    # is the same as 
    # R-squared in linear least squares regression
    import math
    value = 0.0
    Xmean = mean(x)
    Ymean = mean(y)
    N = len(x)
    nominator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    for i in range(N):
        nominator += (x[i] - Xmean) * (y[i] - Ymean)
        denominator1 += (x[i] - Xmean) ** 2
        denominator2 += (y[i] - Ymean) ** 2
    value = nominator / (math.sqrt(denominator1) * math.sqrt(denominator2))
    return value

def linear_regression(x, y):
    # Finds regression coefficients a, b by least square
    # fitting of y_est = a + b * x to y values
    # Returns also R squared value and
    # standard errors for a and b
    #
    # http://mathworld.wolfram.com/LeastSquaresFitting.html
    #
    import math

    N = len(x)
    Xmean = mean(x)
    Ymean = mean(y)
    
    ss_xx = -N * (Xmean ** 2)
    ss_yy = -N * (Ymean ** 2)
    ss_xy = -N * Xmean * Ymean
    for i in range(N):
        ss_xx += x[i] ** 2
        ss_yy += y[i] ** 2
        ss_xy += x[i] * y[i]

    b = ss_xy / ss_xx
    a = Ymean - b * Xmean
    r_squared = ss_xy ** 2 / (ss_xx * ss_yy)
    s = math.sqrt(max(ss_yy - ( (ss_xy ** 2) / ss_xx), 0) / float(N - 2))
    SE_a = s * math.sqrt(1.0 / float(N) + (Xmean ** 2) / ss_xx)
    SE_b = s / math.sqrt(ss_xx)
    
    return a, b, r_squared, SE_a, SE_b

def calc_median_absolute_deviation(values):
    import numpy as np
    if type(values) == list: values_array = np.array(values)
    else: values_array = values
    M = np.median(values_array)
    MAD = 1.4826 * np.median(np.abs(np.array(values_array) - M))
    return MAD

if __name__=='__main__':
    print ('Test data:')
    y_actual = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_predicted = [1.02, 1.98, 3.5, 4.1, 4.9]
    print ('y_predicted = ', y_predicted)
    print ('Mean of y_predicted = ', mean(y_predicted))
    print ('y_actual = ', y_actual)
    print ('Mean of y_actual = ', mean(y_actual))
    print ('-' * 80)
    print ('Errors:')
    print ('Largest positive outlier', largest_positive_outlier(y_predicted, y_actual))
    print ('Largest negative outlier', largest_negative_outlier(y_predicted, y_actual))
    print ('MAE = ', mae(y_predicted, y_actual))
    print ('MSE = ', mse(y_predicted, y_actual))
    print ('RMSE = ', rmse(y_predicted, y_actual))
    print ('Correlation coefficient = ', correlation_coefficient(y_actual, y_predicted))
    print ('Squared correlation coefficient = ', correlation_coefficient(y_actual, y_predicted) ** 2)
    print ('-' * 80)
    a, b, r_squared, SE_a, SE_b = linear_regression(y_actual, y_predicted)
    print ('Regression line y_est = a + b * x for y_predicted (x) vs y_actual (y)')
    print ('a = ', a)
    print ('b = ', b)
    print ('R^2 = ', r_squared)
    print ('Standard error for a = ', SE_a)
    print ('Standard error for b = ', SE_b)
    
    
    
    
    
    
