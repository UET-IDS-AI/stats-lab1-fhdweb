import numpy as np
import matplotlib.pyplot as plt

# Question 1 Generate & Plot Histograms

# generate normal distribution values

def normal_histogram(n):
    data = np.random.normal(0, 1, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1)")
    plt.show()

    return data

# generate uniform distribution values

def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10)")
    plt.show()

    return data

# generate bernoulli distribution values

def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)

    plt.hist(data, bins=10)
    plt.xlabel("Value (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5)")
    plt.show()

    return data

# Question 2 – Sample Mean & Variance

def sample_mean(data):
    data = np.array(data)
    mean = np.sum(data) / len(data)
    return mean

def sample_variance(data):
    data = np.array(data)
    n = len(data)
    mean = sample_mean(data)
    variance = np.sum((data - mean) ** 2) / (n - 1)
    return variance

# Question 3 – Order Statistics

def order_statistics(data):
    data = np.array(data)
    sorted_data = np.sort(data)

    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    median = np.median(sorted_data)
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)

    return (minimum, maximum, median, q1, q3)

# Question 4 Sample Covariance

def sample_covariance(x, y):
    x = np.array(x)
    y = np.array(y)

    n = len(x)
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    return covariance

# Question 5 Covariance Matrix

def covariance_matrix(x, y):
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    matrix = np.array([[var_x, cov_xy],
                       [cov_xy, var_y]])

    return matrix
