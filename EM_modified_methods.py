import math

import numpy as np
import matplotlib.pyplot as plt


# Generates n data points from a mixture of two Gaussian distributions with weights "weight" and "1-weight"
def generate_mixed_data(mu1, sigma1, mu2, sigma2, weight, n):
    first_component = np.random.normal(mu1, sigma1, int(n * weight))
    second_component = np.random.normal(mu2, sigma2, int(n * (1 - weight)))
    return first_component.tolist() + second_component.tolist()


# Calculates the probability of x, when the probability is given by a gaussian with mu and sigma
def normal(mu, sigma, x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


# Select the parameters given i.
def get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y):
    if i == 1:
        return w1_x, w1_y, mu1, sigma1
    else:
        return w2_x, w2_y, mu2, sigma2


# Update rule for the weight. It also returns the sum of the Py posteriors because they will be useful for the other rules
def calculate_next_weight(i, data, mu1, sigma1, mu2, sigma2, w1, w2):
    # This methods expects weights for the probability x and y but here we are only dealing with one of them, so just
    # duplicate them.
    w_i, w_i, mu_i, sigma_i = get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1, w2, w1, w2)

    sum_py_i = 0
    for x in data:
        py_i = w_i * normal(mu_i, sigma_i, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_py_i += py_i
    wi_next = sum_py_i / len(data)
    return wi_next, sum_py_i


# Update rule for mu.
def calculate_next_mu(i, sum_py_i_x, data_x, sum_py_i_y, data_y, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y):
    w_i_x, w_i_y, mu_i, sigma_i = get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y)

    sum_xpy_i_x = 0
    for x in data_x:
        py_i_x = w_i_x * normal(mu_i, sigma_i, x) / (w1_x * normal(mu1, sigma1, x) + w2_x * normal(mu2, sigma2, x))
        sum_xpy_i_x += x * py_i_x

    sum_xpy_i_y = 0
    for x in data_y:
        py_i_y = w_i_y * normal(mu_i, sigma_i, x) / (w1_y * normal(mu1, sigma1, x) + w2_y * normal(mu2, sigma2, x))
        sum_xpy_i_y += x * py_i_y

    mu_i_next = (sum_xpy_i_x + sum_xpy_i_y) / (sum_py_i_y + sum_py_i_x)
    return mu_i_next


# Update rule for sigma
def calculate_next_sigma(i, sum_py1_x, mu_i_next, data_x, sum_py1_y, data_y, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y):
    w_i_x, w_i_y, mu_i, sigma_i = get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y)

    sum_x_minus_mu_py_i_x = 0
    for x in data_x:
        py_i_x = w_i_x * normal(mu_i, sigma_i, x) / (w1_x * normal(mu1, sigma1, x) + w2_x * normal(mu2, sigma2, x))
        sum_x_minus_mu_py_i_x += (x - mu_i_next) * (x - mu_i_next) * py_i_x

    sum_x_minus_mu_py_i_y = 0
    for x in data_y:
        py_i_y = w_i_y * normal(mu_i, sigma_i, x) / (w1_y * normal(mu1, sigma1, x) + w2_y * normal(mu2, sigma2, x))
        sum_x_minus_mu_py_i_y += (x - mu_i_next) * (x - mu_i_next) * py_i_y

    sigma_i_next = math.sqrt( (sum_x_minus_mu_py_i_x + sum_x_minus_mu_py_i_y) / (sum_py1_y + sum_py1_x))
    return sigma_i_next


# EM algorithm modified to use two sets of data that comes from the mixture of the same two gaussians, but each
# with different weights. The parameter of those two gaussians along with the weights of each is the return values of
# the algorithm. The parameter threshold indicates when the algorithm should stop, i.e.: when the parameters don't
# change more than that value.
def EM_modified_estimation(data_x, data_y, parameter_threshold):
    # Initialize and the weights as 0.5
    w1_x, w2_x, w1_y, w2_y = 0.5, 0.5, 0.5, 0.5

    # Initialize mu and sigma as the mean and variance of a random sample of half of the data. This is so we don't start
    # "too far" from the real parameters.
    random_sample = np.random.choice(data_x + data_y, int(np.size(data_x + data_y) / 2))
    mu1 = np.mean(random_sample)
    sigma1 = np.var(random_sample)

    random_sample = np.random.choice(data_x + data_y, int(np.size(data_x + data_y) / 2))
    mu2 = np.mean(random_sample)
    sigma2 = np.var(random_sample)

    step = 0
    while True:
        # Calculate the "t+1" parameters by applying the update rules
        w1_x_next, sum_py1_x = calculate_next_weight(1, data_x, mu1, sigma1, mu2, sigma2, w1_x, w2_x)
        w2_x_next, sum_py2_x = calculate_next_weight(2, data_x, mu1, sigma1, mu2, sigma2, w1_x, w2_x)
        w1_y_next, sum_py1_y = calculate_next_weight(1, data_y, mu1, sigma1, mu2, sigma2, w1_y, w2_y)
        w2_y_next, sum_py2_y = calculate_next_weight(2, data_y, mu1, sigma1, mu2, sigma2, w1_y, w2_y)

        mu1_next = calculate_next_mu(1, sum_py1_x, data_x, sum_py1_y, data_y, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y)
        mu2_next = calculate_next_mu(2, sum_py2_x, data_x, sum_py2_y, data_y, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y)

        sigma1_next = calculate_next_sigma(1, sum_py1_x, mu1_next, data_x, sum_py1_y, data_y, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y)
        sigma2_next = calculate_next_sigma(2, sum_py2_x, mu2_next, data_x, sum_py2_y, data_y, mu1, sigma1, mu2, sigma2, w1_x, w2_x, w1_y, w2_y)

        # Stop if all the parameter of "t+1" vary from the ones of "t" less than the threshold
        stop = abs(w1_x - w1_x_next) < parameter_threshold and \
               abs(w2_x - w2_x_next) < parameter_threshold and \
               abs(w1_y - w1_y_next) < parameter_threshold and \
               abs(w2_y - w2_y_next) < parameter_threshold and \
               abs(mu1 - mu1_next) < parameter_threshold and \
               abs(mu2 - mu2_next) < parameter_threshold and \
               abs(sigma1 - sigma1_next) < parameter_threshold and \
               abs(sigma2 - sigma2_next) < parameter_threshold

        # Assign "t+1" parameters to "t" parameters
        w1_x = w1_x_next
        w2_x = w2_x_next
        w1_y = w1_y_next
        w2_y = w2_y_next
        mu1 = mu1_next
        mu2 = mu2_next
        sigma1 = sigma1_next
        sigma2 = sigma2_next

        step += 1
        if stop:
            return w1_x, w2_x, w1_y, w2_y, mu1, mu2, sigma1, sigma2, step
