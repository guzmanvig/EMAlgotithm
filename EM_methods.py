import math

import numpy as np
import matplotlib.pyplot as plt


# Generates n data points from a mixture of two Gaussian distributions
def generate_mixed_data(mu1, sigma1, mu2, sigma2, weight, n):
    first_component = np.random.normal(mu1, sigma1, int(n * weight))
    second_component = np.random.normal(mu2, sigma2, int(n * (1 - weight)))
    return first_component.tolist() + second_component.tolist()


# Plots the data along with the two true gaussian distributions. Weight is not taken into account
def plot_data(data, mu1, sigma1, mu2, sigma2):
    # Plot histogram of the data
    plt.hist(data, int(len(data) / 3), density=True)

    # Plot true distributions along with the histogram
    x = np.arange(min(mu1 - 4 * sigma1, mu2 - 4 * sigma2),
                  max(mu1 + 4 * sigma1, mu2 + 4 * sigma2), 1)

    plt.plot(x, normal(mu1, sigma1, x), linewidth=2, color='r')
    plt.plot(x, normal(mu2, sigma2, x), linewidth=2, color='g')

    plt.show()


def normal(mu, sigma, x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1, w2):
    if i == 1:
        return w1, mu1, sigma1
    else:
        return w2, mu2, sigma2


def calculate_next_weight(i, data, mu1, sigma1, mu2, sigma2, w1, w2):
    w_i, mu_i, sigma_i = get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1, w2)

    sum_py_i = 0
    for x in data:
        py_i = w_i * normal(mu_i, sigma_i, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_py_i += py_i
    wi_next = sum_py_i / len(data)
    return wi_next, sum_py_i


def calculate_next_mu(i, sum_py_i, data, mu1, sigma1, mu2, sigma2, w1, w2):
    w_i, mu_i, sigma_i = get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1, w2)

    sum_x_py_i = 0
    for x in data:
        py_i = w_i * normal(mu_i, sigma_i, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_x_py_i += x * py_i
    mu_i_next = sum_x_py_i / sum_py_i
    return mu_i_next


def calculate_next_sigma(i, sum_py_i, mu_i_next, data, mu1, sigma1, mu2, sigma2, w1, w2):
    w_i, mu_i, sigma_i = get_i_parameters(i, mu1, sigma1, mu2, sigma2, w1, w2)

    sum_x_minus_mu_py_i = 0
    for x in data:
        py_i = w_i * normal(mu_i, sigma_i, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_x_minus_mu_py_i += (x - mu_i_next) * (x - mu_i_next) * py_i
    sigma_i_next = math.sqrt(sum_x_minus_mu_py_i / sum_py_i)
    return sigma_i_next


def EM_estimation(data, parameter_threshold):
    # Initialize mu as the mean of a random sample of data and sigmas as 2, and the weights as 0.5
    w1, w2 = 0.5, 0.5
    mu1 = np.mean(np.random.choice(data, int(np.size(data) / 2)))
    mu2 = np.mean(np.random.choice(data, int(np.size(data) / 2)))
    sigma1, sigma2 = 2, 2

    step = 0
    while True:

        w1_next, sum_py_1 = calculate_next_weight(1, data, mu1, sigma1, mu2, sigma2, w1, w2)
        w2_next, sum_py_2 = calculate_next_weight(2, data, mu1, sigma1, mu2, sigma2, w1, w2)

        mu1_next = calculate_next_mu(1, sum_py_1, data, mu1, sigma1, mu2, sigma2, w1, w2)
        mu2_next = calculate_next_mu(2, sum_py_2, data, mu1, sigma1, mu2, sigma2, w1, w2)

        sigma1_next = calculate_next_sigma(1, sum_py_1, mu1_next, data, mu1, sigma1, mu2, sigma2, w1, w2)
        sigma2_next = calculate_next_sigma(2, sum_py_2, mu2_next, data, mu1, sigma1, mu2, sigma2, w1, w2)

        stop = abs(w1 - w1_next) < parameter_threshold and \
               abs(w2 - w2_next) < parameter_threshold and \
               abs(mu1 - mu1_next) < parameter_threshold and \
               abs(mu2 - mu2_next) < parameter_threshold and \
               abs(sigma1 - sigma1_next) < parameter_threshold and \
               abs(sigma2 - sigma2_next) < parameter_threshold

        w1 = w1_next
        w2 = w2_next
        mu1 = mu1_next
        mu2 = mu2_next
        sigma1 = sigma1_next
        sigma2 = sigma2_next

        step += 1
        if stop:
            return w1, w2, mu1, mu2, sigma1, sigma2, step
