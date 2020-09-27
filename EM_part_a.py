import math

import numpy as np
import matplotlib.pyplot as plt


def normal(mu, sigma, x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


mu1_true, sigma1_true = 18, 8
mu2_true, sigma2_true = 9, 2
beta = 0.3
n = 1000
parameter_threshold = 0.001

first_component = np.random.normal(mu1_true, sigma1_true, int(n * beta))
second_component = np.random.normal(mu2_true, sigma2_true, int(n * (1 - beta)))
Dy = first_component.tolist() + second_component.tolist()

# Plot histogram of the data
plt.hist(Dy, int(n/3), density=True)

# Plot true distributions along with the histogram
x = np.arange(min(mu1_true - 4 * sigma1_true, mu2_true - 4 * sigma2_true), max(mu1_true + 4 * sigma1_true, mu2_true + 4 * sigma2_true), 1)

plt.plot(x, normal(mu1_true, sigma1_true, x), linewidth=2, color='r')
plt.plot(x, normal(mu2_true, sigma2_true, x), linewidth=2, color='g')

plt.show()

# Start of EM algorithm

w1, w2 = 0.5, 0.5
# Initialize mu as the mean of a random sample of Dy and sigmas as 2
mu1 = np.mean(np.random.choice(Dy, int(np.size(Dy)/2)))
mu2 = np.mean(np.random.choice(Dy, int(np.size(Dy)/2)))
sigma1, sigma2 = 2, 2

step = 0
while True:

    # Calculate next weights
    sum_py_1 = 0
    for x in Dy:
        py_1 = w1 * normal(mu1, sigma1, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_py_1 += py_1
    w1_next = sum_py_1 / len(Dy)

    sum_py_2 = 0
    for x in Dy:
        py_2 = w2 * normal(mu2, sigma2, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_py_2 += py_2
    w2_next = sum_py_2 / len(Dy)

    # Calculate next mu's
    sum_x_py_1 = 0
    for x in Dy:
        py_1 = w1 * normal(mu1, sigma1, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_x_py_1 += x * py_1
    mu1_next = sum_x_py_1 / sum_py_1

    sum_x_py_2 = 0
    for x in Dy:
        py_2 = w2 * normal(mu2, sigma2, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_x_py_2 += x * py_2
    mu2_next = sum_x_py_2 / sum_py_2

    # Calculate next sigma's
    sum_x_minus_mu_py_1 = 0
    for x in Dy:
        py_1 = w1 * normal(mu1, sigma1, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_x_minus_mu_py_1 += (x - mu1_next) * (x - mu1_next) * py_1
    sigma1_next = math.sqrt(sum_x_minus_mu_py_1/sum_py_1)

    sum_x_minus_mu_py_2 = 0
    for x in Dy:
        py_2 = w2 * normal(mu2, sigma2, x) / (w1 * normal(mu1, sigma1, x) + w2 * normal(mu2, sigma2, x))
        sum_x_minus_mu_py_2 += (x - mu2_next) * (x - mu2_next) * py_2
    sigma2_next = math.sqrt(sum_x_minus_mu_py_2 / sum_py_2)

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
        break


print("Estimation finished!")
print("Steps: " + str(step))
print("Weights: " + str((w1, w2)))
print("True weights: " + str((beta, 1-beta)))
print("Mu: " + str((mu1, mu2)))
print("True mus: " + str((mu1_true, mu2_true)))
print("Sigma: " + str((sigma1, sigma2)))
print("True sigmas: " + str((sigma1_true, sigma2_true)))








