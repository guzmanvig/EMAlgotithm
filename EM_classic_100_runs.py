"""
Runs the algorithm 100 times and calculates the mean and variance of the results of each run
"""

import numpy as np

from EM_classic_methods import generate_mixed_data, EM_estimation

mu1_true, sigma1_true = 38, 5
mu2_true, sigma2_true = 23, 3
beta = 0.4
parameter_stop_criteria = 0.001
min_steps = 50
max_steps = 5000
# n = 1000
n = 100


# Here we will store the results
w1_all, w2_all, mu1_all, mu2_all, sigma1_all, sigma2_all, step_all = [], [], [], [], [], [], []

for i in range(0, 100):
    print("Run: ", i)
    # Generate data each time and run the algorithm
    Dy = generate_mixed_data(mu1_true, sigma1_true, mu2_true, sigma2_true, beta, n)

    # Estimate the parameters using EM. The stop criteria indicates that the algorithm will stop once all the
    # parameters don't change more than that value (in percentage)
    w1, w2, mu1, mu2, sigma1, sigma2, step = EM_estimation(Dy, parameter_stop_criteria, min_steps, max_steps)

    # Sets (w1, mu1, sigma1) from this run might not correspond to the same (w1, mu1, sigma1) from before runs,
    # so we need to check to which group it is more likely they belong. We use mu to check since mus are more likely to
    # be different from each other than the other parameters (because otherwise it would not be modeled as a mixture)
    if i != 0 and abs(mu1 - mu1_all[0]) < abs(mu1 - mu2_all[0]):
        w1_all.append(w1)
        w2_all.append(w2)
        mu1_all.append(mu1)
        mu2_all.append(mu2)
        sigma1_all.append(sigma1)
        sigma2_all.append(sigma2)
    else:
        w2_all.append(w1)
        w1_all.append(w2)
        mu2_all.append(mu1)
        mu1_all.append(mu2)
        sigma2_all.append(sigma1)
        sigma1_all.append(sigma2)

    step_all.append(step)

print("Average finished! Results:")
print("Mean of steps: " + str(np.mean(step_all)))
print("-----------------------------------------")
print("Mean of weights: " + str((np.mean(w1_all), np.mean(w2_all))))
print("Variance of weights: " + str((np.var(w1_all), np.var(w2_all))))
print("True weights: " + str((beta, 1-beta)))
print("-----------------------------------------")
print("Mean of mus: " + str((np.mean(mu1_all), np.mean(mu2_all))))
print("Variance of mus: " + str((np.var(mu1_all), np.var(mu2_all))))
print("True mus: " + str((mu1_true, mu2_true)))
print("-----------------------------------------")
print("Mean of sigmas: " + str((np.mean(sigma1_all), np.mean(sigma2_all))))
print("Variance of sigmas: " + str((np.var(sigma1_all), np.var(sigma2_all))))
print("True sigmas: " + str((sigma1_true, sigma2_true)))
