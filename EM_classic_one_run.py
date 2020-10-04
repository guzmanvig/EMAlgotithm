"""
Runs EM algorithm one time and reports the result
"""

from EM_classic_methods import generate_mixed_data, plot_data, EM_estimation

mu1_true, sigma1_true = 38, 5
mu2_true, sigma2_true = 23, 3
beta = 0.3
parameter_stop_criteria = 0.001
min_steps = 50
max_steps = 5000
n = 1000

# Generates n data points from two gaussians with weights beta and 1 - beta
Dy = generate_mixed_data(mu1_true, sigma1_true, mu2_true, sigma2_true, beta, n)

# Plots the data
plot_data(Dy, mu1_true, sigma1_true, mu2_true, sigma2_true)

# Estimate the parameters using EM. The stop criteria indicates that the algorithm will stop once all the
# parameters don't change more than that value (in percentage)
w1, w2, mu1, mu2, sigma1, sigma2, step = EM_estimation(Dy, parameter_stop_criteria, min_steps, max_steps)

print("Estimation finished!")
print("Steps: " + str(step))
print("Weights: " + str((w1, w2)))
print("True weights: " + str((beta, 1-beta)))
print("Mu: " + str((mu1, mu2)))
print("True mus: " + str((mu1_true, mu2_true)))
print("Sigma: " + str((sigma1, sigma2)))
print("True sigmas: " + str((sigma1_true, sigma2_true)))








