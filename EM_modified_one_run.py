"""
Runs a modified version of the EM algorithm and report the results.
This modified version takes into account two different sets of data that comes from the same two gaussians but with
different weights each.
"""
from EM_modified_methods import EM_modified_estimation, generate_mixed_data

mu1_true, sigma1_true = 38, 5
mu2_true, sigma2_true = 23, 3
beta = 0.3
alpha = 0.5
n = 1000
parameter_stop_criteria = 0.001
min_steps = 50
max_steps = 5000

# Generates n data points from two gaussians with weights beta and 1 - beta, and then with alpha and 1-alpha
Dy = generate_mixed_data(mu1_true, sigma1_true, mu2_true, sigma2_true, beta, n)
Dx = generate_mixed_data(mu1_true, sigma1_true, mu2_true, sigma2_true, alpha, n)

# Estimate the parameters using EM. The stop criteria indicates that the algorithm will stop once all the
# parameters don't change more than that value (in percentage)
w1_x, w2_x, w1_y, w2_y, mu1, mu2, sigma1, sigma2, step = EM_modified_estimation(Dx, Dy, parameter_stop_criteria, min_steps, max_steps)

print("Estimation finished!")
print("Steps: " + str(step))
print("Weights: " + str((w1_x, w2_x)) + "and " + str((w1_y, w2_y)))
print("True weights: " + str((beta, 1-beta)) + "and " + str((alpha, 1-alpha)))
print("Mu: " + str((mu1, mu2)))
print("True mus: " + str((mu1_true, mu2_true)))
print("Sigma: " + str((sigma1, sigma2)))
print("True sigmas: " + str((sigma1_true, sigma2_true)))








