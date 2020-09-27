from EM_methods import generate_mixed_data, plot_data, EM_estimation

mu1_true, sigma1_true = 18, 8
mu2_true, sigma2_true = 9, 2
beta = 0.3
n = 1000
parameter_threshold = 0.001

Dy = generate_mixed_data(mu1_true, sigma1_true, mu2_true, sigma2_true, beta, n)
plot_data(Dy, mu1_true, sigma1_true, mu2_true, sigma2_true)
w1, w2, mu1, mu2, sigma1, sigma2, step = EM_estimation(Dy, parameter_threshold)

print("Estimation finished!")
print("Steps: " + str(step))
print("Weights: " + str((w1, w2)))
print("True weights: " + str((beta, 1-beta)))
print("Mu: " + str((mu1, mu2)))
print("True mus: " + str((mu1_true, mu2_true)))
print("Sigma: " + str((sigma1, sigma2)))
print("True sigmas: " + str((sigma1_true, sigma2_true)))








