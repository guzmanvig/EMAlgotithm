import numpy as np

from EM_methods import generate_mixed_data, plot_data, EM_estimation

mu1_true, sigma1_true = 35, 5
mu2_true, sigma2_true = 23, 3
beta = 0.4
n = 1000
parameter_threshold = 0.001

w1_all, w2_all, mu1_all, mu2_all, sigma1_all, sigma2_all, step_all = [], [], [], [], [], [], []
for i in range(0, 100):
    print("Run: ", i)
    Dy = generate_mixed_data(mu1_true, sigma1_true, mu2_true, sigma2_true, beta, n)
    w1, w2, mu1, mu2, sigma1, sigma2, step = EM_estimation(Dy, parameter_threshold)

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
