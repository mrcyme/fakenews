import numpy as np
from scipy.special import softmax

def greedy_choice(a, axis=None):
    max_values = np.amax(a, axis=axis, keepdims=True)
    choices = (a == max_values).astype(float)
    return choices / np.sum(choices, axis=axis, keepdims=True)

def bootstrap_ci(data,ci_val=.95,n_bootstraps=1000):

    bootstrap_means = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    lower_bound = np.percentile(bootstrap_means, (1- ci_val)/2)
    upper_bound = np.percentile(bootstrap_means, 100-(1- ci_val)/2)
    return lower_bound,upper_bound

def extrapolate_advice(a):
    b = a+0
    b[a<.5]=0
    b[a>.5]=1
    return b


def SMInv(Ainv, u, v, alpha=1):
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))
    return Ainv - np.dot(Ainv, np.dot(np.dot(u, v.T), Ainv)) / (1 + np.dot(v.T, np.dot(Ainv, u)))


def skip_diag_masking(A):
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
