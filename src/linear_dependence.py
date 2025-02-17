import numpy as np
import math


def cov_hac_beta(x, y, beta):
    a = np.multiply(x, x.T)
    a = np.linalg.inv(a)
    eps = y - np.multiply(x.T, beta)
    eps_sqr = np.square(eps)
    T = len(y)
    l = math.floor(4 * (T / 100) ** (2 / 9))
    chac = np.multiply.reduce((eps_sqr, x, x.T))
    for j in range(1, l + 1):
        for t in range(j, T):
            wj = 1 - j / (l + 1)
            eps_t_tj = eps[t] * eps[t - j]
            x_t = x[:, t]
            x_tj = x[:t - j]
            chac += wj * eps_t_tj * (np.multiply(x_t, x_tj.T) + np.multiply(x_t.T, x_tj))
    return np.multiply(np.multiply(a, chac), a)


def cov_hc_beta(x, y, beta):
    eps_sqr = np.square(y - np.multiply(x.T, beta))
    a = np.multiply(x, x.T)
    a = np.linalg.inv(a)
    x1 = np.multiply.reduce((eps_sqr, x))
    b = np.multiply(x1, x.T)
    return np.multiply(np.multiply(a, b), a)


def cov_beta(x, y, beta):
    eps = y - np.multiply(x.T, beta)
    var_eps = np.var(eps)
    a = np.multiply(x, x.T)
    return var_eps * np.linalg.inv(a)


def ols_fit(x, y):
    a = np.matmul(x, x.T)
    b = np.matmul(x, y)
    return np.linalg.solve(a, b)
