import numpy as np
import linear_dependence
import distributions


def parcorr(data, num_lags):
    n = len(data)
    res = np.empty(num_lags)
    for i in range(1, num_lags + 1):
        x = np.ones((1 + i, n - i))
        for j in range(1, i + 1):
            x[j, :] = data[i - j:n - j]
        y = data[i:n]
        beta = linear_dependence.ols_fit(x, y)
        res[i - 1] = beta[i]
    return res


def ljung_box_test(data, max_lag):
    n = len(data)
    p = parcorr(data, max_lag)
    Q = 0
    for i in range(1, max_lag + 1):
        Q += p[i - 1] * p[i - 1] / (n - i)
    Q *= n * (n + 2)
    return distributions.ChiSqrDist(mean=0, dispersion=1, degree_freedom=max_lag).ppf(Q)
