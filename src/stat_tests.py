import numpy as np
import linear_dependence
import distributions


def parcorr(data, num_lags):
    n = len(data)
    res = np.array(num_lags)
    for i in range(num_lags):
        x = np.ones(2 + i, n - i - 1)
        for j in range(i + 1):
            x[j, :] = data[j + 1:n]
        y = data[i + 1:n]
        beta = linear_dependence.ols_fit(x, y)
        res[i] = beta[i + 1]
    return res


def ljung_box_test(data, max_lag):
    n = len(data)
    p = parcorr(data, max_lag)
    Q = 0
    for i in range(1, max_lag + 1):
        Q += p[i - 1] * p[i - 1] / (n - i)
    Q *= n * (n + 2)
    return distributions.ChiSqrDist(mean=0, dispersion=1, degree_freedom=max_lag).ppf(Q)
