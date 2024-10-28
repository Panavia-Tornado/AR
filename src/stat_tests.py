import numpy as np
import linear_dependence


def parcorr(data, num_lags):
    n = len(data)
    res = np.array(num_lags)
    for i in range(num_lags):
        x = np.ones(2 + i, n - i - 1)
        for j in range(i+1):
            x[j, :] = data[j + 1:n]
        y = data[i + 1:n]
        beta = linear_dependence.ols_fit(x, y)
        res[i] = beta[i + 1]
    return res
