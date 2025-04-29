import distributions
import stat_tests
import linear_dependence
import numpy as np


class AR:
    def __init__(self, dist=distributions.GaussDist()):
        self.dist = dist
        self.criterias = distributions.criteria

    def criteria(self, name):
        return self.criterias[name](self.likelihood, self.n, len(self.ar_coef))

    def fit(self, data, ar_lags):
        n = len(data)
        self.n = n - ar_lags[-1]
        m = len(ar_lags)
        x = np.ones((1 + m, n - ar_lags[m - 1]))
        for j in range(m):
            x[j + 1, :] = data[ar_lags[m - 1] - ar_lags[j]:n - ar_lags[j]]
        y = data[ar_lags[m - 1]:n]
        self.ar_coef = linear_dependence.ols_fit(x, y)
        self.ar_lags = ar_lags
        eps = y - np.dot(x.T, self.ar_coef)
        self.dist.fit(eps)
        self.likelihood = self.dist.log_likelihood(eps, self.n)
        return eps

    def optimal_fit(self, data, max_ar, criteria='aic'):
        p = stat_tests.parcorr(data, max_ar)
        ar_lags = []
        n = len(data)
        for i in range(len(p)):
            if abs(p[i]) > 2 / n ** 0.5:
                ar_lags.append(i + 1)
        eps = self.fit(data, [ar_lags[0]])
        for i in range(1, len(ar_lags)):
            test_ar = AR()
            test_eps = test_ar.fit(data, ar_lags[:i + 1])
            if test_ar.criteria(criteria) < self.criteria(criteria):
                self.dist = test_ar.dist
                self.ar_coef = test_ar.ar_coef
                self.ar_lags = test_ar.ar_lags
                self.likelihood = test_ar.likelihood
                eps = test_eps
        return eps

    def forecast(self, data, horizont, interval=25E-2):
        forecast_data = np.zeros(horizont)
        var_error = np.zeros(horizont)
        for t in range(horizont):
            y, var = self.ar_coef[0], 1
            for i, lag in enumerate(self.ar_lags):
                if t - lag < 0:
                    y += data[len(data) + t - lag] * self.ar_coef[i + 1]
                else:
                    y += forecast_data[t - lag] * self.ar_coef[i + 1]
                    var *= self.ar_coef[i + 1] * self.ar_coef[i + 1]
            var_error[t] = var
            forecast_data[t] = y
        error = np.sqrt(var_error)
        error *= self.dist.inv_cdf(interval / 2)
        return [forecast_data, error]
