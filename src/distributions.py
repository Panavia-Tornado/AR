import scipy
import numpy as np
import math

criteria = {
    'aic': lambda log_l, n, k: 2 * k - 2 * log_l,
    'aic_c': lambda log_l, n, k: 2 * k - 2 * log_l + (2 * k * k + 2 * k) / (n - k - 1),
    'bic': lambda log_l, n, k: k * np.log(n) - 2 * log_l,
    'hqc': lambda log_l, n, k: 2 * k * np.log(np.log(n)) - 2 * log_l
}


class GaussDist:
    def __init__(self, mean=None, dispersion=None):
        self.mean = mean
        self.dispersion = dispersion

    def fit(self, data):
        self.mean = np.mean(data)
        self.dispersion = np.std(data, ddof=1)

    def pdf(self, x):
        return scipy.stats.norm.pdf(x, loc=self.mean, scale=self.dispersion)

    def cdf(self, x):
        return scipy.stats.norm.cdf(x, loc=self.mean, scale=self.dispersion)

    def inv_cdf(self, x):
        return scipy.stats.norm.ppf(x, loc=self.mean, scale=self.dispersion)

    def log_likelihood(self, res, n, tol_mean=1E-6):
        if abs(self.mean) < tol_mean:
            return -n / 2 * np.log(self.dispersion ** 2)
        else:
            return -n / 2 * (np.log(self.dispersion ** 2) + np.sum(np.square(res - self.mean)) / (
                    self.dispersion ** 2))

    def mvsk(self):
        return [self.mean, self.dispersion, 0, 0]

    def copy(self):
        return GaussDist(mean=self.mean, dispersion=self.dispersion)


class StudentDist:
    def __init__(self, mean=None, dispersion=None, degree_freedom=None):
        self.mean = mean
        self.dispersion = dispersion
        self.degree_freedom = degree_freedom

    def fit(self, data):
        self.mean = np.mean(data)
        self.dispersion = np.std(data)
        res = scipy.stats.t.fit(data, floc=np.mean(data), fscale=np.std(data))
        self.degree_freedom, self.mean, self.dispersion = res

    def pdf(self, x):
        return scipy.stats.t.pdf(x, self.degree_freedom, loc=self.mean, scale=self.dispersion)

    def cdf(self, x):
        return scipy.stats.t.cdf(x, self.degree_freedom, loc=self.mean, scale=self.dispersion)

    def inv_cdf(self, x):
        return scipy.stats.t.ppf(x, self.degree_freedom, loc=self.mean, scale=self.dispersion)

    def log_likelihood(self, res, degree_freedom=None, tol_mean=1E-6):
        self.mean = np.mean(res)
        self.dispersion = np.std(res, ddof=1)
        n = len(res)
        if degree_freedom == None:
            l = n * (np.log(np.gamma((degree_freedom + 1) / 2)) - np.log(
                np.gamma((degree_freedom + 1) / 2)) - 0.5 * np.log((degree_freedom - 2) * np.pi))
        else:
            l = 0
            self.degree_freedom = degree_freedom
        if self.mean < tol_mean:
            return l + (self.degree_freedom + 1) / 2 * np.log(
                np.prod(1 + res / ((self.degree_freedom - 2) * self.dispersion))) + 1 / 2 * np.log(
                self.dispersion ** 2)
        else:
            return l + (self.degree_freedom + 1) / 2 * np.log(
                np.prod(1 + (res - self.mean) / ((self.degree_freedom - 2) * self.dispersion))) + 1 / 2 * np.log(
                self.dispersion ** 2)

    def copy(self):
        return StudentDist(degree_freedom=self.degree_freedom, mean=self.mean, dispersion=self.dispersion)

    def mvsk(self):
        return scipy.stats.t.stats(moments='mvsk', df=self.degree_freedom, loc=self.mean, scale=self.dispersion)


class ChiSqrDist:
    def __init__(self, degree_freedom=None):
        self.degree_freedom = degree_freedom

    def ppf(self, q):
        return scipy.stats.chi2.ppf(q, self.degree_freedom)

    def cdf(self, q):
        return scipy.stats.chi2.cdf(q, self.degree_freedom)
