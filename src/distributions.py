import scipy
import numpy as np
import math

criteria = {
    'aic': lambda log_l, n, k: 2 * k - 2 * log_l,
    'bic': lambda log_l, n, k: k * math.log(n) - 2 * log_l,
    'hqc': lambda log_l, n, k: 2 * k * math.log(math.log(n)) - 2 * log_l
}


class GaussDist:
    def __init__(self, mean=None, dispersion=None):
        self.mean = mean
        self.dispersion = dispersion

    def fit(self, data):
        res = scipy.stats.norm.fit(data)
        self.mean, self.dispersion = res

    def log_likelihood(self, res, tol_mean=1E-6):
        n = len(res)
        if abs(self.mean) < tol_mean:
            return -n / (2 * math.log(self.dispersion ** 2))
        else:
            return -n / (2 * math.log(self.dispersion ** 2)) - np.sum(np.square(res - self.mean)) / (
                    2 * self.dispersion ** 2)


class StudentDist:
    def __init__(self, mean=None, dispersion=None, degree_freedom=None):
        self.mean = mean
        self.dispersion = dispersion
        self.degree_freedom = degree_freedom

    def fit(self, data):
        res = scipy.stats.norm.fit(data)
        self.mean, self.dispersion, self.degree_freedom = res

    def log_likelihood(self, res, tol_mean=1E-6):
        n = len(res)
        if self.mean < tol_mean:
            return (self.degree_freedom + 1) / 2 * math.log(
                np.prod(1 + res / ((self.degree_freedom - 2) * self.dispersion))) + 1 / 2 * math.log(
                self.dispersion ** 2)
        else:
            return (self.degree_freedom + 1) / 2 * math.log(
                np.prod(1 + (res - self.mean) / ((self.degree_freedom - 2) * self.dispersion))) + 1 / 2 * math.log(
                self.dispersion ** 2)

    def log_likelihood_full(self, res, degree_freedom, tol_mean=1E-6):
        self.mean = np.mean(res)
        self.dispersion = (np.var(res - self.mean))**0.5
        self.degree_freedom = degree_freedom
        n = len(res)
        return self.log_likelihood(res, tol_mean) + n * (math.log(math.gamma((degree_freedom + 1) / 2)) - math.log(
            math.gamma((degree_freedom + 1) / 2)) - 0.5 * math.log((degree_freedom - 2) * math.pi))


class ChiSqrDist:
    def __init__(self, mean=None, dispersion=None, degree_freedom=None):
        self.mean = mean
        self.dispersion = dispersion
        self.degree_freedom = degree_freedom

    def ppf(self, q):
        return scipy.stats.chi2.ppf(q, self.mean, self.dispersion, self.degree_freedom)
