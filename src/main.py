import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import distributions
import math
import AR
import stat_tests

image_directory = 'images'

times = []
data = []
with open('Electric_Production.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    first = True
    for row in reader:
        if first:
            first = False
            continue
        else:
            data.append(float(row[-1]))
            times.append(datetime.datetime.strptime(row[0], "%Y-%m-%d").date())

figure, axis = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))

axis[0].plot(times, data)
axis[0].set_xlabel('year')
axis[0].set_ylabel('P(t)')
axis[0].grid()
axis[0].set_title('Original series')

r = []
for i in range(len(data) - 1):
    r.append(np.log((data[i + 1] - data[i]) / (times[i + 1] - times[i]).days + 1))
r = np.array(r)

axis[1].plot(times[:len(times) - 1], r)
axis[1].set_xlabel('year')
axis[1].set_ylabel(r'$r=log(1 + \frac{P(t+1)-P(t)}{date(t+1)-date(t)})$')
axis[1].grid()
axis[1].set_title('Log difference series')
plt.savefig(f'{image_directory}/original and log_dif series.png')
plt.close()

max_lag = math.floor(4 * (len(r) / 100) ** (2 / 9))
max_lag = 50

ar_models = []

p = stat_tests.parcorr(r, max_lag)
ar_lags = []
n = len(r)
for i in range(len(p)):
    if abs(p[i]) > 2 / n ** 0.5:
        ar_lags.append(i + 1)

plt.scatter(np.arange(1, 1 + max_lag), p)
plt.axhline(y=2 / n ** 0.5, color='g', linestyle='--', label='two upper normal boundary')
plt.axhline(y=-2 / n ** 0.5, color='g', linestyle='--', label='two lower normal boundary')
plt.legend()
plt.grid()
plt.xlabel('lag')
plt.title('Partial autocorrelation function')
plt.savefig(f'{image_directory}/PACF.png')
plt.close()

optimal_ar = AR.AR()
optimal_eps = np.array(optimal_ar.optimal_fit(r, ar_lags[-1]))
ars = []
likelihood = []
for i in range(len(ar_lags)):
    my_ar = AR.AR()
    my_ar.fit(r, ar_lags[:i + 1])
    ars.append(my_ar)
    likelihood.append(my_ar.likelihood)

plt.plot(ar_lags, likelihood)
plt.scatter(optimal_ar.ar_lags[-1], optimal_ar.likelihood, label='optimal')
plt.xlabel('lag')
plt.ylabel(r'$\log{L}+\frac{n}{2}\cdot\log{2\pi}$')
plt.title("Log likelihood")
plt.grid()
plt.legend()
plt.savefig(f'{image_directory}/likelihood.png')
plt.close()

criteria_name = {
    'aic': 'Aic information criterion',
    'bic': 'Bayes information criterion',
    'hqc': 'Hannan–Quinn information criterion'
}

for crit in distributions.criteria.keys():
    n = len(r)
    crit_data = []
    for my_ar in ars:
        crit_data.append(my_ar.criteria(crit))
    plt.plot(ar_lags, crit_data, label=criteria_name[crit])
    plt.scatter(optimal_ar.ar_lags[-1], optimal_ar.criteria(crit), label='optimal')
plt.legend()
plt.grid()
plt.xlabel('lag')
plt.title('Information criterions')
plt.savefig(f'{image_directory}/criterions.png')
plt.close()

plt.hist(optimal_eps.flatten(), bins=30, density=True)
eps_sorted = np.sort(optimal_eps)
dist_optimal = optimal_ar.dist.pdf(eps_sorted)
plt.plot(eps_sorted, dist_optimal)
plt.xlabel('error')
plt.ylabel('density')
plt.title('Gauss pdf on optimal AR')
plt.savefig(f'{image_directory}/pdf.png')
plt.close()

forecasted, error = optimal_ar.forecast(r, 20, interval=95E-2)
dates = ['2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01', '2018-07-01', '2018-08-01',
         '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01', '2019-01-01', '2019-02-01', '2019-03-01',
         '2019-04-01', '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01', '2019-09-01']
dates = [datetime.datetime.strptime(x, "%Y-%m-%d").date() for x in dates]
forecast = np.exp(forecasted) - 1
plus_error = np.exp(forecasted + error) - 1
minus_error = np.exp(forecasted - error) - 1
forecast[0] = forecast[0] * (dates[0] - times[-1]).days + data[-1]
plus_error[0] = plus_error[0] * (dates[0] - times[-1]).days + data[-1]
minus_error[0] = minus_error[0] * (dates[0] - times[-1]).days + data[-1]
for i in range(1, 20):
    forecast[i] = forecast[i] * (dates[i] - dates[i - 1]).days + forecast[i - 1]
    plus_error[i] = plus_error[i] * (dates[i] - dates[i - 1]).days + plus_error[i - 1]
    minus_error[i] = minus_error[i] * (dates[i] - dates[i - 1]).days + minus_error[i - 1]
forecast=np.insert(forecast,0,data[-1])
plus_error=np.insert(plus_error,0,data[-1])
minus_error=np.insert(minus_error,0,data[-1])
dates=[times[-1],*dates]
plt.plot(dates,forecast, label='forecasted data')
plt.plot(dates,plus_error,linestyle = '--', label='95% interval positive error')
plt.plot(dates,minus_error,linestyle = '--', label='95% interval negative error')
plt.plot(times[len(times)-25:],data[len(data)-25:], label='original data')
plt.xlabel('date')
plt.ylabel('P(t)')
plt.legend()
plt.title('Forecast with 95% interval error on optimal AR')
plt.savefig(f'{image_directory}/forecast.png')
plt.close()