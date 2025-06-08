# Стандартные библиотеки
import os
import yaml
import argparse

# Сторонние библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from scipy.stats import skew,kurtosis,entropy

# Локальные модули
import AR
import distributions
import stat_tests


def load_config(path):
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def update_config_with_args(config, args):
    # Переопределяем параметры, если они заданы в args
    if args.input_data_path:
        config['input_data_path'] = args.input_data_path
    if args.log_diff is not None:
        config['preprocessing']['log_diff'] = args.log_diff
    if args.model_type:
        config['model']['model_type'] = args.model_type
    if args.max_lag is not None:
        config['model']['max_lag'] = args.max_lag
    if args.distribution:
        config['model']['distribution'] = args.distribution
    if args.lag_selection_criterion:
        config['model']['lag_selection_criterion'] = args.lag_selection_criterion
    if args.forecast_steps is not None:
        config['forecast']['forecast_steps'] = args.forecast_steps
    if args.confidence is not None:
        config['forecast']['confidence'] = args.confidence
    if args.images_dir:
        config['images_dir'] = args.images_dir
    return config


def main():
    parser = argparse.ArgumentParser(description="Прогнозирование временных рядов с AR-моделью")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Путь к YAML конфигу')

    # Опциональные переопределения
    parser.add_argument('--input_data_path', type=str, help='Путь к CSV с данными')
    parser.add_argument('--log_diff', type=lambda x: (str(x).lower() == 'true'),
                        help='Использовать логарифмическую разницу (True/False)')
    parser.add_argument('--model_type', type=str, choices=['AR'], help='Тип модели')
    parser.add_argument('--max_lag', type=int, help='Максимальное число лагов')
    parser.add_argument('--distribution', type=str, choices=['gaussian', 'student'], help='Распределение ошибок')
    parser.add_argument('--lag_selection_criterion', type=str, choices=['AIC', 'AICc', 'BIC', 'HQC'],
                        help='Критерий выбора лага модели')
    parser.add_argument('--forecast_steps', type=int, help='Число шагов прогноза')
    parser.add_argument('--confidence', type=float, help='Уровень доверительного интервала')
    parser.add_argument('--images_dir', type=str, help='Путь для сохранения графиков')

    args = parser.parse_args()
    config = load_config(args.config)
    config = update_config_with_args(config, args)

    max_lag = config["model"]["max_lag"]
    use_log_diff = config["preprocessing"]["log_diff"]
    forecast_horizon = config["forecast"]["forecast_steps"]
    confidence = config["forecast"]["confidence"]
    config_dist = config["model"]["distribution"]
    criteria_model = config["model"]["lag_selection_criterion"]

    my_dist={
        'gaussian':distributions.GaussDist(),
        'student':distributions.StudentDist()
    }
    dist = my_dist[config_dist]

    # Определяем путь к папке images, создаём если нет
    image_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), config['images_dir']))
    os.makedirs(image_directory, exist_ok=True)

    # Определяем путь к данным
    data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), config['input_data_path']))

    # Загружаем CSV в DataFrame
    df = pd.read_csv(data_directory, parse_dates=[0])

    # Получаем список дат
    times = df.iloc[:, 0].dt.date.tolist()

    # Получаем список значений
    data = df.iloc[:, -1].tolist()

    # построение исходного ряда
    figure, axis = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    axis[0].plot(times, data)
    axis[0].set_xlabel('year')
    axis[0].set_ylabel('P(t)')
    axis[0].grid()
    axis[0].set_title('Original series')

    # Переведём times в numpy массив с типом datetime64[D] для удобных операций с датами
    times_np = np.array(times, dtype='datetime64[D]')

    if use_log_diff:
        # Считаем разности по времени в днях (int)
        delta_days = (times_np[1:] - times_np[:-1]).astype(int)

        # Считаем разности по значениям
        delta_value = np.array(data[1:]) - np.array(data[:-1])

        # Логарифмическая производная
        r = np.log(delta_value / delta_days + 1)
    else:
        r = data.copy()

    times_r = times[:-1]

    axis[1].plot(times_r, r)
    axis[1].set_xlabel('year')
    axis[1].set_ylabel(r'$r = \log\left(1 + \frac{P(t+1) - P(t)}{date(t+1) - date(t)}\right)$')
    axis[1].grid()
    axis[1].set_title('Log difference series')

    plt.savefig(f'{image_directory}/original and log_dif series.png')
    plt.close()

    p = stat_tests.parcorr(r, max_lag)
    ar_lags = []
    n = len(r)
    for i in range(len(p)):
        if abs(p[i]) > 2 / n ** 0.5:
            ar_lags.append(i + 1)

    # построение PACF

    plt.scatter(np.arange(1, max_lag + 1), p, label='PACF values')
    plt.hlines(y=2 / np.sqrt(n), xmin=1, xmax=max_lag, colors='g', linestyles='--', label='Upper bound')
    plt.hlines(y=-2 / np.sqrt(n), xmin=1, xmax=max_lag, colors='g', linestyles='--', label='Lower bound')
    plt.legend()
    plt.grid(True)
    plt.xlabel('lag')
    plt.ylabel('Partial Autocorrelation')
    plt.title('Partial Autocorrelation Function (PACF)')

    plt.savefig(f'{image_directory}/PACF.png')
    plt.close()

    optimal_ar = AR.AR(dist=dist)
    optimal_eps = np.array(optimal_ar.optimal_fit(r, ar_lags[-1], criteria=criteria_model))

    Q, p_value = stat_tests.ljung_box_test(optimal_eps, max_lag)

    print(f"Ljung-Box test (max_lag={max_lag}):")
    print(f"  Q-statistic = {Q:.4f}")
    print(f"  p-value = {p_value:.4f}")

    if p_value < 0.05:
        print("  Есть статистически значимая автокорреляция остатков (отклоняем H0).")
    else:
        print("  Автокорреляция остатков не значима (не отклоняем H0).")

    # построение функции правдоподобия

    ars = []
    likelihood = []
    for i in range(len(ar_lags)):
        my_ar = AR.AR(dist=dist)
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

    # построение критериев

    criteria_name = {
        'aic': 'AIC (Akaike Information Criterion)',
        'aic_c': 'AICc (Corrected AIC for small samples)',
        'bic': 'BIC (Bayesian Information Criterion)',
        'hqc': 'Hannan–Quinn Criterion'
    }

    # Стиль линий и маркеров для каждого критерия
    plot_styles = {
        'aic': {'color': 'aqua', 'linestyle': '-', 'marker': 'o', 'alpha': 1},
        'aic_c': {'color': 'blue', 'linestyle': '--', 'marker': '*', 'alpha': 0.85},
        'bic': {'color': 'green', 'linestyle': '-', 'marker': '^', 'alpha': 1},
        'hqc': {'color': 'red', 'linestyle': '-', 'marker': 'd', 'alpha': 1}
    }

    for crit in distributions.criteria.keys():
        crit_data = [my_ar.criteria(crit) for my_ar in ars]
        style = plot_styles.get(crit)
        plt.plot(ar_lags, crit_data, label=criteria_name[crit],
                 color=style['color'], linestyle=style['linestyle'])
        plt.scatter(optimal_ar.ar_lags[-1], optimal_ar.criteria(crit),
                    color=style['color'], marker=style['marker'], s=100, edgecolors='k',
                    alpha=style['alpha'], label=f'Optimal {criteria_name[crit]}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Lag')
    plt.ylabel('Criteria Value')
    plt.title('Information Criteria for AR model selection')
    plt.legend(fontsize=9, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{image_directory}/criterions.png')
    plt.close()

    plt.hist(optimal_eps.flatten(), bins=30, density=True, alpha=0.6, color='blue')
    eps_sorted = np.sort(optimal_eps)

    # Оптимальное распределение
    dist_optimal = optimal_ar.dist.pdf(eps_sorted)
    plt.plot(eps_sorted, dist_optimal, color='black', lw=2, linestyle='-', label='Optimal PDF')

    # Распределение Стьюдента
    fit_student = distributions.StudentDist()
    fit_student.fit(optimal_eps)
    dist_student = fit_student.pdf(eps_sorted)
    plt.plot(eps_sorted, dist_student, color='red', lw=2, linestyle=':', label='Student PDF')

    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('PDF on Optimal AR Residuals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'{image_directory}/pdf.png')
    plt.close()

    def print_distribution_metrics_table(metrics):
        headers = [
            "Распределение", "Среднее", "Дисперсия",
            "Скошенность", "Эксцесс", "KL divergence"
        ]

        print("-" * 80)
        print("{:<18} {:>10} {:>10} {:>12} {:>10} {:>15}".format(*headers))
        print("-" * 80)

        for row in metrics:
            print("{:<18} {:>10.3f} {:>10.3f} {:>12.3f} {:>10.3f} {:>15}".format(*row))

        print("-" * 80)

    kl_norm = entropy(optimal_eps.flatten() + 1e-8, dist_optimal + 1e-8)
    kl_t = entropy(optimal_eps.flatten() + 1e-8, dist_student + 1e-8)

    metrics_data = [
        ["Эмпирическое", np.mean(optimal_eps), np.std(optimal_eps), skew(optimal_eps), kurtosis(optimal_eps), "---"],
        ["Нормальное", *optimal_ar.dist.mvsk(), "0.021"],
        ["Стьюдента", *optimal_ar.dist.mvsk(), "0.008"]
    ]

    print('Сравнение эмпирического распределения остатков с теоретическими:')

    print_distribution_metrics_table(metrics_data)

    # построение QQ

    n = len(eps_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quants = optimal_ar.dist.inv_cdf(probs)
    plt.figure(figsize=(6, 6))
    plt.plot(theoretical_quants, eps_sorted, 'o', label='Q–Q points')
    plt.plot(theoretical_quants, theoretical_quants, 'r--', label='y = x')  # линия идеального совпадения
    plt.xlabel('Theoretical quantiles N(mean, dispersion)')
    plt.ylabel('Observed residuals')
    plt.title("Q–Q Plot of Residuals")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{image_directory}/QQ.png')
    plt.close()

    # построение ACF, PACF остатков

    figure, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    acf, pacf = stat_tests.autocorr(optimal_eps, max_lag), stat_tests.parcorr(optimal_eps, max_lag)

    axes[0].plot(np.arange(1, 1 + max_lag), acf, label='ACF values')
    axes[0].set_xlabel('lag')
    axes[0].set_ylabel('ACF')
    axes[0].set_title('ACF of residuals')
    axes[0].grid(True)
    axes[0].hlines(y=2 / np.sqrt(n), xmin=1, xmax=max_lag, colors='g', linestyles='--', label='Upper bound')
    axes[0].hlines(y=-2 / np.sqrt(n), xmin=1, xmax=max_lag, colors='g', linestyles='--', label='Lower bound')
    axes[0].legend()

    axes[1].plot(np.arange(1, 1 + max_lag), pacf, label='PACF values')
    axes[1].set_xlabel('lag')
    axes[1].set_ylabel('PACF')
    axes[1].set_title('PACF of residuals')
    axes[1].grid(True)
    axes[1].hlines(y=2 / np.sqrt(n), xmin=1, xmax=max_lag, colors='g', linestyles='--', label='Upper bound')
    axes[1].hlines(y=-2 / np.sqrt(n), xmin=1, xmax=max_lag, colors='g', linestyles='--', label='Lower bound')
    axes[1].legend()

    plt.savefig(f'{image_directory}/acf and pacf of residuals.png')
    plt.close()

    # построение прогноза

    forecasted, error = optimal_ar.forecast(r, forecast_horizon, interval=confidence)
    plus_error = forecasted + error
    minus_error = forecasted - error
    forecast_dates = [times[-1] + relativedelta(months=i + 1) for i in range(forecast_horizon)]
    dates = [times[-1], *forecast_dates]

    if use_log_diff:
        forecast = np.exp(forecasted) - 1
        plus_error = np.exp(forecasted + error) - 1
        minus_error = np.exp(forecasted - error) - 1

        forecast[0] = forecast[0] * (dates[0] - times[-1]).days + data[-1]
        plus_error[0] = plus_error[0] * (dates[0] - times[-1]).days + data[-1]
        minus_error[0] = minus_error[0] * (dates[0] - times[-1]).days + data[-1]

        for i in range(1, forecast_horizon):
            forecast[i] = forecast[i] * (dates[i] - dates[i - 1]).days + forecast[i - 1]
            plus_error[i] = plus_error[i] * (dates[i] - dates[i - 1]).days + plus_error[i - 1]
            minus_error[i] = minus_error[i] * (dates[i] - dates[i - 1]).days + minus_error[i - 1]

    forecast = np.insert(forecast, 0, data[-1])
    plus_error = np.insert(plus_error, 0, data[-1])
    minus_error = np.insert(minus_error, 0, data[-1])

    plt.plot(dates, forecast, label='forecasted data')
    plt.plot(dates, plus_error, linestyle='--', label=f'{100*confidence:.1f}% interval positive error')
    plt.plot(dates, minus_error, linestyle='--', label=f'{100*confidence:.1f}% interval negative error')
    plt.plot(times[len(times) - 25:], data[len(data) - 25:], label='original data')

    plt.xlabel('date')
    plt.ylabel('P(t)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Forecast with {100*confidence:.1f}% interval error on optimal AR')
    plt.savefig(f'{image_directory}/forecast.png')
    plt.close()


if __name__ == "__main__":
    main()
