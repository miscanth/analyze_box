from flask import jsonify, render_template, request, Blueprint
from methods import *
import pandas as pd
from parsing import parse_input_data
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Создаем Blueprint маршрутизатор
views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def my_index_view():
    return render_template('index.html')

@views_bp.route('/analyze', methods=['POST'])
def analyze():
    # Проверьте, был ли загружен файл
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if file.filename.endswith(('.csv', '.xlsx', '.xls')):
            try:
                if file.filename.endswith('.csv'):
                    data = pd.read_csv(file)
                else:
                    data = pd.read_excel(file)

                data = data.dropna(axis=1, how='all')
                if data.empty:
                    return jsonify({'error': 'No data found after preprocessing'}), 400

                # преобразуйте столбцы объектов с небольшим количеством уникальных значений в категорию
                for col in data.select_dtypes(include=['object']):
                    if data[col].nunique() < 10:
                        data[col] = data[col].astype('category')

            except Exception as e:
                return jsonify({'error': f"Error reading file: {str(e)}"}), 400
    else:
        return jsonify({'error': 'No data provided'}), 400

    # Сгенерируйте все анализы
    results = {
        'descriptive': analyze_descriptive(data),
        'correlation': analyze_correlation(data),
        'anova': analyze_anova(data),
        'clustering': analyze_clustering(data)
    }

    # Remove None values
    results = {k: v for k, v in results.items() if v is not None}

    return jsonify(results)



@views_bp.route('/anova', methods=['GET', 'POST'])
def anova():
    # Гарантируем, что всегда есть минимум 2 группы
    default_groups = ["1, 2, 3", "4, 5, 6"]
    initial_groups = default_groups
    anova_result = None
    plots = None
    conclusion = ""

    if request.method == 'POST':
        groups = []
        i = 1
        while f'group{i}' in request.form:
            val = request.form[f'group{i}'].strip()
            if val:
                groups.append(val)
            i += 1
        if groups: # Если есть данные от пользователя — используем их
            initial_groups = groups

        data = []
        labels = []
        for idx, group_str in enumerate(groups):
            values = parse_input_data(group_str)
            if values:
                data.extend(values)
                labels.extend([f'Group {idx + 1}'] * len(values))

        if len(set(labels)) < 2 or len(data) == 0:
            conclusion = "Not enough data or groups."
        else:
            df = pd.DataFrame({'Group': labels, 'Value': data})
            unique_groups = sorted(set(labels), key=lambda x: int(x.split()[1]))  # Группы 1, 2, 3...

            group_data = [df[df['Group'] == g]['Value'].values for g in unique_groups]
            if all(len(arr) > 0 for arr in group_data):
                try:
                    f_value, p_value = stats.f_oneway(*group_data)
                    anova_result = {'f_value': f_value, 'p_value': p_value}
                    conclusion = "Statistically significant differences (p < 0.05)" if p_value < 0.05 else "No significant differences (p ≥ 0.05)"
                    plots = create_anova_plots(df, unique_groups)
                except Exception as e:
                    conclusion = f"Error in ANOVA: {str(e)}"
            else:
                conclusion = "One or more groups are empty."

    return render_template(
        'anova.html',
        active_page='anova',
        initial_groups=initial_groups,
        anova_result=anova_result,
        anova_conclusion=conclusion,
        plots=plots
    )


@views_bp.route('/ttest', methods=['GET', 'POST'])
def ttest():
    group1 = "1, 2, 3, 4, 5"
    group2 = "2, 3, 4, 5, 6"
    equal_var = True
    result = None
    plots = None
    conclusion = ""

    if request.method == 'POST':
        group1 = request.form.get('group1', group1)
        group2 = request.form.get('group2', group2)
        equal_var = 'equal_var' in request.form

        data1 = parse_input_data(group1)
        data2 = parse_input_data(group2)

        if len(data1) > 1 and len(data2) > 1:
            stat, p = stats.ttest_ind(data1, data2, equal_var=equal_var)
            result = {'statistic': stat, 'pvalue': p}
            conclusion = "Means are significantly different (reject H0)" if p < 0.05 else "No significant difference in means (fail to reject H0)"
            plots = create_ttest_plots(data1, data2)

    return render_template(
        'ttest.html',
        active_page='ttest',
        group1=group1,
        group2=group2,
        equal_var=equal_var,
        result=result,
        conclusion=conclusion,
        plots=plots
    )


@views_bp.route('/shapiro', methods=['GET', 'POST'])
def shapiro():
    data_input = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
    result = None
    plots = None
    conclusion = ""

    if request.method == 'POST':
        data_input = request.form.get('data', data_input)
        data = parse_input_data(data_input)

        if len(data) > 3 and len(data) < 5000:  # Shapiro-Wilk применим для 3–5000 значений
            stat, p = stats.shapiro(data)
            result = {'statistic': stat, 'pvalue': p}
            conclusion = "Data looks normally distributed (fail to reject H0)" if p > 0.05 else "Data does not look normally distributed (reject H0)"
            plots = create_shapiro_plots(data)

    return render_template(
        'shapiro.html',
        active_page='shapiro',
        data_input=data_input,
        result=result,
        conclusion=conclusion,
        plots=plots
    )


@views_bp.route('/levene', methods=['GET', 'POST'])
def levene():
    initial_groups = ["1, 2, 3", "4, 5, 6", "7, 8, 9"]
    result = None
    plots = None
    conclusion = ""

    if request.method == 'POST':
        # Получаем данные из формы
        groups = []
        i = 1
        while f'group{i}' in request.form:
            groups.append(request.form[f'group{i}'])
            i += 1
        initial_groups = groups

        # Подготавливаем данные для теста Левен
        data = []
        labels = []
        for i, group_str in enumerate(groups):
            values = parse_input_data(group_str)
            if values:
                data.append(values)
                labels.append(f'Group {i + 1}')

        if len(data) >= 2:
            # Выполняем тест Левена
            stat, p = stats.levene(*data)
            result = {'statistic': stat, 'pvalue': p}
            conclusion = "Variances are equal (fail to reject H0)" if p > 0.05 else "Variances are not equal (reject H0)"

            # Подготовка данных для визуализации
            plot_data = []
            plot_labels = []
            for i, group in enumerate(data):
                plot_data.extend(group)
                plot_labels.extend([labels[i]] * len(group))
            df = pd.DataFrame({'Group': plot_labels, 'Value': plot_data})
            plots = create_levene_plots(df, labels)

    return render_template(
        'levene.html',
        active_page='levene',
        initial_groups=initial_groups,
        result=result,
        conclusion=conclusion,
        plots=plots
    )


@views_bp.route('/mannwhitney', methods=['GET', 'POST'])
def mannwhitney():
    group1 = "1, 2, 3, 4, 5"
    group2 = "6, 7, 8, 9, 10"
    use_continuity = True
    alternative = 'two-sided'
    result = None
    plots = None
    conclusion = ""

    if request.method == 'POST':
        group1 = request.form.get('group1', group1)
        group2 = request.form.get('group2', group2)
        use_continuity = 'use_continuity' in request.form
        alternative = 'less' if 'alternative' in request.form else 'two-sided'

        data1 = parse_input_data(group1)
        data2 = parse_input_data(group2)

        if len(data1) > 0 and len(data2) > 0:
            stat, p = stats.mannwhitneyu(data1, data2,
                                         use_continuity=use_continuity,
                                         alternative=alternative)
            result = {'statistic': stat, 'pvalue': p}
            conclusion = "Distributions are significantly different (reject H0)" if p < 0.05 else "No significant difference in distributions (fail to reject H0)"
            plots = create_mannwhitney_plots(data1, data2)

    return render_template(
        'mannwhitney.html',
        active_page='mannwhitney',
        group1=group1,
        group2=group2,
        use_continuity=use_continuity,
        alternative=alternative,
        result=result,
        conclusion=conclusion,
        plots=plots
    )


@views_bp.route('/fisher', methods=['GET', 'POST'])
def fisher():
    a = "10"
    b = "20"
    c = "15"
    d = "15"
    alternative = 'two-sided'
    result = None
    plots = None
    conclusion = ""

    if request.method == 'POST':
        a = int(request.form.get('a', a))
        b = int(request.form.get('b', b))
        c = int(request.form.get('c', c))
        d = int(request.form.get('d', d))
        alternative = 'less' if 'alternative' in request.form else 'two-sided'

        # Создаём таблицу сопряжённости
        table = [[a, b], [c, d]]

        # Выполняем точный тест Фишера
        oddsratio, p = stats.fisher_exact(table, alternative=alternative)

        result = {'oddsratio': oddsratio, 'pvalue': p}
        conclusion = "Significant association (reject H0)" if p < 0.05 else "No significant association (fail to reject H0)"
        plots = create_fisher_plot(a, b, c, d)

    return render_template(
        'fisher.html',
        active_page='fisher',
        a=a,
        b=b,
        c=c,
        d=d,
        alternative=alternative,
        result=result,
        conclusion=conclusion,
        plots=plots
    )


@views_bp.route('/tukey', methods=['GET', 'POST'])
def tukey():
    initial_groups = ["1, 2, 3", "4, 5, 6", "7, 8, 9"]
    result = None
    plots = None  # Словарь для хранения графиков
    if request.method == 'POST':
        groups = []
        i = 1
        while f'group{i}' in request.form:
            group_data = request.form[f'group{i}']
            if group_data.strip():
                groups.append(group_data)
            i += 1
        if not groups:
            groups = initial_groups
        else:
            initial_groups = groups

        data = []
        labels = []
        for i, group_str in enumerate(groups):
            values = parse_input_data(group_str)
            if values:
                data.extend(values)
                labels.extend([f'Group {i + 1}'] * len(values))
        if len(set(labels)) >= 2:
            try:
                df = pd.DataFrame({'Group': labels, 'Value': data})
                unique_labels = sorted(set(labels))

                # Проверим, что данные не пустые
                if len(df) == 0:
                    raise ValueError("No data available for the Tukey test.")

                tukey_result = pairwise_tukeyhsd(df['Value'], df['Group'])

                result = []
                for i in range(len(tukey_result.groupsunique)):
                    for j in range(i + 1, len(tukey_result.groupsunique)):
                        idx = j - 1 - i * (i + 1) // 2
                        result.append({
                            'group1': tukey_result.groupsunique[i],
                            'group2': tukey_result.groupsunique[j],
                            'meandiff': tukey_result.meandiffs[idx],
                            'pvalue': tukey_result.pvalues[idx],
                            'reject': tukey_result.reject[idx]
                        })

                # Передаем график в словарь
                plots = {
                    'tukeyplot': create_tukey_plot(df, unique_labels)
                }

            except Exception as e:
                print(f"Error in Tukey test: {str(e)}")
                result = None
                plots = None

    return render_template(
        'tukey.html',
        active_page='tukey',
        initial_groups=initial_groups,
        result=result,
        plots=plots  # Теперь plots — это словарь
    )

@views_bp.route('/irq', methods=['GET', 'POST'])
def irq():
    data_input = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100"
    result = None
    plots = None

    if request.method == 'POST':
        data_input = request.form.get('data', data_input)
        data = parse_input_data(data_input)

        if len(data) > 0:
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers = [x for x in data if x < lower_fence or x > upper_fence]

            result = {
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_fence': lower_fence,
                'upper_fence': upper_fence,
                'outliers': outliers
            }
            plots = create_irq_plots(data)

    return render_template(
        'irq.html',
        active_page='irq',
        data_input=data_input,
        result=result,
        plots=plots
    )


@views_bp.route('/pearson', methods=['GET', 'POST'])
def pearson():
    return correlation_analysis('pearson', 'Pearson')


@views_bp.route('/spearman', methods=['GET', 'POST'])
def spearman():
    return correlation_analysis('spearman', 'Spearman')


@views_bp.route('/kendall', methods=['GET', 'POST'])
def kendall():
    return correlation_analysis('kendall', 'Kendall')


@views_bp.route('/linear_regression', methods=['GET', 'POST'])
def linear_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    result = None
    plots = None

    if request.method == 'POST':
        x_values = request.form.get('x_values', x_values)
        y_values = request.form.get('y_values', y_values)

        x = np.array(parse_input_data(x_values)).reshape(-1, 1)
        y = np.array(parse_input_data(y_values))

        if len(x) == len(y) and len(x) > 1:
            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Обучение модели линейной регрессии
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Предсказание значений
            y_pred = model.predict(X_test)

            # Вычисление метрик
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Словарь с результатами
            result = {
                'intercept': model.intercept_,
                'coefficient': model.coef_[0],
                'r_squared': r2,
                'mse': mse
            }

            # Построение линии регрессии
            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
            y_range = model.predict(x_range)

            plots = {
                'regression_plot': create_regression_plot(
                    x.flatten(), y,
                    x_range.flatten(), y_range,
                    'Linear Regression'
                )
            }

    return render_template(
        'regression.html',
        active_page='linear_regression',
        title='Linear Regression',
        regression_type='linear',
        x_values=x_values,
        y_values=y_values,
        result=result,
        plots=plots
    )


@views_bp.route('/polynomial_regression', methods=['GET', 'POST'])
def polynomial_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "1, 4, 9, 16, 25"
    degree = "2"
    result = None
    plots = None

    if request.method == 'POST':
        x_values = request.form.get('x_values', x_values)
        y_values = request.form.get('y_values', y_values)
        degree = request.form.get('degree', degree)

        x = np.array(parse_input_data(x_values)).reshape(-1, 1)
        y = np.array(parse_input_data(y_values))
        degree = int(degree)

        if len(x) == len(y) and len(x) > 1:
            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Создание полиномиальных признаков
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            # Обучение модели
            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            # Предсказание значений
            y_pred = model.predict(X_test_poly)

            # Вычисление метрик
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Словарь с результатами
            result = {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': r2,
                'mse': mse
            }

            # Строим графики
            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
            x_range_poly = poly.transform(x_range)
            y_range = model.predict(x_range_poly)

            plots = {
                'regression_plot': create_regression_plot(
                    x.flatten(), y,
                    x_range.flatten(), y_range,
                    f'Polynomial Regression (Degree {degree})'
                )
            }

    return render_template(
        'regression.html',
        active_page='polynomial_regression',
        title='Polynomial Regression',
        regression_type='polynomial',
        x_values=x_values,
        y_values=y_values,
        degree=degree,
        result=result,
        plots=plots
    )


@views_bp.route('/ridge_regression', methods=['GET', 'POST'])
def ridge_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    alpha = 1.0
    result = None
    plots = None

    if request.method == 'POST':
        x_values = request.form.get('x_values', x_values)
        y_values = request.form.get('y_values', y_values)
        alpha = float(request.form.get('alpha', alpha))

        x = np.array(parse_input_data(x_values)).reshape(-1, 1)
        y = np.array(parse_input_data(y_values))

        if len(x) == len(y) and len(x) > 1:
            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Обучение модели
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            # Предсказание значений
            y_pred = model.predict(X_test)
            # Вычисление метрик
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            # Словарь с результатами
            result = {
                'intercept': model.intercept_,
                'coefficient': model.coef_[0],
                'r_squared': r2,
                'mse': mse
            }

            # Построение линии регрессии
            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
            y_range = model.predict(x_range)

            plots = {
                'regression_plot': create_regression_plot(
                    x.flatten(), y, x_range.flatten(), y_range, 'Ridge Regression'
                )
            }

    return render_template(
        'regression.html',
        active_page='ridge_regression',
        title='Ridge Regression',
        regression_type='ridge',
        x_values=x_values,
        y_values=y_values,
        result=result,
        plots=plots,
        alpha=alpha
    )


@views_bp.route('/lasso_regression', methods=['GET', 'POST'])
def lasso_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    alpha = 1.0
    result = None
    plots = None

    if request.method == 'POST':
        x_values = request.form.get('x_values', x_values)
        y_values = request.form.get('y_values', y_values)
        alpha = float(request.form.get('alpha', alpha))

        x = np.array(parse_input_data(x_values)).reshape(-1, 1)
        y = np.array(parse_input_data(y_values))

        if len(x) == len(y) and len(x) > 1:
            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Обучение модели
            model = Lasso(alpha=alpha)
            model.fit(X_train, y_train)
            # Предсказание значений
            y_pred = model.predict(X_test)
            # Вычисление метрик
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            # Словарь с результатами
            result = {
                'intercept': model.intercept_,
                'coefficient': model.coef_[0],
                'r_squared': r2,
                'mse': mse
            }

            # Построение линии регрессии
            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
            y_range = model.predict(x_range)

            plots = {
                'regression_plot': create_regression_plot(
                    x.flatten(), y, x_range.flatten(), y_range, 'Lasso Regression'
                )
            }

    return render_template(
        'regression.html',
        active_page='lasso_regression',
        title='Lasso Regression',
        regression_type='lasso',
        x_values=x_values,
        y_values=y_values,
        result=result,
        plots=plots,
        alpha=alpha
    )


@views_bp.route('/logistic_regression', methods=['GET', 'POST'])
def logistic_regression():
    x_values = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
    y_values = "0, 0, 0, 0, 1, 1, 1, 1, 1, 1"
    result = None
    plots = None

    if request.method == 'POST':
        x_values = request.form.get('x_values', x_values)
        y_values = request.form.get('y_values', y_values)

        x = np.array(parse_input_data(x_values)).reshape(-1, 1)
        y = np.array(parse_input_data(y_values))

        if len(x) == len(y) and len(x) > 1:
            #  Добавление случайной второй признаковой переменной для визуализации в 2D
            x = np.hstack([x, np.random.randn(len(x), 1)])

            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Масштабирование данных
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Обучение модели
            model = LogisticRegression(max_iter=1000, solver='liblinear')
            model.fit(X_train, y_train)

            # Предсказание значений
            y_pred = model.predict(X_test)

            # Вычисление точности
            accuracy = (y_pred == y_test).mean()

            # Словарь с результатами
            result = {
                'intercept': model.intercept_[0],
                'coefficients': model.coef_[0].tolist(),
                'accuracy': accuracy
            }

            # Строим график
            plots = {
                'regression_plot': create_logistic_plot(x, y, model)
            }

    return render_template(
        'regression.html',
        active_page='logistic_regression',
        title='Logistic Regression',
        regression_type='logistic',
        x_values=x_values,
        y_values=y_values,
        result=result,
        plots=plots
    )