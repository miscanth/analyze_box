import pandas as pd
from flask import render_template, request
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from parsing import parse_input_data
import base64
import plotly.express as px
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --- Функции для построения статистических графиков ---
def create_anova_plots(df, labels):
    """Создаём графики для визуализации ANOVA"""
    try:
        df['Group'] = pd.Categorical(df['Group'], categories=labels, ordered=True)
        df = df.sort_values('Group')
        group_stats = df.groupby('Group')['Value'].agg(['mean', 'std', 'count']).reindex(labels)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Boxplot', 'Means with 95% CI', 'Distribution', 'Q-Q per Group'),
            specs=[[{"type": "box"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )

        colors = [f'hsl({i * 137}, 50%, 50%)' for i in range(len(labels))]  # распределение цветов

        # 1. Boxplot
        for i, group in enumerate(labels):
            data = df[df['Group'] == group]['Value']
            fig.add_trace(go.Box(y=data, name=group, marker_color=colors[i]), row=1, col=1)

        # 2. Средние значения с доверительными интервалами
        ci = 1.96 * group_stats['std'] / np.sqrt(group_stats['count'])
        fig.add_trace(go.Bar(
            x=labels,
            y=group_stats['mean'],
            error_y=dict(type='data', array=ci, visible=True),
            marker_color=colors
        ), row=1, col=2)

        # 3. Распределения
        for i, group in enumerate(labels):
            data = df[df['Group'] == group]['Value']
            fig.add_trace(go.Histogram(x=data, name=group, opacity=0.6, marker_color=colors[i]), row=2, col=1)

        # 4. Q-Q графики
        for i, group in enumerate(labels):
            data = df[df['Group'] == group]['Value']
            (osm, osr), _ = stats.probplot(data, dist="norm")
            fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name=group, marker_color=colors[i]), row=2, col=2)

        fig.update_layout(height=600, showlegend=False, title_text="ANOVA Diagnostics")
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

        return {
            'summary_plot': fig.to_html(full_html=False, include_plotlyjs=True, div_id="summaryPlot")
        }
    except Exception as e:
        print(f"Error in create_anova_plots: {e}")
        return {}


def create_ttest_plots(data1, data2):
    """Создаём графики для сравнения двух групп с помощью t-теста"""
    # Сравнение Boxplot
    boxplot = go.Figure()
    boxplot.add_trace(go.Box(
        y=data1,
        name='Group 1',
        boxpoints='all',
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    boxplot.add_trace(go.Box(
        y=data2,
        name='Group 2',
        boxpoints='all',
        marker_color='rgba(214, 39, 40, 0.7)'
    ))
    boxplot.update_layout(title='Boxplot Comparison')

    # Сравнение гистограмм
    hist = go.Figure()
    hist.add_trace(go.Histogram(
        x=data1,
        name='Group 1',
        opacity=0.7,
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    hist.add_trace(go.Histogram(
        x=data2,
        name='Group 2',
        opacity=0.7,
        marker_color='rgba(214, 39, 40, 0.7)'
    ))
    hist.update_layout(
        title='Distribution Comparison',
        barmode='overlay',
        xaxis_title='Value',
        yaxis_title='Frequency'
    )

    # Средние значения с доверительными интервалами
    means_fig = go.Figure()
    means_fig.add_trace(go.Bar(
        x=['Group 1', 'Group 2'],
        y=[np.mean(data1), np.mean(data2)],
        error_y=dict(
            type='data',
            array=[1.96 * np.std(data1) / np.sqrt(len(data1)),
                   1.96 * np.std(data2) / np.sqrt(len(data2))],
            visible=True
        ),
        marker_color=['rgba(55, 128, 191, 0.7)', 'rgba(214, 39, 40, 0.7)']
    ))
    means_fig.update_layout(
        title='Means with 95% Confidence Intervals',
        yaxis_title='Mean Value'
    )

    return {
        'boxplot': boxplot.to_html(full_html=False, include_plotlyjs=False),
        'histogram': hist.to_html(full_html=False, include_plotlyjs=False),
        'means_plot': means_fig.to_html(full_html=False, include_plotlyjs=False)
    }


def create_shapiro_plots(data):
    """Создаём графики для теста Шапиро-Уилка на нормальность"""
    # Преобразуем массив NumPy в список, если необходимо
    if isinstance(data, np.ndarray):
        data = data.tolist()

    # Гистограмма с наложением нормального распределения
    hist = go.Figure()
    hist.add_trace(go.Histogram(
        x=data,
        name='Data',
        opacity=0.7,
        marker_color='rgba(55, 128, 191, 0.7)',
        histnorm='probability density'
    ))

    # Кривая нормального распределения
    x = np.linspace(min(data), max(data), 100).tolist()
    pdf = stats.norm.pdf(x, np.mean(data), np.std(data)).tolist()
    hist.add_trace(go.Scatter(
        x=x, y=pdf,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='rgba(214, 39, 40, 0.7)')
    ))

    hist.update_layout(
        title='Distribution with Normal Fit',
        xaxis_title='Value',
        yaxis_title='Density',
        barmode='overlay'
    )

    # Q-Q график
    qq = stats.probplot(data, dist="norm")
    qq_x = qq[0][0].tolist()
    qq_y = qq[0][1].tolist()
    qq_line_x = qq[0][0].tolist()
    qq_line_y = (qq[1][0] * qq[0][0] + qq[1][2]).tolist()

    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(
        x=qq_x, y=qq_y,
        mode='markers',
        name='Data Points'
    ))
    qq_fig.add_trace(go.Scatter(
        x=qq_line_x, y=qq_line_y,
        mode='lines',
        name='Theoretical Normal'
    ))
    qq_fig.update_layout(
        title='Q-Q Plot',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles'
    )

    return {
        'distplot': hist.to_html(full_html=False, include_plotlyjs=False),
        'qqplot': qq_fig.to_html(full_html=False, include_plotlyjs=False)
    }


def create_levene_plots(df, labels):
    """Создаём графики для теста Левена"""
    # Boxplot
    fig = go.Figure()

    for i, group in enumerate(labels):
        # Преобразуем данные в список, если это необходимо
        group_data = df[df['Group'] == group]['Value']
        if isinstance(group_data, (np.ndarray, pd.Series)):
            group_data = group_data.tolist()

        fig.add_trace(go.Box(
            y=group_data,
            name=group,
            boxpoints='all',
            marker_color=f'hsl({i * 40}, 50%, 50%)',
            jitter=0.3,
            pointpos=-1.8
        ))

    fig.update_layout(
        title='Boxplot by Group',
        yaxis_title='Values',
        showlegend=True
    )

    return {
        'boxplot': fig.to_html(full_html=False, include_plotlyjs=False)
    }


def create_mannwhitney_plots(group1, group2):
    """Создаём графики для теста Манна–Уитни"""
    # Boxplot
    boxplot = go.Figure()
    boxplot.add_trace(go.Box(
        y=group1,
        name='Group 1',
        boxpoints='all',
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    boxplot.add_trace(go.Box(
        y=group2,
        name='Group 2',
        boxpoints='all',
        marker_color='rgba(214, 39, 40, 0.7)'
    ))
    boxplot.update_layout(title='Boxplot Comparison')

    # Гистограммы
    hist = go.Figure()
    hist.add_trace(go.Histogram(
        x=group1,
        name='Group 1',
        opacity=0.7,
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    hist.add_trace(go.Histogram(
        x=group2,
        name='Group 2',
        opacity=0.7,
        marker_color='rgba(214, 39, 40, 0.7)'
    ))
    hist.update_layout(
        title='Distribution Comparison',
        barmode='overlay',
        xaxis_title='Value',
        yaxis_title='Frequency'
    )

    return {
        'boxplot': boxplot.to_html(full_html=False, include_plotlyjs=False),
        'histogram': hist.to_html(full_html=False, include_plotlyjs=False)
    }


def create_fisher_plot(a, b, c, d):
    """Создаём график для визуализации результатов точного теста Фишера"""
    labels = ['Group 1', 'Group 2']
    success = [a, c]
    failure = [b, d]

    fig = go.Figure(data=[
        go.Bar(name='Success', x=labels, y=success, marker_color='rgba(55, 128, 191, 0.7)'),
        go.Bar(name='Failure', x=labels, y=failure, marker_color='rgba(214, 39, 40, 0.7)')
    ])

    fig.update_layout(
        title='Contingency Table Visualization',
        barmode='group',
        yaxis_title='Count'
    )

    return {
        'barplot': fig.to_html(full_html=False, include_plotlyjs=False)
    }


def create_tukey_plot(df, labels):
    """Создаём боксплот для визуализации результатов критерия Тьюки"""
    try:
        fig = go.Figure()
        for i, group in enumerate(labels):
            group_data = df[df['Group'] == group]['Value']
            if len(group_data) == 0:
                continue  # Пропустить пустую группу
            fig.add_trace(go.Box(
                y=group_data,
                name=group,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker_color=f'hsl({i * 360 / len(labels)}, 70%, 50%)',
                line_color=f'hsl({i * 360 / len(labels)}, 70%, 50%)'
            ))
        fig.update_layout(
            title='Group Comparisons (Boxplot)',
            yaxis_title='Values',
            showlegend=True,
            height=500
        )
        # Возвращаем HTML-репрезентацию графика
        return fig.to_html(full_html=False, include_plotlyjs=True)
    except Exception as e:
        print(f"Error creating Tukey plot: {str(e)}")
        return "<p>Error generating plot</p>"


def create_irq_plots(data):
    """Создаём графики для анализа межквартильного размаха (IQR)"""
    # Boxplot
    boxplot = go.Figure()
    boxplot.add_trace(go.Box(
        y=data,
        boxpoints='all',
        name='Data',
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    boxplot.update_layout(title='Boxplot with Outliers')

    return {
        'boxplot': boxplot.to_html(full_html=False, include_plotlyjs=False)
    }


def create_correlation_plots(x, y, x_name="Variable 1", y_name="Variable 2"):
    """Создаём графики для визуализации корреляции между двумя переменными"""
    # Scatter plot
    scatter = go.Figure()
    scatter.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='Data points',
        marker=dict(color='rgba(55, 128, 191, 0.7)', size=10)
    ))

    # Добавляем линию регрессии
    slope, intercept, _, _, _ = stats.linregress(x, y)
    scatter.add_trace(go.Scatter(
        x=x, y=slope * np.array(x) + intercept,
        mode='lines', name='Regression line',
        line=dict(color='rgba(214, 39, 40, 0.7)')
    ))
    scatter.update_layout(title=f'{x_name} vs {y_name}')

    # Гистограммы переменных X и Y
    hist1 = go.Figure()
    hist1.add_trace(go.Histogram(x=x, name=x_name))
    hist1.update_layout(title=f'Distribution of {x_name}')

    hist2 = go.Figure()
    hist2.add_trace(go.Histogram(x=y, name=y_name))
    hist2.update_layout(title=f'Distribution of {y_name}')

    return {
        'scatter': scatter.to_html(full_html=False, include_plotlyjs=False),
        'hist1': hist1.to_html(full_html=False, include_plotlyjs=False),
        'hist2': hist2.to_html(full_html=False, include_plotlyjs=False)
    }


def create_regression_plot(x, y, x_pred, y_pred, title):
    """Создаём график линейной регрессии с использованием Plotly"""

    fig = go.Figure()

    # Добавляем реальные точки данных
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Actual Data',
        marker=dict(color='blue', size=10)
    ))

    # Добавляем линию регрессии
    fig.add_trace(go.Scatter(
        x=x_pred, y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='X values',
        yaxis_title='Y values',
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_logistic_plot(x, y, model):
    """Создаём график с границей решений для логистической регрессии"""
    try:
        # Создаём сетку значений для построения границы решений
        h = 0.02  # шаг сетки
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Предсказываем значения по всей сетке
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Создаем графики
        fig = go.Figure()

        # Добавляем границу решений
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale=[[0, 'blue'], [1, 'red']],
            opacity=0.3,
            showscale=False,
            name='Decision Boundary'
        ))

        # Добавляем точки данных
        fig.add_trace(go.Scatter(
            x=x[:, 0], y=x[:, 1],
            mode='markers',
            marker=dict(
                color=y,
                colorscale=[[0, 'blue'], [1, 'red']],
                size=10,
                line=dict(color='black', width=1)
            ),
            name='Data Points'
        ))

        fig.update_layout(
            title='Logistic Regression Decision Boundary',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            showlegend=True,
            autosize=True
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')  # Используем CDN для Plotly.js
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        return None
    
# --- Логика статистических анализов ---
def correlation_analysis(method, title):
    var1 = "1, 2, 3, 4, 5"
    var2 = "2, 3, 4, 5, 6"
    result = None
    plots = None
    conclusion = ""

    if request.method == 'POST':
        var1 = request.form.get('var1', var1)
        var2 = request.form.get('var2', var2)

        x = parse_input_data(var1)
        y = parse_input_data(var2)

        if len(x) == len(y) and len(x) > 1:
            if method == 'pearson':
                corr, p_value = stats.pearsonr(x, y)
            elif method == 'spearman':
                corr, p_value = stats.spearmanr(x, y)
            else:  # kendall
                corr, p_value = stats.kendalltau(x, y)

            result = {'coefficient': corr, 'p_value': p_value}
            conclusion = "Statistically significant correlation (p < 0.05)" if p_value < 0.05 else "No significant correlation (p ≥ 0.05)"
            plots = create_correlation_plots(x, y, "Variable 1", "Variable 2")

    return render_template(
        'correlation.html',
        title=title,
        active_page=method.lower(),
        var1=var1,
        var2=var2,
        result=result,
        conclusion=conclusion,
        plots=plots
    )


def fig_to_base64(fig):
    """Convert plotly figure to base64 encoded image"""
    img_bytes = fig.to_image(format="png")
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


def analyze_descriptive(data):
    """Описательная статистика"""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return None

    results = {
        'stats': '',
        'plots': []
    }

    # Описательная статистика
    try:
        desc_stats = numeric_data.describe().round(3)
        results['stats'] = desc_stats.to_html(classes="table table-striped")
    except Exception as e:
        results['stats'] = f"<div class='alert alert-danger'>Error: {str(e)}</div>"

    # Визуализации (первые 3 числовых столбца)
    for col in numeric_data.columns[:3]:
        try:
            # Гистограмма
            fig = px.histogram(numeric_data, x=col, title=f"Distribution of {col}")
            results['plots'].append(fig_to_base64(fig))

            # Boxplot
            fig = px.box(numeric_data, y=col, title=f"Boxplot of {col}")
            results['plots'].append(fig_to_base64(fig))
        except:
            continue

    return results


def analyze_correlation(data):
    """Correlation analysis"""
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) < 2:
        return None

    try:
        #Корреляционная матрица
        corr_matrix = numeric_data.corr().round(3)
        matrix_html = corr_matrix.to_html(classes="table table-striped")

        # Тепловая карта
        fig = px.imshow(corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        title="Correlation Matrix",
                        width=600,
                        height=600)
        plot_base64 = fig_to_base64(fig)

        return {
            'plot': plot_base64,
            'matrix': matrix_html
        }
    except Exception as e:
        print(f"Correlation error: {str(e)}")
        return None


def analyze_anova(data):
    """ANOVA analysis"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    # Для ручного ввода (формат групп)
    if len(numeric_cols) == 0 and len(data.columns) > 0:
        # Melt the dataframe for ANOVA
        df_melt = data.melt(var_name='Group', value_name='Value').dropna()
        groups = df_melt.groupby('Group')['Value'].apply(list)

        # Односторонний ANOVA
        f_val, p_val = stats.f_oneway(*groups)

        # Таблица результатов
        results_df = pd.DataFrame({
            'Source': ['Between Groups', 'Within Groups'],
            'F-value': [round(f_val, 4), ''],
            'p-value': [round(p_val, 4), '']
        })
        results_html = results_df.to_html(classes="table table-striped", index=False)

        # Визуализация
        fig = px.box(df_melt, x='Group', y='Value', title="ANOVA Results")
        plot_base64 = fig_to_base64(fig)

        # Последующий тест, если он значимый (p < 0,05) и более чем в 2 группах
        posthoc_html = ""
        if p_val < 0.05 and len(groups) > 2:
            try:
                tukey_results = pairwise_tukeyhsd(df_melt['Value'], df_melt['Group'])
                posthoc_df = pd.DataFrame(data=tukey_results._results_table.data[1:],
                                          columns=tukey_results._results_table.data[0])
                posthoc_html = posthoc_df.to_html(classes="table table-striped", index=False)
            except Exception as e:
                posthoc_html = f"<div class='alert alert-warning'>Post-hoc test error: {str(e)}</div>"

        return {
            'plot': plot_base64,
            'results': results_html,
            'posthoc': posthoc_html
        }

    # Для загрузки файлов с категориальными и числовыми столбцами
    elif len(numeric_cols) > 0 and len(categorical_cols) > 0:
        # Используйте первый числовой и первый категориальный столбцы
        num_col = numeric_cols[0]
        cat_col = categorical_cols[0]

        try:
            # Подготовка данных
            groups = data.groupby(cat_col)[num_col].apply(list)

            # Односторонний ANOVA
            f_val, p_val = stats.f_oneway(*groups)

            # Таблица результатов
            results_df = pd.DataFrame({
                'Source': ['Between Groups', 'Within Groups'],
                'F-value': [round(f_val, 4), ''],
                'p-value': [round(p_val, 4), '']
            })
            results_html = results_df.to_html(classes="table table-striped", index=False)

            # Визуализацияn
            fig = px.box(data, x=cat_col, y=num_col, title=f"ANOVA: {num_col} by {cat_col}")
            plot_base64 = fig_to_base64(fig)

            # Последующий тест, если он значимый (p < 0,05) и более чем в 2 группах
            posthoc_html = ""
            if p_val < 0.05 and len(groups) > 2:
                try:
                    # Подготовьте данные для теста Тьюки
                    values = []
                    labels = []
                    for group_name, group_values in groups.items():
                        values.extend(group_values)
                        labels.extend([str(group_name)] * len(group_values))

                    tukey_results = pairwise_tukeyhsd(values, labels)
                    posthoc_df = pd.DataFrame(data=tukey_results._results_table.data[1:],
                                              columns=tukey_results._results_table.data[0])
                    posthoc_html = posthoc_df.to_html(classes="table table-striped", index=False)
                except Exception as e:
                    posthoc_html = f"<div class='alert alert-warning'>Post-hoc test error: {str(e)}</div>"

            return {
                'plot': plot_base64,
                'results': results_html,
                'posthoc': posthoc_html
            }
        except Exception as e:
            print(f"ANOVA error: {str(e)}")
            return None

    return None


def analyze_clustering(data):
    """Clustering analysis"""
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) < 2:
        return None

    try:
        # Стандартизация данных
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # Определите оптимальное значение k, используя метод elbow (попробуйте 2-5 кластеров)
        wcss = []
        for i in range(2, 6):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        # Соответствует оптимальному k (используя 3 для визуализации)
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        # Добавление кластеров к данным
        plot_data = numeric_data.copy()
        plot_data['Cluster'] = clusters.astype(str)

        # Создайте трехмерную точечную диаграмму
        if len(numeric_data.columns) >= 3:
            fig = px.scatter_3d(
                plot_data,
                x=numeric_data.columns[0],
                y=numeric_data.columns[1],
                z=numeric_data.columns[2],
                color='Cluster',
                title="Cluster Visualization (3D)"
            )
        else:
            fig = px.scatter(
                plot_data,
                x=numeric_data.columns[0],
                y=numeric_data.columns[1],
                color='Cluster',
                title="Cluster Visualization"
            )

        plot_base64 = fig_to_base64(fig)

        # Кластерная статистика
        stats_df = plot_data.groupby('Cluster').mean().round(3)
        stats_html = stats_df.to_html(classes="table table-striped")

        return {
            'plot': plot_base64,
            'stats': stats_html
        }
    except Exception as e:
        print(f"Clustering error: {str(e)}")
        return None