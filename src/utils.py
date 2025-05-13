import matplotlib.pyplot as plt
import pandas as pd 
from sktime.utils.plotting import plot_series

def plot_vendor_histories(df, n_vendors_to_plot=None):
    """
    Plot historical sales for each vendor individually.

    Parameters:
    - df: DataFrame with multi-index (vendor_code, date)
    - n_vendors_to_plot: Number of vendors to plot (None for all)
    """
    vendor_codes = df.index.get_level_values(0).unique()

    if n_vendors_to_plot is not None:
        vendor_codes = vendor_codes[:n_vendors_to_plot]

    n_vendors = len(vendor_codes)
    fig, axes = plt.subplots(n_vendors, 1, figsize=(14, 4*n_vendors))

    if n_vendors == 1:
        axes = [axes]

    for i, vendor in enumerate(vendor_codes):
        vendor_data = df.xs(vendor, level=0)

        ax = axes[i]
        vendor_data['valorVenda'].plot(ax=ax, color='blue')

        ax.set_title(f'Vendas históricas para {vendor}')
        ax.set_xlabel('Data')
        ax.set_ylabel('Vendas')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(forecast_data, actual, predicted, title="Actual vs Predicted"):
    """Helper function to plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))

    plot_series(
        actual,
        predicted,
        labels=["Dado real", "Previsão"],
        title=title
    )
    plt.fill_between(
        forecast_data['y_pred'].index,
        forecast_data['lower_ci'],
        forecast_data['upper_ci'],
        color='blue',
        alpha=0.2,
        label='95% Intervalo de Confiança'
    )
    plt.show()

def plot_vendor_forecasts(historical_data, forecasts_dict, n_vendors_to_plot=None):
    """
    Plot historical data and forecasts for each vendor.

    Parameters:
    - historical_data: Original DataFrame with multi-index (vendor_code, date)
    - forecasts_dict: Dictionary of forecasts from vendor_forecasts
    - n_vendors_to_plot: Number of vendors to plot (None for all)
    """
    vendor_codes = list(forecasts_dict.keys())

    if n_vendors_to_plot is not None:
        vendor_codes = vendor_codes[:n_vendors_to_plot]

    n_vendors = len(vendor_codes)
    fig, axes = plt.subplots(n_vendors, 1, figsize=(14, 5*n_vendors))

    if n_vendors == 1:
        axes = [axes]

    for i, vendor in enumerate(vendor_codes):
        vendor_history = historical_data.xs(vendor, level=0)

        vendor_fcst = forecasts_dict[vendor]

        ax = axes[i]
        vendor_history['valorVenda'].plot(ax=ax, label='Historical Sales', color='blue')
        vendor_fcst.plot(ax=ax, label='Forecast', color='red', linestyle='--')

        forecast_start = vendor_fcst.index[0]
        ax.axvline(x=forecast_start, color='green', linestyle=':',
                  label=f'Início do Forecast ({forecast_start.strftime("%Y-%m-%d")})')

        ax.set_title(f'Forecast de vandas para {vendor}')
        ax.set_xlabel('Data')
        ax.set_ylabel('Vendas')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_forecast_vs_goal(historical_data, forecast_dict, goals_df, vendor_code):
    """Plot cumulative forecast progress toward annual goal"""

    history = historical_data.xs(vendor_code, level=0)
    forecast = forecast_dict[vendor_code].head(365)  # Ensure 1-year forecast
    annual_goal = goals_df.loc[goals_df['usuario_sig_id'] == vendor_code, 'venda_valor'].values[0]

    cumulative_forecast = forecast.cumsum()
    daily_goal_rate = annual_goal / 365
    goal_line = pd.Series([daily_goal_rate * i for i in range(1, 365)],
                        index=forecast.index)

    plt.figure(figsize=(14, 7))

    plt.plot(cumulative_forecast.index, cumulative_forecast,
            'b-', label='Projeção de vendas cumulativas')
    plt.plot(goal_line.index, goal_line,
            'r--', label='Trajetória de metas')

    plt.fill_between(cumulative_forecast.index,
                   cumulative_forecast,
                   goal_line,
                   where=(cumulative_forecast >= goal_line),
                   facecolor='green', alpha=0.1, interpolate=True)
    plt.fill_between(cumulative_forecast.index,
                   cumulative_forecast,
                   goal_line,
                   where=(cumulative_forecast < goal_line),
                   facecolor='red', alpha=0.1, interpolate=True)

    plt.title(f'Projeção de Metas para {vendor_code}\n(Meta anual: {annual_goal:,.0f})')
    plt.xlabel('Data')
    plt.ylabel('Vendas cumulativas')
    plt.legend()
    plt.grid(True)

    final_diff = cumulative_forecast[-1] - annual_goal
    plt.annotate(f'Projeção {"excedente" if final_diff >=0 else "deficit"}: {final_diff:,.0f}',
               xy=(cumulative_forecast.index[-1], cumulative_forecast[-1]),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
               arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.show()

def plot_vendor_forecast(vendor_id, historical_data, forecast_data):
    """
    Plot historical data and forecast with confidence intervals

    Parameters:
    - vendor_id: Vendor identifier
    - historical_data: pd.Series with historical 'valorVenda' data
    - forecast_data: Dictionary containing:
        'forecast': pd.Series of predicted values
        'lower_ci': pd.Series of lower bounds
        'upper_ci': pd.Series of upper bounds
    """
    plt.figure(figsize=(14, 7))

    # Plot historical data
    historical_data.plot(
        label='Histórico de Vendas',
        color='green',
        linewidth=2,
        alpha=0.7
    )

    # Plot forecasted data
    forecast_data['forecast'].plot(
        label='Forecast',
        color='blue',
        linewidth=2,
        style='--'  # Dashed line for forecast
    )

    # Plot confidence interval
    plt.fill_between(
        forecast_data['forecast'].index,
        forecast_data['lower_ci'],
        forecast_data['upper_ci'],
        color='blue',
        alpha=0.2,
        label='95% Intervalo de Confiança'
    )

    # Add vertical line at forecast start point
    forecast_start = forecast_data['forecast'].index[0]
    plt.axvline(
        x=forecast_start,
        color='red',
        linestyle=':',
        alpha=0.7,
        label='Início do Forecast'
    )

    # Formatting
    plt.title(f'Histórico de Vendas e Forecast para o Vendedor {vendor_id}')
    plt.xlabel('Data')
    plt.ylabel('Vendas')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Rotate x-axis labels if needed
    if len(historical_data) + len(forecast_data['forecast']) > 30:
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'../artifacts/forecast_{vendor_id}.png', dpi=300, bbox_inches='tight')

def calculate_annual_projection(forecast_df):
    """Sum forecasted values to get annual projection"""
    start_date = forecast_df.index[0]
    end_date = start_date + pd.DateOffset(days=365)
    forecast_1year = forecast_df[(forecast_df.index >= start_date) & (forecast_df.index < end_date)]

    return forecast_1year.sum()