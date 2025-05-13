import mlflow
from ..logger import logging
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sktime.split import temporal_train_test_split
from alibi_detect.saving import save_detector, load_detector
import numpy as np
from ..dataops.data_loader import CSVDataLoader
from ..dataops.data_preprocessor import TimeSeriesPreprocessor
from ..dataops.data_quality import DataQuality
# Forecasting
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.boxcox import LogTransformer
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.utils.plotting import plot_series

# Model versioning
import mlflow
from sktime.utils.mlflow_sktime import (
    save_model,
    log_model,
    load_model,
    pyfunc
)

class TrainTestForecastingPipeline:
    """
    Pipeline for time series forecasting
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_loader = CSVDataLoader(config)
        self.preprocessor = TimeSeriesPreprocessor(config)
        self.data_quality = DataQuality(config) 
        self.horizon = config['forecasting']['horizon']
        self.metrics = config['forecasting']['metrics']
        self.target = config['forecasting']['target']
        self.coverage = config['quality']['coverage']
        self.p_val = config['quality']['p_val']
        self.window_size = config['quality']['window_size']

    
    def run(self) -> dict:
        """
        Execute forecasting pipeline
        """
        with mlflow.start_run():
            raw_data = self.data_loader.load_data()
            processed_data, goals_data = self.preprocessor.preprocess(raw_data)
            self.data_loader.save_data(processed_data)
            self.data_loader.save_data(goals_data)

            vendor_codes = processed_data.index.get_level_values(0).unique()

            for vendor in vendor_codes:
                vendor_data = processed_data.xs(vendor, level=0)
                y = vendor_data['valorVenda']

                try:
                    y_train, y_test = temporal_train_test_split(y, test_size=50)
                    drift_detector = self.data_quality.initialize_drift_detector(y, self.window_size)

                    plot_series(y_train, y_test, labels=["y_train", "y_test"]);

                    model = LogTransformer() * Prophet(
                        seasonality_mode='additive',
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False
                    )

                    model.fit(y)

                    unique_test_index = y_test.index.unique()
                    fh = unique_test_index

                    recent_data = y.iloc[-self.window_size:].values.reshape(-1, 1)
                    drift_preds = drift_detector.predict(recent_data)

                    drift_detected = drift_preds['data']['is_drift'] == 1
                    drift_score = drift_preds['data']['distance'][0]
                    p_value = drift_preds['data']['p_val'][0]

                    pred_intervals = model.predict_interval(fh, coverage=self.coverage)

                    lower = pred_intervals.loc[:, ([self.target], [self.coverage], ['lower'])]
                    upper = pred_intervals.loc[:, ([self.target], [self.coverage], ['upper'])]

                    if isinstance(lower, pd.DataFrame):
                        lower = lower.iloc[:, 0]
                    if isinstance(upper, pd.DataFrame):
                        upper = upper.iloc[:, 0]

                    y_pred = model.predict(fh)

                    if not self.data_quality.validate_forecast(y_pred, lower, upper, vendor):
                        logging.info('aaaa')

                except:
                    print('aaa')
                    pass

            self._log_artifacts(model, y_test, y_pred)

            return {
                'model': model,
                'forecast': y_pred
            }

        
    def _log_artifacts(self, model, y_test):
        pass