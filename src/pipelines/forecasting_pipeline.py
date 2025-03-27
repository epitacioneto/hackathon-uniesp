import mlflow
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from fbprophet import Prophet
from ..dataops.data_loader import CSVDataLoader
from ..dataops.data_preprocessor import TimeSeriesPreprocessor

class ForecastingPipeline:
    """
    Pipeline for time series forecasting
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_loader = CSVDataLoader(config)
        self.preprocessor = TimeSeriesPreprocessor(config)
        self.horizon = config['forecasting']['horizon']
        self.metrics = config['forecasting']['metrics']
    
    def run(self) -> dict:
        """
        Execute forecasting pipeline
        """
        with mlflow.start_run():
            raw_data = self.data_loader.load_data()
            mlflow.log_text(raw_data.head().to_markdown(), "raw_data_sample.md")
            processed_data = self.preprocessor.preprocess(raw_data)
            self.data_loader.save_data(processed_data)

            model = Prophet()
            
            train_prophet = 

            metrics = self._evaluate_model(model, X_test, y_test)

            future_forecast = model.predict(self.horizon)

            self._log_artifacts(model, y_test, future_forecast)

            return {
                'model': model,
                'metrics': metrics,
                'forecast': future_forecast
            }
        
    def _temporal_split(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split time series into train/test
        """
        test_size = self.config['data']['test_size']
        split_idx = int(len(y) * (1 - test_size))

        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if X is not None and not X.empty:
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        else:
            X_train, X_test = pd.DataFrame(), pd.DataFrame()

        return X_train, X_test, y_train, y_test
    
    def _evaluate_model(self, model, y_test: pd.Series, X_test: pd.DataFrame = None) -> dict:
        """
        Evaluate model performance on test set
        """
        
