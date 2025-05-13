import mlflow
from ..logger import logging
import pandas as pd
from ..dataops.data_loader import CSVDataLoader
from ..dataops.data_preprocessor import DataPreprocessor
from ..dataops.data_quality import DataQuality
# Forecasting
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.boxcox import LogTransformer
from ..exception import CustomException
from datetime import datetime
# Model versioning
import mlflow
from sktime.utils import mlflow_sktime
import sys
from utils import plot_actual_vs_predicted, plot_vendor_forecast, plot_forecast_vs_goal, plot_vendor_forecasts, plot_vendor_histories
from mlflow.models.signature import infer_signature

class ForecastingPipeline:
    """
    Pipeline for time series forecasting
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_loader = CSVDataLoader(config)
        self.preprocessor = DataPreprocessor(config)
        self.data_quality = DataQuality(config)
        self.horizon = config['forecasting']['horizon']
        self.metrics = config['forecasting']['metrics']
        self.target = config['forecasting']['target']
        self.coverage = config['quality']['coverage']
        self.p_val = config['quality']['p_val']
        self.window_size = config['quality']['window_size']
        self.pyfunc_predict_conf = {
            "predict_method": {
                "predict": {},
                "predict_interval": {"coverage": self.coverage},
            }
        }
        self.params = {
            'seasonality_mode':'additive',
            'yearly_seasonality':True,
            'weekly_seasonality':True,
            'daily_seasonality':False
        }
    
    def run(self):
        """
        Execute forecasting pipeline
        """
        logging.info('Loading our raw data')
        raw_data = self.data_loader.load_data()
        logging.info('Preprocessing our raw data')
        processed_data, goals_data = self.preprocessor.preprocess(raw_data)
        logging.info('Data cleaning done, saving our processed data')
        self.data_loader.save_data(processed_data)
        vendor_codes = processed_data.index.get_level_values(0).unique()
        
        for vendor in vendor_codes:
            logging.info(f'Starting forecasting for vendor {vendor}')
            with mlflow.start_run(run_name=f"forecasting_vendor_{vendor}", nested=True):
                vendor_data = processed_data.xs(vendor, level=0)
                y = vendor_data['valorVenda']

                try:
                    fh = ForecastingHorizon(pd.date_range(y.index[-1],
                                            periods=365,
                                            freq='D')[1:],
                                            is_relative=False)
                    
                    model = LogTransformer() * Prophet(**self.params)
                    model.fit(y)

                    mlflow.log_params(self.params)
                    mlflow.set_tag("Model Info", f"Vendor forecasting for {datetime.now()}")

                    pred_intervals = model.predict_interval(fh, coverage=self.coverage)
                    
                    signature = infer_signature(y, pred_intervals)
                    model.pyfunc_predict_conf = self.pyfunc_predict_conf
                    
                    logging.info(f'{vendor}: Logging their model to MLFlow')
                    mlflow_sktime.log_model(model, artifact_path="model", signature=signature)

                    lower = pred_intervals.loc[:, ([self.target], [self.coverage], ['lower'])]
                    upper = pred_intervals.loc[:, ([self.target], [self.coverage], ['upper'])]

                    if isinstance(lower, pd.DataFrame):
                        lower = lower.iloc[:, 0]
                    if isinstance(upper, pd.DataFrame):
                        upper = upper.iloc[:, 0]

                    y_pred = model.predict(fh)

                    forecast = {'vendor_id': vendor, 'forecast': y_pred, 'lower_ci': lower, 'upper_ci': upper}
                    
                    self._run_plots(vendor, y, forecast)
                    mlflow.log_artifact(f"../artifacts/forecast_{vendor}.png", artifact_path="model/artifacts")

                except Exception as e:
                    raise CustomException(e, sys)

        
    def _run_plots(self, vendor, y, forecast):
        plot_vendor_forecast(vendor, y, forecast)
        pass
