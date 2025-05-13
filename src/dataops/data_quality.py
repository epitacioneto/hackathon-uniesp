import logging
import numpy as np
from alibi_detect.cd import KSDrift
from ..exception import CustomException
import sys

class DataQuality:
    def __init__(self, config):
        self.config = config

    def drift_detector(self, historical_data):
        try:
            X_ref = historical_data.values.reshape(-1, 1)

            drift_detector = KSDrift(
                x_ref = X_ref,
                p_val = self.config['quality']['p_val']
            )

            return drift_detector
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def validate_forecast(self, y_pred, lower, upper, vendor_id):
        try:
            checks = {
                "No NaN values": lambda: not y_pred.isnull().any(),
                "No infinite values": lambda: not np.isinf(y_pred).any(),
                "Positive values": lambda: y_pred.min() >= 0,
                "Valid CIs": lambda: (upper >= lower).all(),
                "Non-empty": lambda: len(y_pred) > 0
            }

            failed = [name for name, check in checks.items() if not check()]
            if failed:
                logging.info(f"Validation failed for {vendor_id}: {', '.join(failed)}")
                return False
            return True
        except Exception as e:
            raise CustomException(e, sys)