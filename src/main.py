from config.config import ConfigLoader
from src.pipelines.forecasting_pipeline import ForecastingPipeline
import mlflow
from exception import CustomException
import sys

def main():
    try:
        config_loader = ConfigLoader()
        config = config_loader.get_config()

        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        pipeline = ForecastingPipeline(config)
        pipeline.run()       

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()