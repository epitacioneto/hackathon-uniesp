from config.config import ConfigLoader
from pipelines.forecasting_pipeline import ForecastingPipeline
import mlflow

def main():
    
    config_loader = ConfigLoader()
    config = config_loader.get_config()

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    pipeline = ForecastingPipeline(config)
    results = pipeline.run()
        
    # Register best model
    if results["best_score"] > threshold:
        register_best_model(mlflow.active_run().info.run_id, "sales_forecast")

if __name__ == "__main__":
    main()