import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

class DataLoader(ABC):
    """
    Abstract base class for data loaders
    """
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_data(self, data: pd.DataFrame, path: str) -> None:
        pass

class CSVDataLoader(DataLoader):
    """
    Implementation of CSV data loading
    """

    def __init__(self, config: dict):
        self.raw_path = config['data']['raw_path']
        self.processed_path = config['data']['processed_path']

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        """
        return pd.read_csv(self.raw_path, parse_dates=['date'])
    
    def save_data(self, data: pd.DataFrame, path: Optional[str] = None) -> None:
        """
        Save processed data to parquet
        """
        save_path = path or self.processed_path
        data.to_parquet(save_path, index=False)