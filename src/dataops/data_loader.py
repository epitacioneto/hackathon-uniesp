import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict

class DataLoader(ABC):
    """
    Abstract base class for data loaders
    """
    @abstractmethod
    def load_data(self) -> Dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def save_data(self, data: pd.DataFrame, path: Optional[str] = None) -> None:
        pass

class CSVDataLoader(DataLoader):
    """
    Implementation of CSV data loading
    """

    def __init__(self, config: dict):
        self.raw_path = config['data']['raw_paths']
        self.processed_path = config['data']['processed_path']

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        """
        data = {}
        for name, path in self.raw_path.items():
            data[name] = pd.read_csv(path, sep=';')
        return data
    
    def save_data(self, data: pd.DataFrame, path: Optional[str] = None) -> None:
        """
        Save processed data to parquet
        """
        save_path = path or self.processed_path
        data.to_csv(save_path, index=False)

class ParquetDataLoader(DataLoader):

    def __init__(self, config: dict):
        self.processed_path = config['data']['processed_path']

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        """
        return pd.read_parquet(self.processed_path, sep=';')
    