import pandas as pd
from typing import Dict
from ..exception import CustomException
import sys
from ..logger import logging

class DataPreprocessor:
    """
    Handles data preprocessing tasks
    """

    def __init__(self, config: dict):
        self.config = config
        #self.feature_config = config['features']
    
    def preprocess(self, data):
        df, meta = self._clean_data(data)
        return df, meta
    def _clean_data(self, data):
        try:
            vendedores = data['raw_ped_vendedores']
            meta = data['raw_meta_anual']
            pedidos = data['raw_gprint_path']
            vendedores = vendedores.rename(columns={'idGPrint':'codVendedor'})
            meta['venda_valor'] = meta['venda_valor'].str.replace(',', '.').astype('float64')
            merged = pd.merge(pedidos, vendedores, on='codVendedor')
            df = merged[merged['status'] == 'ATIVO']
            df['dataHoraPrimeiroCadastro'] = pd.to_datetime(df['dataHoraPrimeiroCadastro'])
            df = df.set_index(['idUsuarioSIG', 'dataHoraPrimeiroCadastro']).sort_index()
            df['valorVenda'] = df['valorVenda'].str.replace(',', '.').astype('float64')

            return df, meta
        
        except Exception as e:
            raise CustomException(e, sys)