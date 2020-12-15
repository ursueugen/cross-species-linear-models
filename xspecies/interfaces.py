import abc
import pandas as pd
    
class Model(metaclass=abc.ABCMeta):
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError
    
    @abc.abstractmethod
    def score(self, X: pd.DataFrame, y: pd.Series) -> dict:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_features_weights(self) -> dict:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_features_signs(self) -> dict:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_features_ranks(self) -> dict:
        raise NotImplementedError