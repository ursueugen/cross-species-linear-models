"""
Classes:
 - BaseSplitter
 - BaseFilter
 - BasePreprocessor
 - Base_NA_Handler
"""

from typing import List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseSplitter:
    """
    Attributes:
        ratio: float in (0,1)
    
    Methods:
        split(Dataset) -> Dataset, Dataset
    """
    
    def __init__(self, test_train_ratio: float):
        
        if ((test_train_ratio >= 1) 
            or (test_train_ratio < 0)):
            raise ValueError("Test ratio has"
                            " to be in [0,1)")
        
        self.ratio = test_train_ratio
    
    def split(self, dataset: 'Dataset') -> ('Dataset', 'Dataset'):
        
        n_samples = len(dataset)
        test_size = int(self.ratio * n_samples)
        idxs_random = (np.random.choice(np.arange(0, n_samples-1, 1), 
                                     size=test_size, replace=False)
                       .tolist())
        
        test_dataset = dataset[idxs_random]
        train_dataset = dataset[[i for i in range(n_samples)
                                if i not in idxs_random]]
        
        return train_dataset, test_dataset


class BaseFilter:
    """
    Attributes:
        static:
            OPERATORS
            OPERANDS
        
        filter_ops
    
    Methods:
        static config_to_options
        apply
    """
    
    OPERATORS = {"<", ">", "=", "!=", "in"}
    OPERANDS = {"samples", "features"}
    
    def __init__(self, config: dict):
        
        options = BaseFilter.config_to_options(config)
        for opt in options:
            if ((len(opt) != 4) or (type(opt[1]) != str)
                or (opt[0] not in BaseFilter.OPERANDS)
                or (opt[2] not in BaseFilter.OPERATORS)):
                raise ValueError("Invalid filter option"
                                ": {}".format(opt))
        self.filter_ops = options
    
    
    def config_to_options(filter_config: dict) -> List[Tuple]:
        """
        Convert FILTER_CONFIG dict to list of tuples indicating
         filtering operations to be performed.
        An option is defined as a tuple:
         (category in {'index', 'samples'}, column, operator in OPERATORS, val - float)
        """

        options = []
        for cat, d in filter_config.items():
            for column, (operator, val) in d.items():
                opt = (cat, column, operator, val)
                options.append(opt)
        return options
    
    
    def apply(self, dataset: 'Dataset') -> 'Dataset':
        
        new_ds = dataset.copy()
        
        for operation in self.filter_ops:
            
            cat, col, operator, val = operation
            
            if cat == 'samples':
                if col == 'index':
                    col_val = new_ds.samples_meta.index
                else:
                    col_val = new_ds.get_samples_col(col)
            
            elif cat == 'features':
                if col == 'index':
                    col_val = new_ds.features_meta.index
                else:
                    col_val = new_ds.get_features_col(col)
            
            if operator == "<":
                mask = col_val < val
            elif operator == ">":
                mask = col_val > val
            elif operator == "=":
                mask = col_val == val
            elif operator == "!=":
                mask = col_val != val
            elif operator == "in":
                mask = col_val.isin(val)
            
            if col == 'index':
                ids = col_val[mask]
            else:
                ids = col_val.index[mask]
            
            if cat == 'samples':
                new_ds = new_ds.filter_samples(ids)
            elif cat == 'features':
                new_ds = new_ds.filter_features(ids)
        
        return new_ds


class BasePreprocessor:

    CONFIG_KEYS = {"features_transforms", "samples_transforms"}
    FEATURE_TRANSFORMS = {"log2", "standardize"}
    SAMPLE_TRANSFORMS = {"log2", "standardize"}
    LOG2 = lambda array: np.log2(array + 1)
    
    def __init__(self, config: dict):
        
        for key, val in config.items():
            if key not in BasePreprocessor.CONFIG_KEYS:
                raise ValueError("Invalid configuration option"
                                " in PREPROCESSOR_CONFIG"
                                ": {}".format((key, val)))
        
        for s in config['features_transforms']:
            if s not in BasePreprocessor.FEATURE_TRANSFORMS:
                raise ValueError("Invalid feature transformation"
                                ": {}".format(s))
        
        for s in config['samples_transforms']:
            if s[1] not in BasePreprocessor.SAMPLE_TRANSFORMS:
                raise ValueError("Invalid sample transformation"
                                ": {}".format(s))
        
        self.samples_transforms = config['samples_transforms']
        self.features_transforms = config['features_transforms']


    def preprocess(self, train_ds: 'Dataset',
                  test_ds: 'Dataset' or None = None) -> ('Dataset', 'Dataset'):
                
        train_ds_proc = train_ds.copy()
        
        if test_ds is not None:
            test_ds_proc = test_ds.copy()
        else:
            test_ds_proc = None
        
        train_ds_proc, test_ds_proc = (
            self._apply_features_transforms(
                train_ds_proc, test_ds_proc) )
        
        train_ds_proc, test_ds_proc = (
            self._apply_samples_transforms(
                train_ds_proc, test_ds_proc) )
        
        return train_ds_proc, test_ds_proc
    
    
    def _apply_features_transforms(self, 
                    train_ds: 'Dataset', test_ds: 'Dataset' or None = None) -> (
                    Tuple['Dataset', 'Dataset']):
        
        for transf in self.features_transforms:
            
            if transf == 'log2':
                
                train_ds.data.loc[:, :] = train_ds.data.apply(
                    BasePreprocessor.LOG2, axis='columns')
                if test_ds is not None:
                    test_ds.data.loc[:, :] = test_ds.data.apply(
                        BasePreprocessor.LOG2, axis='columns')
                
            elif transf == 'standardize':
                
                scaler = StandardScaler()
                train_ds.data.loc[:, :] = scaler.fit_transform(
                        train_ds.data)
                if test_ds is not None:
                    # Use scaler fitted on train to avoid train->test leakage
                    test_ds.data.loc[:, :] = scaler.transform(
                            test_ds.data)
            
            else:
                raise NotImplementedError(
                    "Only log2 and standardize transforms"
                    " are currently available."
            )
        
        return train_ds, test_ds
    
    
    def _apply_samples_transforms(self, 
                    train_ds: 'Dataset', test_ds: 'Dataset' or None = None) -> (
                    Tuple['Dataset', 'Dataset']):
        
        for (col, transf) in self.samples_transforms:
            
            if transf == 'log2':
                
                train_ds.samples_meta[[col]] = (
                    train_ds.samples_meta[[col]].apply(
                        BasePreprocessor.LOG2, axis='columns'))
                if test_ds is not None:
                    test_ds.samples_meta[[col]] = (
                        test_ds.samples_meta[[col]].apply(
                            BasePreprocessor.LOG2, axis='columns'))
            
            elif transf == 'standardize':
                
                scaler = StandardScaler()
                train_ds.samples_meta[[col]] = scaler.fit_transform(
                    train_ds.samples_meta[[col]])
                if test_ds is not None:
                    test_ds.samples_meta[[col]] = scaler.transform(
                        test_ds.samples_meta[[col]])
            
            else:
                raise NotImplementedError(
                    "Only log2 and standardize transforms"
                    " are currently available."
            )
        
        return train_ds, test_ds
    

# TODO
class Base_NA_Handler:
    
    PROPERTIES = {"samples_min_obs", "features_min_obs", "samples_cols_to_clear"}
    
    def __init__(self, config: dict):
        
        if set(config.keys()) != Base_NA_Handler.PROPERTIES:
            raise ValueError("Invalid config for Base_NA_Handler.")

        self._configure(config)
    
    def _configure(self, config: dict):
        self.samples_min_n_obs = config['samples_min_obs']
        self.features_min_n_obs = config['features_min_obs']
        self.samples_cols_to_clear = config['samples_cols_to_clear']
    
    
    def apply(self, dataset: 'Dataset') -> 'Dataset':
        
        samples_n_obs = ( ~dataset.data.isna() ).sum(axis='columns')
        features_n_obs = ( ~dataset.data.isna() ).sum(axis='index')
        
        samples_mask = samples_n_obs >= self.samples_min_n_obs
        features_mask = features_n_obs >= self.features_min_n_obs
        
        samples_to_keep = dataset.data.index[samples_mask].to_list()
        features_to_keep = dataset.data.columns[features_mask].to_list()
        
        new = dataset.filter_samples(samples_to_keep)
        new = new.filter_features(features_to_keep)
        
        for col in self.samples_cols_to_clear:
            samples_to_keep = new.samples_meta.index[
                ~new.samples_meta[col].isna()].to_list()
            new = new.filter_samples(samples_to_keep)
            
        return new