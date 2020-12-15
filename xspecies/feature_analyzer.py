"""
Classes:
 - FeatureAnalyzerResults
 - FeatureAnalyzer
"""

import time
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
import pandas as pd
from xspecies.feature_analyzer_components import BaseSplitter, BaseFilter, BasePreprocessor, Base_NA_Handler
from xspecies.models import BaseModel
from xspecies.utils import (check_and_get_subdirectory, 
                            all_filepaths_exist,
                            save_json, load_json)


class FeatureAnalyzer:
    """
    """
    
    def __init__(self, name, model, filtor, splitter, preprocessor, na_handler=None):
        
        # TODO: Add empty filter, splitter, preprocessor
        self.name = name
        self.filter = filtor
        self.splitter = splitter
        self.preprocessor = preprocessor
        self.model = model
        self.na_handler = na_handler
    
    def weight_features(self, dataset, target, control_vars=None):
        """
        Main method of the class. Performs the dataset upstream
         processing and uses the model to get features' weights.
        
        Args:
            dataset: Dataset
            
            target: str, column name in dataset.samples_meta
                    indicating the target variable.
                    
            control_vars: list, column names in dataset.samples_meta 
                          to control for in the model. Supported by
                          univariate feature selection models.
        """
        
        has_splitter = self.splitter is not None
        has_preprocessor = self.preprocessor is not None
        has_na_handler = self.na_handler is not None
        
        ds_filt = self.filter.apply(dataset)
        
        if has_na_handler:
            ds_filt = self.na_handler.apply(ds_filt)
        
        if ( has_splitter and has_preprocessor ):
            scores = self._fit_with_split_and_preprocessing(
                                        ds_filt, 
                                        target, control_vars)
            
        elif (not has_splitter):
            
            if has_preprocessor:
                ds_filt, _ = self.preprocessor.preprocess(ds_filt, None)
            
            X, y = (FeatureAnalyzer.get_data_for_model(
                ds_filt, target, control_vars))
            self.model.fit(X, y, control_vars)
            scores = self.model.score(X, y)
        
        else:
            raise NotImplementedError()
            
        
        weights = self.model.get_features_weights()
        signs = self.model.get_features_signs()
        ranks = self.model.get_features_ranks()
        extra = self.model.get_features_extra()
                
        res = FeatureAnalyzerResults(
            name=self.name, feature_names=list(weights.keys()),
                weights=weights, ranks=ranks, signs=signs,
                scores=scores, extra=extra)
        
        return res
    
    def _fit_with_split_and_preprocessing(self, dataset: 'Dataset', 
                                          target, control_vars) -> dict:
        """
        Fit datasets with model with splitting and preprocessing.
        
        Side-effects: Model gets fitted
        Returns: scores dict
        """
        
        assert (self.splitter is not None) and (self.preprocessor is not None)
        
        ds_train, ds_test = self.splitter.split(ds_filt)    
        ds_train_proc, ds_test_proc = (
            self.preprocessor.preprocess(ds_train, ds_test))
        
        X_train, y_train = (
            BaseFeatureAnalyzer.get_data_for_model(
                ds_train_proc, target, control_vars))
        
        X_test, y_test = (
            BaseFeatureAnalyzer.get_data_for_model(
                ds_test_proc, target, control_vars))       
        
        if control_vars:
            self.model.fit(X_train, y_train, control_vars)
        else:
            self.model.fit(X_train, y_train)
        
        
        score_train = self.model.score(X_train, y_train)
        score_test = self.model.score(X_test, y_test)
        scores = {"train": score_train, "test": score_test}
        return scores
    
    
    def get_data_for_model(dataset, target, control_vars=None):
        
        if target not in dataset.samples_meta.columns:
            raise KeyError("Target variable {} not found"
                          " in the dataset {}".format(
                          target, dataset.name))
        
        y = dataset.samples_meta[target].copy()
        
        if control_vars:
            X = pd.merge(dataset.data.copy(),
                        dataset.samples_meta[control_vars].copy(),
                        left_index=True,
                        right_index=True,
                        how='inner')

        else:
            X = dataset.data.copy()
        
        return X, y


class FeatureAnalyzerResults:
    """
    Class for storing, saving and loading results of 
     a single feature analysis.
     
    Attributes:
        static META_FILENAME: str
        static SCORES_FILENAME: str
        static DATAFRAME_FILENAME: str
        
        name: str
        features_names: List[str]
        weights: dict
        ranks: dict
        signs: dict
        score: dict
        extra: dict
    
    Methods:
        __init__(name, feature_names, weights, ranks, signs, scores, extra)
        __len__() -> int
        __eq__(other) -> bool
        copy() -> 'FeatureAnalyzerResults'
        check_repr_inv()
        to_dataframe() -> pd.DataFrame
        save(dir_path)
        static load(name, dir_path) -> 'FeatureAnalyzerResults'
        static from_dataframe(df) -> Tuple[dict]
        static generate_filepaths(path) -> Tuple['Path']
    """
    
    META_FILENAME = "meta.json"
    SCORES_FILENAME = "scores.json"
    DATAFRAME_FILENAME = "feature_analysis_results.csv"
    
    
    def __init__(self, name: str, feature_names: List[str],
                weights: dict, ranks: dict, signs: dict,
                scores: dict, extra: None or dict = None):
        """
        Initializes and checks representation.
        
        Raises AssertionError if representation violated.
        """
        self.name = name
        self.feature_names = feature_names
        self.weights = weights
        self.ranks = ranks
        self.signs = signs
        self.scores = scores
        
        if extra is not None:
            self.extra = extra
        else:
            self.extra = {f: [] for f in feature_names}  # empty extra
        
        self.check_repr_inv()
    
    def __len__(self):
        """Returns the number of features in results."""
        self.check_repr_inv()
        return len(self.weights)
    
    def __eq__(self, other: 'FeatureAnalyzerResults') -> bool:
        is_equal = (self.name == other.name 
                      and self.feature_names == other.feature_names
                      and self.weights == other.weights
                      and self.ranks == other.ranks
                      and self.signs == other.signs
                      and self.scores == other.scores
                      and self.extra == other.extra)
        return is_equal
    
    def copy(self) -> 'FeatureAnalyzerResults':
        # Deep copy
        self.check_repr_inv()
        new = FeatureAnalyzerResults(self.name, self.feature_names[:],
                self.weights.copy(), self.ranks.copy(), self.signs.copy(),
                self.scores.copy(), self.extra.copy())
        return new
    
    
    def check_repr_inv(self):
        """
        Checks the representation invariant:
            - Keys of weights, ranks, signs and extra have to be the same.
            - score is unconstrained. Depending on the model, it may store
                evaluation results of univariate models or multivariate,
                so it can't be restricted to have the same keys as the
                previously mentioned attributes.
        
        Raises AssertionError if representation invariant violated.
        """
        
        assert (set(self.weights.keys()) == set(self.ranks.keys())
                == set(self.signs.keys()) == set(self.extra.keys()) 
                and type(self.scores) == dict ), (
                "Representation invariant violated.")
            

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns dataframe with all descriptors of features.
         DataFrame structure: rows are indexed by feature (e.g. gene),
             columns are various results by feature, e.g. weight,
             sign, rank, R2, pval, etc.
        """
        
        self.check_repr_inv()
        
        to_df = lambda d, col: pd.DataFrame.from_dict(
            d, orient='index', columns=[col])
        
        cols = map(to_df, 
                   (self.weights, self.signs, self.ranks),
                   ('weights', 'signs', 'ranks')
                  )
        df = pd.concat(cols, axis='columns')
        return df


    def save(self, dir_path: Path, name: None or str = None):
        """
        Creates a directory at dir_path with name given by
         FeatureAnalyzerResults's name.
        
        Raises: OSError if dir_path does not exist or if a
                directory for the Result's names already exists.
        """
        
        if name:
            subdir_name = name
        else:
            subdir_name = self.name

        dir_this_results = check_and_get_subdirectory(
                            dir_path, subdir_name)
        
        dir_this_results.mkdir()
        
        meta_filepath, scores_filepath, dataframe_filepath = (
            FeatureAnalyzerResults.generate_filepaths(dir_this_results))
        
        meta = {'time': str(datetime.fromtimestamp(time.time())),
                'name': self.name,
                'feature_names': self.feature_names,
                'extra': self.extra}

        save_json(meta, meta_filepath)
        save_json(self.scores, scores_filepath)
        self.to_dataframe().to_csv(dataframe_filepath, 
                                   header=True, index=True)
        
    
    def load(dir_load: Path) -> 'FeatureAnalyzerResults':
        """
        Loads a FeatureAnalyzerResults from its directory.
        
        Raises: OSError if files required for loading are not found.
        """
        
        meta_fp, scores_fp, dataframe_fp = (
            FeatureAnalyzerResults.generate_filepaths(dir_load))
        
        files_reqd = [meta_fp, scores_fp, dataframe_fp]
        if not all_filepaths_exist(files_reqd):
            not_exist = list(
                filter(lambda fp: not fp.is_file(), files_reqd)
            )
            raise FileNotFoundError(
                f"Some of the required files do not exist: {not_exist}"
            )
        
        
        meta = load_json(meta_fp)
        scores = load_json(scores_fp)
        dataframe = pd.read_csv(dataframe_fp, header=0, index_col=0)
        
        weights, signs, ranks = (
            FeatureAnalyzerResults.from_dataframe(dataframe) )
        
        res = FeatureAnalyzerResults(name=meta['name'],
                feature_names=meta['feature_names'],
                weights=weights, signs=signs, ranks=ranks, 
                scores=scores, extra=meta['extra'])
        
        return res
    
    def from_dataframe(df: pd.DataFrame) -> Tuple[dict]:
        """
        Convert dataframe storing weights, signs and ranks into dicts.
        
        Raises KeyError if df.columns != ['weights', 'signs', 'ranks']
        """
        
        COLNAMES = ['weights', 'signs', 'ranks']
        if (df.columns != COLNAMES).all():
            raise KeyError("DataFrame with violated structure.")
        
        to_dict = lambda s: df[s].to_dict()
        weights, signs, ranks = tuple(map(to_dict, COLNAMES))
        
        return weights, signs, ranks
    
    def generate_filepaths(path: Path) -> Tuple['Path']:
        """
        Generates tuple of filepaths for meta, scores and dataframe.
        """

        create_fp = lambda s: path / s
        
        meta_filepath = create_fp(FeatureAnalyzerResults.META_FILENAME)
        scores_filepath = create_fp(FeatureAnalyzerResults.SCORES_FILENAME)
        dataframe_filepath = create_fp(
            FeatureAnalyzerResults.DATAFRAME_FILENAME)
        
        return meta_filepath, scores_filepath, dataframe_filepath