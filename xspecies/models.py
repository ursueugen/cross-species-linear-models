"""
Classes:
 - BaseModel
 - LinearRegressionModel
"""

import warnings
from typing import List
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (mean_squared_error, 
    mean_absolute_error, median_absolute_error, r2_score)
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf

from xspecies.interfaces import Model


class BaseModel(Model):
    
    def __init__(self):
        raise NotImplemented("Base Model can't be "
                            "instantiated.")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> dict:
        pred = self.model.predict(X)
        scores = BaseModel.get_score_dict(y, pred)
        return scores
    
    def get_score_dict(y_true, y_pred) -> dict:
        scores = {}
        
        def compute_dist(y_true, y_pred, dist):
            # TODO
            try:
                mask = ~y_true.isna()
                if mask.sum() > 5:
                    return dist(y_true[mask], y_pred[mask])
                else:
                    warnings.warn("Unefficient way of doing it.")
                    raise ValueError
            except ValueError:
                return np.nan
        
        scores['mse'] = compute_dist(y_true, y_pred, mean_squared_error)
        scores['mean_ae'] = compute_dist(y_true, y_pred, mean_absolute_error)
        scores['median_ae'] = compute_dist(y_true, y_pred, median_absolute_error)
        scores['r2'] = compute_dist(y_true, y_pred, r2_score)
        
        return scores
    
    def get_score_none() -> dict:
        scores = {}
        scores['mse'] = None
        scores['mean_ae'] = None
        scores['median_ae'] = None
        scores['r2'] = None
        return scores
        
    def get_features_weights(self) -> dict:
        raise NotImplemented("Base Model doesn't have "
                            "model-specific methods.")
    
    def get_features_signs(self) -> dict:
        raise NotImplemented("Base Model doesn't have "
                            "model-specific methods.")
    
    def get_features_ranks(self) -> dict:
        raise NotImplemented("Base Model doesn't have "
                            "model-specific methods.")


class LinearRegressionModel(BaseModel):
    """
    An univariate linear regression model. Works as a model
     that predicts the target variable from a single feature
     to be weighted and other features that are considered as
     control variables, which are not weighted.
    
    E.g.: Lifespan ~ ExpressionGene_i + Mass + Temperature
     The model weights ExpressionGene_i, but doesn't weight
     the Mass and Temperature variables. They are used in the
     model for the purpose of control.
    
    Attributes:
        static RANK_METHOD: str
        formula: str
        target: str
        control_vars: List[str]
        features_models: dict
    
    Methods:
        __init__(formula)
        fit(X, y)
        score(X, y)
        generate_predictive_features() -> generator
        get_features_weights() -> pd.Series
        get_features_signs() -> pd.Series
        get_features_ranks() -> pd.Series
        get_features_extra() -> dict
        
    """
    
    # aka "competition" ranking
    RANK_METHOD = "min"
    
    def __init__(self, formula: str):
        self.formula = formula


    def fit(self, X: pd.DataFrame, y: pd.Series, 
            control_vars=None):
        
        if (control_vars is not None 
            and not set(control_vars).issubset(X.columns)):
            
            if type(control_vars) != list:
                raise ValueError("control vars is not list.")
            raise KeyError("Some of the control variables are"
                          " not found in the data columns.")
        
        data = pd.merge(X, y, how='inner',
                    left_index=True, right_index=True)
        
        features_models = {}
        
        features_iter = (
            LinearRegressionModel.generate_predictive_features(
                data, y.name, control_vars) )
            
        for feat in features_iter:
            
            formula = self.formula.format(feature=feat)
            
            try:
                model = smf.ols(formula=formula,
                             data=data, missing='drop')
            
                results = model.fit()
            
                features_models[feat] = {"model": model,
                                        "results": results}
            except ValueError:
                features_models[feat] = {"model": None,
                                        "results": None}
        
        self.features = list(features_models.keys())
        self.target = y.name
        self.control_vars = control_vars
        self.features_models = features_models


    def score(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Score the model for each feature. The computed scores are
         from the BaseModel. If custom scores are required, they
         should be implemented in a method 
             get_score_dict(y_true, y_pred).
        """
        
        data = pd.merge(X, y, how='inner',
                    left_index=True, right_index=True)
        
        scores = {}
        
        for feat in self.features:
            
            model_res = self.features_models[feat]["results"]
            
            if model_res is not None:
                y_true = data[feat]
                y_pred = model_res.predict(data)
                scores[feat] = BaseModel.get_score_dict(y_true, y_pred)
            else:
                scores[feat] = BaseModel.get_score_none()
            
        return scores
    
    
    def generate_predictive_features(df: pd.DataFrame, target: str,
                            control_vars: None or List[str] = None):
        """
        Generate the predictive features names. Predictive features
         are those that are not target or control_vars.
        """
        
        # Working with statsmodels formula API requires having both
        #  predictive features and target vars in single dataframe.
        #  We also control for variables that are not to be weighted,
        #  these are called control_vars, and have to be skipped.
        
        for col in df.columns:
            
            if (control_vars is not None
                and (col in control_vars)):
                continue
            elif col == target:
                continue
            else:
                yield col
        
        
    def get_features_weights(self) -> pd.Series:
        """
        Returns dictionary of features with the weight in (0,1) as
         value.
        """
        
        coefs_abs = abs(self._get_target_coef())
        weights = coefs_abs / coefs_abs.sum()  # normalized weights
        return weights.to_dict()


    def get_features_signs(self) -> pd.Series:
        """
        Returns a Series with signs corresponding to each feature.
        """
        
        coefs = self._get_target_coef()
        signs = pd.Series(data=np.sign(coefs),
                         index=self.features)
        return signs.to_dict()

    
    def get_features_ranks(self) -> pd.Series:
        """
        Returns Series of ranks. Ties are breaked by the
         method specified in the class' static variable RANK_METHOD.
        
        For more details check:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
        """
        
        coefs_abs = abs(self._get_target_coef())
        ranks_array = rankdata(coefs_abs, 
                        method=LinearRegressionModel.RANK_METHOD)
        ranks = pd.Series(data=ranks_array, index=self.features)
        return ranks.to_dict()
    
    
    def _get_target_coef(self) -> pd.Series:
        """
        Returns a Series with coefficients from the linear
         regression corresponding to the target variable from
         the models corresponding to each feature.
        """
        
        # TODO
        def coef_extractor(f):
            if self.features_models[f]['results'] is None:
                # 0 is reasonable choice. Doesn't influence weight normalization,
                #  does influence ranking, but can be managed. is_none flag is
                #  added in extra to later filter out such cases.
                return 0
            else:
                return self.features_models[f]['results'].params[self.target]
        
        coefs = pd.Series(data=map(
            coef_extractor, self.features),
                         index=self.features)

        return coefs


    def get_features_extra(self) -> pd.DataFrame:
        """
        Get extra features results, that are specific to the model,
         e.g. p-values.
        """
        
        d = {}
        for feat in self.features:
            
            res = self.features_models[feat]['results']
            
            if res is not None:
                # Get cooks distance to look at outliers
                # A cooks distance ratio > 3 may indicate an outlier.
                infl = res.get_influence()
                cooks = infl.cooks_distance[0]  # check with inf.summary
                cooks_ratios = np.sort(cooks / np.mean(cooks))
                
                if len(cooks_ratios) == 1:
                    cooks_ratios = [np.nan, cooks_ratios[-1]]
                
                d[feat] = {
                    "is_none": False,
                    "pval": res.pvalues[self.target],
                    "n_obs": res.nobs,
                    "r2_adj": res.rsquared_adj,
                    "cooks_dist_ratio_1st": cooks_ratios[-1],
                    "cooks_dist_ratio_2nd": cooks_ratios[-2],
                }
            
            else:
                d[feat] = {
                    "is_none": True,
                    "pval": None,
                    "n_obs": None,
                    "r2_adj": None,
                    "cooks_dist_ratio_1st": None,
                    "cooks_dist_ratio_2nd": None,
                }

        return d