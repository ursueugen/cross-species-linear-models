import pdb
import warnings
import itertools
from pathlib import Path
import sys
# sys.path.insert(0, ".")  # include path to xspecies module


import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt 
from venn import venn
from upsetplot import from_contents, UpSet


from xspecies.feature_analyzer import FeatureAnalyzerResults
from xspecies.utils import adjust_pval_series, GeneConverter


def load_feature_results(name_template, 
                         path, 
                         organs, 
                         include_humans,
                         targets) -> dict:
    res = {}
    for organ in organs:
        res[organ] = {}
        for include_hum in include_humans:
            res[organ][include_hum] = {}
            for target in targets:
                res[organ][include_hum][target] = {}
    
    iterable = itertools.product(
        organs, include_humans, targets)
    
    for organ, include_hum, target in iterable:
        name = name_template.format(
            organ=organ, include_human=include_hum, 
            target=target)
        dir_path = path / name
        res[organ][include_hum][target] = (
            FeatureAnalyzerResults.load(dir_path) )
    
    return res


def build_dataframe_from_analyzerResuls(analyzer_res) -> (
    pd.DataFrame):
    """
    Builds a custom dataframe of interest from an 
        FeatureAnalysisResults object.
    """

    df_extra = pd.DataFrame.from_dict(
        analyzer_res.extra, orient='index')[
        ['pval', 'r2_adj', 'n_obs']
    ]
    
    series_signs = pd.DataFrame.from_dict(analyzer_res.signs,
                       orient='index', columns=['sign'])
    
    df = pd.concat((df_extra, series_signs), axis=1)
    return df


def build_dataframe(res: dict,     
                    organs,
                    include_humans,
                    targets) -> pd.DataFrame:
    """
    Build the dataframe with model results.
    """
    
    iterable = itertools.product(
        organs, include_humans)
    
    dfs = []
    for (organ, include_human) in iterable:
        
        df = None
        for target in targets:

            analyzer_res = res[organ][include_human][target]
            df_i = build_dataframe_from_analyzerResuls(analyzer_res)
            df_i.columns = target + "_" + df_i.columns
            
            if df is None:
                df = df_i.copy()
            else:
                df = pd.merge(df, df_i, left_index=True,
                             right_index=True, how='inner')
        
        df.insert(loc=0, column='organ',
                  value=organ)
        df.insert(loc=1, column='human_samples',
                  value=include_human)
        
        dfs.append(df)

    df_final = pd.concat(dfs, axis=0)
    return df_final


def adjust_pvalues(df, strict, targets):
    """
    Replaces pvalues columns with adjusted pvalues.
    It does it not the most conservative way. It adjusts
     groups of p-values by organ:sample-type group, rather
     than taking all the generated p-values and adjusting.
    """
    
    if not strict:
        dfs = []
        for name, df_g_ in df.groupby(by=['organ', 'human_samples']):
            organ, include_human = name

            df_group = df_g_.copy()
            for target in targets:

                col_pval = target + "_" + "pval"
                pvals_adj = adjust_pval_series(
                    df_group.loc[:, col_pval].copy().fillna(1.0)
                )

                df_group[col_pval] = pvals_adj

                df_group.rename(inplace=True, columns={
                    col_pval: target + "_" + "adjpval"
                })

            dfs.append(df_group)

        return pd.concat(dfs, axis=0)
    
    else:
        # Strict 1
        df_ = df.copy()
        for target in targets:
            try:
                col_pval = target + "_" + "pval"
                pvals_adj = adjust_pval_series(
                    df_.loc[:, col_pval].copy().fillna(1.0)
                )
                df_[target + '_' + 'adjpval'] = pvals_adj
                
#                 df_[col_pval] = pvals_adj

#                 df_.rename(inplace=True, columns={
#                     col_pval: target + "_" + "adjpval"
#                 })

            except:
                import pdb
                pdb.set_trace()

        return df_


def add_indicators_of_unique_association(df, alpha, targets) -> pd.DataFrame:
    """
    Adds a column for each var in targets that is True
        iff the gene is associated with var and NOT associated
        with the other variables in targets.
        Otherwise, sets False.
    """
    
    for var in targets:
        mask_sign_var = df[var + "_adjpval"] <= alpha
        
        mask_unsign_other_vars = None
        for other_var in targets:
            if other_var == var:
                continue
            if mask_unsign_other_vars is None:
                mask_unsign_other_vars = (df[other_var + "_adjpval"]
                                          >= alpha)
            else:
                mask_unsign_other_vars = (mask_unsign_other_vars 
                                          & 
                                          (df[other_var + "_adjpval"] 
                                              >= alpha))
        
        mask = (mask_sign_var & mask_unsign_other_vars)
        
        df[var + "_uniquely_associated"] = mask
    
    return df


def add_gene_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns of gene names at column_index = 0.
    Assumes index is gene ids.
    """
    
#     if 'symbol' not in df.columns:
    gene_ids = df.index.to_list()

    conv = GeneConverter()
    gene_symbols = conv.convert_to_symbols(gene_ids)

    df.insert(loc=0, column='symbol', value=gene_symbols)

    return df


def add_genesets_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add genesets names. Used for genesets analysis.
    """
#     if 'geneset' not in df.columns:
    genesets = df.index.map(lambda s: s.split("_")[0])
    df.insert(loc=0, column="geneset", value=genesets)
    return df
    

def filter_sign_lifespan_association(df: pd.DataFrame, alpha) -> (
    pd.DataFrame):
    """
    Filter the maxlifespan column to keep only pvals < alpha
    """
    df_filt = df.copy()
    col_maxls_pval = "maxlifespan_adjpval"
    df_filt = df_filt.loc[df_filt[col_maxls_pval] < alpha, :].copy()
    return df_filt


def get_significant_genes_per_organ(df,
                    to_include_humans, 
                    var_name, alpha) -> dict:
    
    sets = dict()
    for name, df_g in df.groupby(by=['organ', 'human_samples']):

        organ, include_humans = name
        
        if not to_include_humans:
            if include_humans == 'withHumans':
                continue
        else:
            raise NotImplementedError()
        
        col = var_name + '_adjpval'
        mask_sign = df_g[col] < alpha
        sets[organ] = set( df_g.index[mask_sign].to_list() )
    
    return sets


def get_uniquely_associated_genes_per_organ(
                df, to_include_humans,
                var_name, alpha) -> dict:
    """
    Get sets of genes per organ that are associated only with
        the var_name and not associated with the other vars.
    """
    
    col_uniq_indicator = var_name + "_uniquely_associated"
    mask_uniq = df[col_uniq_indicator]
    df_uniq = df.loc[mask_uniq].copy()
    
    sets = get_significant_genes_per_organ(
        df_uniq, to_include_humans=to_include_humans, 
        var_name=var_name, alpha=alpha)
    
    return sets


def plot_venn(sets, path):
    
    if len(sets) > 1:
        fig = venn(sets).get_figure()
        fig.savefig(path)
    elif len(sets) in {0, 1}:
        print(f'plot_venn: No sets to intersect for {path}')


def plot_upset(sets, path):
    
    if len(sets) > 1:
        df_upset = from_contents(sets)
        upset_plot = UpSet(df_upset, sort_by='degree',
             sort_categories_by='cardinality',
             show_counts=True, show_percentages=True)
        fig = plt.figure()
        upset_plot.plot(fig=fig)
        fig.savefig(path)
    elif len(sets) in {0, 1}:
        print(f'plot_upset: No sets to intersect for {path}')


def produce_plots(df, dir_path: Path, alpha, targets):
    
    plots_dir = dir_path / "plots/"
    if not plots_dir.is_dir():
        plots_dir.mkdir()
    
    for var in targets:
        gene_sets_by_organ = get_significant_genes_per_organ(
            df,
            to_include_humans=False,
            var_name=var, alpha=alpha)
        
        gene_sets_by_organ_uniquely_associated = (
            get_uniquely_associated_genes_per_organ(
                df, to_include_humans=False,
                var_name=var, alpha=alpha)
        )

        
        try:
            plot_venn(gene_sets_by_organ, plots_dir 
                      / ("byorgan_" + var + "_venn.png"))
            plot_upset(gene_sets_by_organ, plots_dir 
                       / ("byorgan_" + var + "_upset.png"))
        except ZeroDivisionError:
            pass

        try:
            plot_venn(gene_sets_by_organ_uniquely_associated, plots_dir 
                      / ("unique_byorgan_" + var + "_venn.png"))
            plot_upset(gene_sets_by_organ_uniquely_associated, plots_dir 
                       / ("unique_byorgan_" + var + "_upset.png"))
        except ValueError:
            pass
        except TypeError:
            pass
        except ZeroDivisionError:
            pass


def save_table(df, path_to_dir, name_template):
    
    byorgan_dir = path_to_dir / "tables_by_organ/"
    if not byorgan_dir.is_dir():
        byorgan_dir.mkdir()
    
    for organ, df_g in df.groupby(by="organ"):
        fpath = byorgan_dir / name_template.format(organ)
        df_g.to_csv(fpath, header=True, index=True)

        
def process_df(df, feature_type, alpha, targets):
    
    if feature_type == 'GENES':
        # TODO: rm?
        # Keep only non-human samples
        warnings.warn("Keeping non-human samples", UserWarning)
        mask_no_humans = df['human_samples'] ==  'withoutHumans'
        df = df.loc[mask_no_humans].copy()

        df = adjust_pvalues(df, strict=True, targets=targets)
        df = add_indicators_of_unique_association(df, alpha=alpha, targets=targets)
        df = add_gene_symbols(df)
        return df
    else:
        df = add_genesets_name(df)
        df = adjust_pvalues(df, strict=True, targets=targets)
        df = add_indicators_of_unique_association(df, alpha=alpha, targets=targets)
        return df


def execute_extract_results(input_dir, results_dir, alpha, feature_type,
                organs, include_humans, targets, name_template_callback=None):
    
    assert feature_type in {'genes', 'genesets'}
    
    results_dir.mkdir(exist_ok=True)
    
    if name_template_callback is not None:
        name_template = name_template_callback(feature_type)
    else:
        if feature_type == 'genes':
            name_template = "Xspecies_LinearRegression_Genes_{organ}_{include_humans}_{target}"
        else:
            name_template = "Xspecies_LinearRegression_Genesets_{organ}_{include_humans}_{target}"
    
    
    res_dict = load_feature_results(name_template, 
                                    input_dir, 
                                    organs, 
                                    include_humans,
                                    targets)

    df = build_dataframe(res_dict,
                        organs,
                         include_humans,
                         targets)

    df = process_df(df, feature_type, alpha, targets)
        
    R2_THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5]
    for r2_th in R2_THRESHOLDS:
            
        res_dir = results_dir / ("R2_THRESHOLD_{}").format(int(r2_th*100))
            
        if not res_dir.is_dir():
            res_dir.mkdir()
            
        mask_ = df['maxlifespan_r2_adj'] >= r2_th
        df_ = df.loc[mask_].copy()
            
        save_table(df_, res_dir, "models_on_{}.csv")
        df_.to_csv(
            res_dir / "linear_models_on_species_vars.csv", 
            header=True, index=True
        )
        produce_plots(df_, res_dir, alpha=alpha, targets=targets)


# execute_extract_results(FEATURE_ANALYSIS_DIR, RESULTS_DIR, ALPHA, FEATURE_TYPE, 
# ORGANS, INCLUDE_HUMANS, TARGETS)
