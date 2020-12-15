import itertools
from pathlib import Path
import sys
# sys.path.insert(0, ".")  # include path to xspecies module

import click

import numpy as np
import pandas as pd
import seaborn as sns

from xspecies.dataset import DatasetBuilder, Dataset
from xspecies.feature_analyzer_components import (BaseFilter,
                            BaseSplitter, BasePreprocessor)
from xspecies.models import LinearRegressionModel
from xspecies.feature_analyzer import FeatureAnalyzer

from utils_extract_results import execute_extract_results


# CLI Interface
@click.group()
@click.option('--type_input', '-t', 
              type=click.Choice(['genes', 'genesets'], case_sensitive=False))
@click.pass_context
def linearmodels(ctx, type_input):
    ctx.ensure_object(dict)
    ctx.obj['type_input'] = type_input


@linearmodels.command()
@click.argument('input', 
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output', 
                type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.pass_context
def analyze_models(ctx, input, output):
    """Fit and save model data for every organ, target variable and human inclusion selection."""
    
    type_input = ctx.obj['type_input']
    
    click.echo(f"Input dataset: {input}, Output directory: {output}, Input type: {type_input}")
    
    param_dict = configure_analysis(input, output, type_input, command='analyze_models')
    
    if type_input == 'genes':
        log2_transform_features = True
    elif type_input == 'genesets':
        log2_transform_features = False
    
    analysis_results = run_analysis(
        analysis_dir=param_dict['output_path'],
        name_template=param_dict['name_template'], 
        dataset=param_dict['dataset'],
        organs=param_dict['organs'],
        include_humans=param_dict['include_humans'],
        targets=param_dict['targets'],
        with_log2_features=log2_transform_features
    )


@linearmodels.command()
@click.argument('input', 
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output', 
                type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.pass_context
def extract_results(ctx, input, output):  # input output
    """Compile results from fitted model data obtained through analyze-models."""
    type_input = ctx.obj['type_input']
    
    click.echo(f"Input dataset: {input}, Output directory: {output}, Input type: {type_input}")
    
    param_dict = configure_analysis(input, output, type_input, command='extract_results')
    
    input_path = Path(input)
    output_path = Path(output)
    
    execute_extract_results(
        input_dir=input_path,
        results_dir=output_path,
        alpha=param_dict['alpha'],
        feature_type=type_input,
        organs=param_dict['organs'],
        include_humans=param_dict['include_humans'],
        targets=param_dict['targets'],
        name_template_callback=get_name_template
    )

    
def configure_analysis(input_dir: str, output_dir: str, type_input: str,
                      command: str) -> dict:
    
    assert command in {'analyze_models', 'extract_results'}
    
    o_path = Path(output_dir)
    if not o_path.is_dir():
        o_path.mkdir()
    
    template = get_name_template(type_input)
    
    param_dict = {
        'output_path': o_path,
        'organs': ["Brain", "Heart", "Lung", "Liver", "Kidney"],
        'include_humans': ['withHumans', 'withoutHumans'],
        'targets': ['maxlifespan', 'mass', 'temperature',
                    'metabolicRate', 'gestation', 'mtGC'],
        'name_template': template,
        'alpha': 0.05
    }

    if command == 'analyze_models':

        xspecies_ds = get_dataset(input_dir)
        add_duplicate_cols(xspecies_ds)    
        param_dict.update({'dataset': xspecies_ds})

    return param_dict


def get_name_template(type_input: str) -> str:
    assert type_input in {'genes', 'genesets'}
    if type_input == 'genes':
        return "Xspecies_LinearRegression_Genes_{organ}_{include_human}_{target}"
    else:
        return "Xspecies_LinearRegression_Genesets_{organ}_{include_human}_{target}"


def get_dataset(input_dir):
    path = Path(input_dir)
    dataset_name = path.name
    datasets_dir = path.parent
    xspecies_ds = Dataset.load(dataset_name, datasets_dir)
    return xspecies_ds


def add_duplicate_cols(ds: 'Dataset'):
    # adds convenient column names
    def add_duplicate_col(name_dup, colname):
        ds.samples_meta[name_dup] = ds.samples_meta[colname]
    add_duplicate_col('maxlifespan', 'Maximum longevity (yrs)')
    add_duplicate_col('mass', 'Body mass (g)')
    add_duplicate_col('temperature', 'Temperature (K)')
    add_duplicate_col('metabolicRate', 'Metabolic rate (W)')
    add_duplicate_col('gestation', 'Gestation/Incubation (days)')


def create_filter(organ: str, 
                  include_humans: str):
    
    FILTER_CONFIG = {
        "samples": {
            "tissue": ("=", organ),
            "Class": ("=", "Mammalia")
        },
    }


    samples_index_operation = {"organism": ("!=", "Homo_sapiens")}
    if include_humans == 'withoutHumans':
        FILTER_CONFIG["samples"].update(samples_index_operation)
        
    return BaseFilter(FILTER_CONFIG)


def create_preprocessor(with_log2_features=True):
    
    PREPROCESSOR_CONFIG = {
        "samples_transforms": [("maxlifespan", "log2"),
                               ("maxlifespan", "standardize")
                              ],
    }

    if with_log2_features:
        PREPROCESSOR_CONFIG["features_transforms"] = (
            ["log2", "standardize"]
        )
    else:
        PREPROCESSOR_CONFIG["features_transforms"] = (
            ["standardize"]
        )
    
    return BasePreprocessor(PREPROCESSOR_CONFIG)


def model_per_organ_and_target(analysis_dir,
                            dataset: 'Dataset',
                            name: str, 
                            formula: str,
                            organ: str, 
                            include_humans: str,
                            target: str, 
                            control_vars: list or None,
                            with_log2_features):
    """
    Run analysis on a selection of organ, human inclusion and target variable.
    """
    
    MODEL = LinearRegressionModel(formula)
    FILTER = create_filter(organ, include_humans)
    PREPROCESSOR = create_preprocessor(with_log2_features)
    
    feature_analyzer = FeatureAnalyzer(
        name=name, model=MODEL, filtor=FILTER,
        preprocessor=PREPROCESSOR,
        na_handler=None, 
        splitter=None)  #TODO: No split

    results = feature_analyzer.weight_features(
        dataset, target=target, control_vars=control_vars)
    
    results.save(analysis_dir)
    
    return results


def run_analysis(analysis_dir, 
                 name_template,
                 dataset,
                 organs,
                 include_humans,
                 targets,
                 with_log2_features):
    """
    Iterate through model options, run each model on appropriate data
     and save results.
    """

    formula_template = 'Q("{feature}") ~ 1 + {{{target}}}'
    
    results = {}
    for (organ, include_human, target) in itertools.product(organs, include_humans, targets):
        
        name = name_template.format(
            organ=organ, 
            include_human=include_human,
            target=target)
        
        print(name)
        
        if (analysis_dir / name).is_dir():
            continue

        formula = ("".join(formula_template.split(" ")[:-1])
                    + " {}".format(target) )

        results[name] = model_per_organ_and_target(
                analysis_dir,
                dataset,
                name, formula, organ, include_human,
                target=target,
                control_vars=None,
                with_log2_features=with_log2_features
        )
    
    return results


if __name__ == "__main__":
    linearmodels(obj={})
