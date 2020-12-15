from typing import List
from pathlib import Path
import requests
from requests.exceptions import HTTPError
import json

import pandas as pd
from statsmodels.stats.multitest import multipletests


# TODO: obsolete
def load_one2one_orthologs(path="data/one_to_one_orthologs.txt") -> list:
    """
    Get the previously computed list of one-to-one orthologs in
    mammals.
    """
    with open(path, "r") as f:
        return f.read().split()


def check_and_get_subdirectory(base_dir: Path, subdir_name: str) -> Path:
    """
    Checks existance of base_dir, NON-existance of base_dir/subdir, and
     returns the Path(base_dir / subdir).
    
    Raises: 
        FileNotFoundError if the base_dir does not exist
        OSError if the base_dir and subdir both exist.
    
    Returns: path to an unexisting subdirectory subdir in the
             base directory base_dir.
    """
    
    if not base_dir.is_dir():
        raise FileNotFoundError("Directory {} not found"
                                .format(base_dir))
        
    subdir = base_dir / subdir_name
        
    if subdir.is_dir():
        raise OSError(
            f"Directory {subdir} exists and shouldn't be overwritten."
        )
    
    return subdir

def all_filepaths_exist(filepaths: List['Path']):
    """
    Returns True if all filepaths exist, otherwise returns False.
    """
    return all([f.is_file() for f in filepaths])


def save_json(d: dict, path: Path):
    """
    Saves dictionary d at path in json format.
    """
    with open(str(path), "w+") as f:
        json.dump(d, f)

def load_json(path: Path) -> dict:
    """
    Loads json file into dict.
    """
    with open(str(path), "r") as f:
        return json.load(f)


def adjust_pval_series(pval: pd.Series) -> pd.Series:
    """
    Returns a series of adjusted pvalues from a series of
        raw pvalues.
    """
    
    array = multipletests(pval, method='fdr_bh')[1]
    s = pd.Series(data=array, name=pval.name, index=pval.index)
    return s


class GeneConverter:
    """
    Converter between gene identifiers and names.
    
    Attributes:
        static LOOKUP_PATH: Path
        lookup: pd.DataFrame
    
    Methods:
        convert_to_symbols(ids: List[str])
        convert_to_ids(names: List[str])
        _are_all_in_lookup(ids: List[str], col: str)

    """
    
    LOOKUP_PATH = Path('data/ens2names_lookup.tsv')
    LOOKUP_COLUMN_NAMES = ['ensembl_id', 'name']
    
    
    def __init__(self):
        self.lookup = pd.read_csv(
            GeneConverter.LOOKUP_PATH, 
            sep='\t', header=0
        )
    
    
    def convert_to_symbols(self, ids: list) -> List[str]:
        """
        Converts a list of ensembl ids to names.
        Raises KeyError if a symbol not found in lookup.
        """
        
        if not self._are_all_in_lookup(ids, 'ensembl_id'):
            raise KeyError("Some ids not found.")
        
        symbols = (
            self.lookup.set_index("ensembl_id").loc[ids]["name"]
        ).to_list()
        
        return symbols
    
    
    def convert_to_ids(self, names: list) -> List[str]:
        """
        Converts a list of gene names/symbols to ensembl ids.
        Raises KeyError if a symbol not found in lookup.
        """
        
        if not self._are_all_in_lookup(names, 'name'):
            raise KeyError("Some symbols not found.")
        
        ids = (
            self.lookup.set_index("names").loc[names]["ensembl_id"]
        ).to_list()
        
        return ids
    
    
    def _are_all_in_lookup(self, values: List[str], col: str) -> bool:
        """
        Checks internally whether all queried values are in lookup.
        
        Args:
            values: List[str], values to be checked
            col: str, indicates column from lookup to check
        
        Raises ValueError if not invalid col as input.
        
        Returns True if all are found in col, otherwise False.
        """
        
        if col not in GeneConverter.LOOKUP_COLUMN_NAMES:
            raise ValueError("Invalid col argument: {}"
                            .format(col))
        
        mask_in_values = self.lookup[col].isin(values)
        records_found = (
            self.lookup.loc[mask_in_values][col].to_list()
        )     
    
        return set(records_found) == set(values)


# TODO: looks obsolete
def build_dataframe_with_tissue_species(ds, organs, one2one_orthologs):
    """
    Build a dataframe from Dataset with specified tissues and species.
    """

    mask_mammalia = ds.samples_meta['Class'] == 'Mammalia'
    mask_organs = ds.samples_meta['tissue'].isin(organs)
    
    df = pd.merge(ds.samples_meta[['tissue', 'Common name']]
                  .loc[mask_mammalia & mask_organs].copy(),
            ds.data[one2one_orthologs].copy(),
                  left_index=True, right_index=True,
            how='inner')
    
    return df


def read_gmt(path) -> dict:
    """
    Load gmt file representing gene sets into dict {geneset: gene symbols}.
    """
            
    d = {}
    with open(path, "r") as f:
        for line in f.readlines():
                    
            if line == '\n':
                continue
                    
            row = line.split("\t")
            geneset_name = row[0]
                    
            # row[1] is a link
            gene_symbols = row[2:]
            d[geneset_name] = [s for s in gene_symbols if s not in {'', '\n'}]

    return d