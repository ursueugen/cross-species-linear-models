"""
Classes for performing ssGSEA, a form of enrichment analysis.

Classes:
    GeneSetContainer - contains gene sets, as input to ssGSEA
    Enricher - computes enrichment scores (ES)
"""


import warnings
from typing import List
from pathlib import Path
import math
import pandas as pd
import gseapy as gp
from xspecies.dataset import Dataset
from xspecies.utils import GeneConverter, read_gmt


class GenesetContainer:
    """
    https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
    
    Methods:
        get_geneset_path(geneset: str) -> Path
        get_avaliable_genesets() -> List[str]
        get_geneset_dict() -> dict
    """
    
    def __init__(self):
        self._paths = {
            "kegg": Path("data/genesets/c2.cp.kegg.v7.1.symbols.gmt"),
            "kegg_updated": Path("data/genesets/Human_KEGG_June_01_2020_symbol_BADERLABS.gmt")
        }
    
    def get_available_genesets(self) -> List[str]:
        return list(self._paths.keys())
    
    def get_geneset_dict(self) -> dict:
        return self._paths.copy()
    
    def get_geneset_path(geneset: str) -> Path:
        """
        Get path to gene set.
        """
        path = self._paths.get(geneset, None)
        
        if path is None:
            raise KeyError("Invalid geneset name: {}"
                          .format(geneset))
        else:
            return path


class Enricher:
    """
    Class to perform ernichment.
    """
    
    ENRICHMENT_DIR_PATH = Path("data/enrichments/")
    
    def __init__(self, processes: int):

        self.sample_norm_method = "rank"
        self.processes = processes
        self.permutation_num = 0
        
        self.min_size = 0
        self.max_size = 2000
        self.scale = False
        self.no_plot = True
        self.verbose = True
    
    def ssgsea(self, name: str, df: pd.DataFrame,
               gene_sets: dict) -> pd.DataFrame:
        """
        Perform single sample GSEA.
        
        Args:
            name: str, name of analysis, used to create directory and store results.
            df: pd.DataFrame, the data
            gene_sets: dict, a dict of gene sets for which to perform the analysis.
                       Structure is (gene set name) -> Path(...)
        Returns: 
            DataFrame with all the samples NES and a column indicating the gene set name.
        """
        
        dfs = []
        for name_gs, val_gs in gene_sets.items():
            outdir_gs = Enricher.ENRICHMENT_DIR_PATH / name
            
            if not outdir_gs.is_dir():
                outdir_gs.mkdir()            
            
            if isinstance(val_gs, Path):
                geneset_dict = read_gmt(val_gs)
            else:
                geneset_dict = val_gs.copy()
            
            # Perform the enrichment in batches of size step_size
            genesets = list(geneset_dict.keys())
            step_size = 40
            dfs_geneset = []
            for i in range(0, len(genesets) + step_size, step_size):
                
                # Define genesets_batch
                if i == 0:
                    continue
                if i < len(genesets):
                    genesets_batch = genesets[i-step_size: i]
                else:
                    genesets_batch = genesets[i-step_size: len(genesets)]
                
                genesets_batch_dict = {geneset: geneset_dict[geneset]
                                       for geneset in genesets_batch}
                
                print(i-step_size, i, len(genesets_batch_dict))
                
                ss = gp.ssgsea(
                    data=df, 
                    gene_sets=genesets_batch_dict, 
                    outdir=str(outdir_gs / name_gs),
                    sample_norm_method=self.sample_norm_method,
                    min_size=self.min_size, max_size=self.max_size,
                    permutation_num=self.permutation_num,
                    scale=self.scale, processes=self.processes,
                    no_plot=self.no_plot,
                    verbose=self.verbose
                )
                
                df_res_batch = pd.DataFrame.from_dict(ss.resultsOnSamples, orient='index').transpose()
                dfs_geneset.append(df_res_batch)
                
            # Aggregate batches
            df_res = pd.concat(dfs_geneset, axis=0, sort=False)
            df_res['gene_set'] = name_gs
            
            df_res.to_csv(outdir_gs / (name_gs + ".csv"), header=True, index=True)
            dfs.append(df_res)

        # Aggregate the different categories of gene sets (e.g. KEGG, GO, etc.)
        df_agg = pd.concat(dfs, axis=0, sort=False)
        df_agg.to_csv(outdir_gs / "enrichment_results.csv", header=True, index=True)
        return df_agg


class EnrichmentDatasetBuilder:
    """
    Builds a Dataset which contains the enrichment scores (ES) for every sample.
    """

    def __init__(self, name: str, xspecies_ds: 'Dataset', paths_list: list):
        
        self.name = name
        warnings.warn("Hard-coded the EnrichmentDatasetBuilder.")
        self.xspecies_dataset = xspecies_ds.copy()
        self.genesets_results_paths = paths_list[:]

    
    def build_dataset(self) -> 'Dataset':
        
        genesets_enrichment, genesets_meta = (
            self._load_genesets()
        )
        samples_meta = self._load_samples_metadata()
        
        # Process to satisfy representation of a Dataset
        
        
        dataset = Dataset(
            name=self.name,
            data=genesets_enrichment,
            features_meta=genesets_meta,
            samples_meta=samples_meta,
        )
        return dataset
        
    
    def _load_genesets(self) -> tuple:
        
        col_geneset_name = "gene_set"
        
        # Results may be scattered across multiple files, conatenate all
        dfs = []
        for path in self.genesets_results_paths:
            df_geneset = pd.read_csv(
                path, header=0, index_col=0
            )
            dfs.append(df_geneset)
        
        # Before concat, assert columnnames are in same order,
        #  not necessary anymore, confirmed concat respects 
        #  column names.
#         assert all( [(df.columns == dfs[0].columns) for df in dfs] )
        df_genesets = pd.concat(dfs, axis=0, sort=False)
#         import pdb
#         pdb.set_trace()
        df_genesets = df_genesets.loc[df_genesets['gene_set'] != 'kegg'].copy()
        
        genesets_enrichment = df_genesets.drop(
            col_geneset_name, axis=1).transpose()
        genesets_meta = df_genesets[[col_geneset_name]].copy()
        return genesets_enrichment, genesets_meta
    
    
    def _load_samples_metadata(self) -> pd.DataFrame:
        return self.xspecies_dataset.samples_meta.copy()
