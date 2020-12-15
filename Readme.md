# Cross-species: linear models analysis of genes and genesets associations with MLS and other species traits.

To reproduce the linear analysis:

1) Get the genes and genesets datasets data from the link below. There can be found 2 folders for generating the genes `XSPECIES_GENES` and genesets `XSPECIES_GENESETS_KEGG` datasets, respectively.
https://drive.google.com/drive/folders/16UI0VxdwCNrHZ1gRmhpZBJK1t-oJk2Ji?usp=sharing

2) Install and activate the environment necessary for running the code.

```bash
conda env create -f environment.yml
conda activate xspecies
```

3) Run the analysis via a CLI interface. From the repository's directory:

```bash
# Generate results for genes
python linearmodels.py --type_input genes analyze-models PATH_TO_GENES_DATASET_DIR OUTPUT_PATH_FOR_GENES_MODELS_DIR
python linearmodels.py --type_input genes extract-results OUTPUT_PATH_FOR_GENES_MODELS_DIR OUTPUT_PATH_GENES_SUMMARY_DIR

# Generate results for genesets
python linearmodels.py --type_input genesets analyze-models PATH_TO_GENESETS_DATASET_DIR OUTPUT_PATH_FOR_GENESETS_MODELS_DIR
python linearmodels.py --type_input genesets extract-results OUTPUT_PATH_FOR_GENESETS_MODELS_DIR OUTPUT_PATH_GENESETS_SUMMARY_DIR

# CLI interface help:
python linearmodels.py --help
```

4) Explore the results in `OUTPUT_PATH_GENES_SUMMARY_DIR` and `OUTPUT_PATH_GENESETS_SUMMARY_DIR`.
