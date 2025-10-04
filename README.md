# ECHO-iGEM-IITMadras
ECHO (Epigenetic Control with Hybrid Optimization) is a software tool developed by the iGEM team of IIT Madras, which enables identification of those methylation sites (CpG islands) that are most relevant in regulating a gene's expression.

Given a methylation dataset and the mapped gene expression values, ECHO trains two models -
1. An AdaptiveRegressiveCNN (inspired by DeepMethyGene, Yan et al, 2025) - a model capable of better predictive accuracy, but loses individual feature importance
2. An ElasticNet Linear Regressor (inspired by geneEXPLORE, Kim et al, 2020) - a model with lesser predictive accuracy, but retains individual feature importance

The algorithm then methylates sites predicted important (by the ElasticNet), and runs modified inputs via the AdaptiveRegressiveCNN, to provide a final recommendation of sites for methylation to upregulate/downregulate gene expression.

> This tool is developed to eventually aide in experiment design for use with the dCas9-DMNT system designed by iGEM IIT-Madras, 2025.

Visit our wiki at [https://2025.igem.wiki/iit-madras/](url) for more info!

## Installation Instructions

Cloning this repo requires the `lfs` (large file size) module of git. So, first make sure its installed by running `git lfs install` in terminal.
Clone the repo into your desired directory with `git clone https://github.com/harsha-1305/ECHO-iGEM-IITMadras.git`. Follow with `git lfs pull` if required.

To be safe, manually create a top-level empty directory `instances` within `ECHO-iGEM-IITMadras`.

Before proceeding, ensure your directory `ECHO-iGEM-IITMadras/` looks as follows -

```bash
ECHO-iGEM-IITMadras
├── cli.py
├── data
│   ├── chrom_sorted_cpg_probes
│   │   ├── chr1.csv
│   │   ├── chr10.csv
│   │   ├── chr11.csv
│   │   ├── chr12.csv
│   │   ├── chr13.csv
│   │   ├── chr14.csv
│   │   ├── chr15.csv
│   │   ├── chr16.csv
│   │   ├── chr17.csv
│   │   ├── chr18.csv
│   │   ├── chr19.csv
│   │   ├── chr2.csv
│   │   ├── chr20.csv
│   │   ├── chr21.csv
│   │   ├── chr22.csv
│   │   ├── chr3.csv
│   │   ├── chr4.csv
│   │   ├── chr5.csv
│   │   ├── chr6.csv
│   │   ├── chr7.csv
│   │   ├── chr8.csv
│   │   ├── chr9.csv
│   │   ├── chrX.csv
│   │   └── chrY.csv
│   ├── geneEx_data.csv
│   ├── probeMap_hugo_gencode_good_hg19_V24lift37_probemap
│   └── probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy
├── echo
│   ├── __init__.py
│   ├── algorithms.py
│   ├── checks.py
│   ├── instances.py
│   ├── models.py
│   └── trial.py
└── instances
```

A quick breakdown -
1. cli.py - python script equipped with a command line interface; the main python file we will be calling from terminal to execute our programs
2. data/ - directory containing sorted probe datasets, gene expression, and relevant maps
3. echo/ - contains the main scripts for the ECHO algorithms
4. instances/ - the output file where your worksessions will be stored

## A Sample Runthrough -

I provide a quick runthrough on gene AADAT while considering methylations within a site of 10Mb. Launch terminal and navigate within the ECHO-iGEM-IITMadras folder, then, execute the steps in order -
1. `python -m cli run-setup ` - checks if all data files were installed properly
2. `python -m cli create-instance -gi AADAT -w 10` - Creates file `instances\AADAT_10.0Mb`.
  - Inputs -
    - -gi | --geneID : the HGNC geneID for your gene
    - -w  | --window : the window of methylation on chromosome being considered
3. `python -m cli compile-probes -gi AADAT -w 10` - Creates file `instances\AADAT_10.0Mb_gene_specific_probes.csv`. Compiles probes on the gene's chromosome within the window of consideration.
4. `python -m cli train-elasticNet -gi AADAT -w 10` - Creates file `instances\AADAT_10.0Mb_gene_specific_elasticNet_weights.csv`. Trains an ElasticNet on the gene dataset, and saves the learned weights.
5. `python -m cli plot-elasticNet -gi AADAT -w 10 -ud -sd` - Visualizes the learnt ElasticNet weights.
  - Inputs -
    - -ud | --use_distance : Plot with distances on the X axis (instead of CpG number).
    - -sd | --save_distance : Saves a column with chromosome distance to `instances\AADAT_10.0Mb_gene_specific_elasticNet_weights.csv`.
6. `python -m cli train-arCNN -gi AADAT -w 10` - Creates files `instances\AADAT_10.0Mb_arCNN_cnn.pth` and `instances\AADAT_10.0Mb_arCNN_meta.json`. Trains an AdaptiveRegressiveCNN on the gene dataset, and saves the learned model parameters.
7. `python -m cli get-gradCAM -gi AADAT -w 10` - Creates file `instances\AADAT_10.0Mb_gradCAM_importance.csv`. Calculates the grad_CAM importances.
8. `python -m cli plot-gradCAM -gi AADAT -w 10 -ud -sd` - Visualizes the learnt grad_CAM importances.
9. `python -m cli calc-imp-cpgs -gi AADAT -w 10 -a es -dir 1` - Creates file `instances\AADAT_10.0Mb_imp_cpgs_elasticNet_sequential_1.csv`. Arranges CpGs sites in order or importance.
  - Inputs -
    - -a   | --algorithm : Specify the algorithm used -
      1. standard_sequential   | ss : Sequential methylation from downstream to upstream
      2. circular_sequential   | cs : Methylates sites closest to genes first
      3. gradCAM_sequential    | gs : Methylates according to gradCAM importance
      4. elasticNet_sequential | es : Methylates according to elasticNet importance (recommended)
    - -dir | --direction : Specify whether to upregulate or downregulate -
      1.  1 : Calculate for upregulation
      2. -1 : Calculate for downregulation
10. `python -m cli compare-weights -gi AADAT -w 10` - Plots and compares ElasticNet weights with the AdaptiveRegressiveCNN gradCAM values.

Check the output file at `instances\AADAT_10.0Mb_imp_cpgs_elasticNet_sequential_1.csv` for the final list of recommendations.

## Credits

Thank you to the IIT Madras faculty and our alumni for helping us develop software for Analysis of DNA Interactions for Tuning Regulatory Impacts.

## Reach out to us!

Contact us via email at [igem@smail.iitm.ac.in](mail)!
