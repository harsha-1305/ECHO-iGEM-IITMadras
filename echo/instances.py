from pathlib import Path
import pandas as pd
import os
import json
import sys

def create_instance(gene_id: str, window: float, dest_path) -> None :
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).resolve().parent.parent
    file_path = base_path / "data"
    
    gene_map_path = file_path / "probeMap_hugo_gencode_good_hg19_V24lift37_probemap"
    gene_map = pd.read_csv(gene_map_path, sep='\t').drop(['strand', 'gene', 'chromEnd', 'thickStart', 'thickEnd', 'blockCount', 'blockSizes', 'blockStarts'], axis=1)

    if gene_id not in gene_map['id'].values :
        print(f"CRITICAL ERROR : '{gene_id}' not available")
        return
    gene_loc = gene_map[gene_map['id'] == gene_id].iloc[0].to_list()
    gene_chr = gene_loc[1]
    gene_stt = gene_loc[2]

    instance_name = gene_id + f'_{window}Mb'
    instance_path = dest_path / instance_name
    os.mkdir(instance_path)
    print(f"Created instance at {instance_path}")

    json_str = '{"geneID" : "'+gene_id+'", "window" : '+str(window * (10**6))+', "chr" : "'+gene_chr+'", "start" : '+str(gene_stt)+'}'
    with open(instance_path / "instance_data.json", "w") as f :
        f.write(json_str)
        print("Updated instance_data.json")

def compile_probes(instance_path) -> None :
    with open(instance_path/ "instance_data.json", 'r') as file :
        instance_data = json.load(file)
    geneID = instance_data["geneID"]
    window = instance_data["window"]
    chomID = instance_data["chr"]
    start = instance_data["start"]
    print(f"Loaded gene {geneID} with window {window/(10**6)}Mb, {chomID}, {start}")

    chrom_meth_data_path = Path(__file__).parent.parent / f'data/chrom_sorted_cpg_probes/{chomID}.csv'
    chrom_meth_data = pd.read_csv(chrom_meth_data_path)
    print(f"Loaded probes of {chomID} of shape {chrom_meth_data.shape}. Beginning compiling.")

    cgs_in_range = pd.DataFrame(columns = chrom_meth_data.columns.values)
    for i, row in chrom_meth_data.iterrows():
        if row['chromEnd'] > start + window :
            break
        if row['chromStart'] > start - window :
            cgs_in_range.loc[len(cgs_in_range)] = row
    cgs_in_range = cgs_in_range.drop(columns = ['chrom', 'chromEnd'])

    # Fetching gene expression data
    data_path = Path(__file__).parent.parent / "data"
    geEx_data_path = data_path / "geneEx_data.csv"
    geEx_data = pd.read_csv(geEx_data_path)

    gene_geEx = geEx_data[geEx_data['sample'] == geneID]
    gene_geEx = gene_geEx[['sample'] + list(cgs_in_range.columns.values[2:])].T
    gene_geEx.iloc[0] = [start]
    gene_geEx = gene_geEx.rename(index={'sample' : 'chromStart'})
    gene_geEx.columns = [geneID]

    # Merging above info to create final gene_specific dataset

    gene_data = cgs_in_range.T
    gene_data.columns = gene_data.iloc[0]
    gene_data = gene_data[1:]

    gene_data.insert(loc=0, column = geneID, value = gene_geEx[geneID])
    gene_data.rename(index={'chromStart' : 'geneStart'})
    gene_data.iloc[0] = gene_data.iloc[0] - gene_data.iloc[0][0]

    csv_name = geneID + f'_{window/(10**6)}Mb_gene_specific_probes.csv'
    gene_data_path = instance_path / csv_name
    gene_data.to_csv(gene_data_path)

    print(f"Finised compiling data of shape {gene_data.shape} and exported to {gene_data_path}")
