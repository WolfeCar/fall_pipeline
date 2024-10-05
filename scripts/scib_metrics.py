# ###############################################################################################################################################################################
# # !/usr/bin/env python
# # coding: utf-8

# import scanpy as sc
# import scib
# import warnings
# import pandas as pd
# import numpy as np
# import scipy.sparse as sp
# from datetime import datetime
# import os
# from typing import Any, Union
# import adata

# warnings.filterwarnings('ignore')

# # File paths

# counts_file = snakemake.input[0]
# obs_file = snakemake.input[1]
# output_path = snakemake.output[0]

# batch_key = 'assay'
# label_key = 'cell_type'

# # Load the counts matrix from the .npz file
# counts_data = np.load(counts_file)['embeddings']  # array1(jack) or arr_0(tony) or embeddings(jaggar)?

# # Load the observation metadata from the .pkl file
# obs_data = pd.read_pickle(obs_file)

# # Ensure the counts matrix has the correct shape
# if counts_data.shape[0] != obs_data.shape[0]:
#     raise ValueError("The number of cells in the counts matrix does not match the observation metadata.")

# # Assuming the latent space is created from the counts data
# latent_space_data = counts_data  # Replace this with your model's transformation if necessary

# # Create AnnData object for the latent space
# adata_latent = sc.AnnData(X=sp.csr_matrix(latent_space_data))
# adata_latent.obs = obs_data.copy()  # Use the same observation metadata

# # Align cell identifiers
# if not np.array_equal(adata.obs.index, adata_latent.obs.index):
#     adata_latent.obs.index = adata.obs.index

# # Ensure datasets have the same cells after alignment
# if not np.array_equal(adata.obs.index, adata_latent.obs.index):
#     raise ValueError("The datasets still have different cell identifiers after alignment.")

# # Convert the batch_key and label_key columns to categorical type
# for key in [batch_key, label_key]:
#     adata_latent.obs[key] = adata_latent.obs[key].astype('category')
#     adata.obs[key] = adata.obs[key].astype('category')

# # Perform PCA for the silhouette score calculation and clustering
# sc.tl.pca(adata_latent, n_comps=50)
# adata_latent.obsm['X_pca'] = adata_latent.obsm['X_pca'][:, :50]  # Ensure 50 components
# sc.pp.neighbors(adata_latent, use_rep='X_pca')
# sc.tl.louvain(adata_latent, key_added='louvain')

# # Compute metrics
# kbet = scib.me.kBET(adata_latent, batch_key=batch_key, label_key=label_key, type_='embed', embed='X_pca')
# graph_conn = scib.me.graph_connectivity(adata_latent, label_key=label_key)
# nmi = scib.me.nmi(adata_latent, cluster_key='louvain', label_key=label_key)
# ari = scib.me.ari(adata_latent, cluster_key='louvain', label_key=label_key)
# asw_label = scib.me.silhouette(adata_latent, group_key=label_key, embed='X_pca')
# asw_label_batch = scib.me.silhouette_batch(adata_latent, batch_key=batch_key, group_key=label_key, embed='X_pca')
# pcr_batch = scib.me.pcr_comparison(adata, adata_latent, batch_key)
# clisi = scib.me.clisi_graph(adata_latent, label_key=label_key, type_='embed', use_rep='X_pca')

# # Store results in a dictionary
# results = {
#     'kBET': kbet,
#     'NMI': nmi,
#     'ARI': ari,
#     'ASW_label': asw_label,
#     'ASW_label_batch': asw_label_batch,
#     'PCR_batch': pcr_batch,
#     'Graph Connectivity': graph_conn,
#     'cLISI': clisi
# }

# # Save results to a CSV file
# df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])
# df.to_csv(output_path)

# print("Metrics calculation completed successfully.")
###############################################################################################################################################################################

###############################################################################################################################################################################################################
# file often takes 50-60 mins to run, will output warnings -CJW
###############################################################################################################################################################################################################

# !/usr/bin/env python
# coding: utf-8

import scanpy as sc
import scib
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from typing import Any, Union

warnings.filterwarnings('ignore')

# /mnt/projects/debruinz_project/carly_wolfe/assay_vae_z.npz
# File paths
# uncorrected_path = '/mnt/projects/debruinz_project/carly_wolfe/1mhuman_use.h5ad'
# #^ Raw data 
# counts_file = '/mnt/projects/debruinz_project/carly_wolfe/assay_vae_z.npz'
# obs_file = '/mnt/projects/debruinz_project/carly_wolfe/assay_vae_z.pkl'
# var_file = '/mnt/projects/debruinz_project/summer_census_data/human_gene_metadata.pkl' #Stays the same
# output_path = '/mnt/home/wolfeca/scib-pipeline/bioc/monday_night/metrics_assay_vae_z.csv'

counts_file = snakemake.input[0]
obs_file = snakemake.input[1]
uncorrected_path = snakemake.input[2]
output_path = snakemake.output[0]

batch_key = 'assay'
label_key = 'cell_type'

# Load the counts matrix from the .npz file
counts_data = np.load(counts_file)['embeddings'] # array1(jack) or arr_0(tony) or embeddings(jaggar)?

# Load the observation metadata from the .pkl file
obs_data = pd.read_pickle(obs_file)

# Ensure the counts matrix has the correct shape
if counts_data.shape[0] != obs_data.shape[0]:
    raise ValueError("The number of cells in the counts matrix does not match the observation metadata.")

# Assuming the latent space is created from the counts data
latent_space_data = counts_data  # Replace this with your model's transformation if necessary

# Create AnnData object for the latent space
adata_latent = sc.AnnData(X=sp.csr_matrix(latent_space_data))
adata_latent.obs = obs_data.copy()  # Use the same observation metadata

# Load the uncorrected data
adata = sc.read(uncorrected_path, cache=True)

# Align cell identifiers
if not np.array_equal(adata.obs.index, adata_latent.obs.index):
    adata_latent.obs.index = adata.obs.index

# Ensure datasets have the same cells after alignment
if not np.array_equal(adata.obs.index, adata_latent.obs.index):
    raise ValueError("The datasets still have different cell identifiers after alignment.")

# Convert the batch_key and label_key columns to categorical type
for key in [batch_key, label_key]:
    adata_latent.obs[key] = adata_latent.obs[key].astype('category')
    adata.obs[key] = adata.obs[key].astype('category')

# Perform PCA for the silhouette score calculation and clustering
sc.tl.pca(adata_latent, n_comps=50)
adata_latent.obsm['X_pca'] = adata_latent.obsm['X_pca'][:, :50]  # Ensure 50 components
sc.pp.neighbors(adata_latent, use_rep='X_pca')
sc.tl.louvain(adata_latent, key_added='louvain')

# Compute
kbet = scib.me.kBET(adata_latent, batch_key=batch_key, label_key=label_key, type_='embed', embed='X_pca')
graph_conn = scib.me.graph_connectivity(adata_latent, label_key=label_key)
nmi = scib.me.nmi(adata_latent, cluster_key='louvain', label_key=label_key)
ari = scib.me.ari(adata_latent, cluster_key='louvain', label_key=label_key)
asw_label = scib.me.silhouette(adata_latent, group_key=label_key, embed='X_pca')
pcr_batch = scib.me.pcr_comparison(adata, adata_latent, batch_key)
isolated_label_silhouette = scib.me.isolated_labels_asw(adata_latent, label_key=label_key, batch_key=batch_key, embed='X_pca')
clisi = scib.me.clisi_graph(adata_latent, label_key=label_key, type_='embed', use_rep='X_pca')

# # Compute iLISI using custom function
# def ilisi_graph(
#     adata: Any,
#     batch_key: str,
#     label_key: str,
#     type_: str,
#     use_rep: str = "X_pca",
#     k0: int = 90,
#     subsample: Union[int, None] = None,
#     scale: bool = True,
#     n_cores: int = 1,
#     verbose: bool = False
# ) -> float:
#     if type_ not in ['knn', 'embed', 'full']:
#         raise ValueError("type_ must be one of 'knn', 'embed' or 'full'")
    
#     if verbose:
#         print("Starting iLISI computation...")
    
#     if type_ == 'embed':
#         if use_rep not in adata.obsm:
#             raise ValueError(f"{use_rep} not found in adata.obsm")
#         data = adata.obsm[use_rep]
#     elif type_ == 'full':
#         data = adata.X
#     elif type_ == 'knn':
#         sc.pp.neighbors(adata, n_neighbors=k0)
#         data = adata.obsp['distances']
    
#     if subsample is not None:
#         if not (0 < subsample <= 100):
#             raise ValueError("subsample must be an integer between 1 and 100")
#         idx = np.random.choice(np.arange(adata.n_obs), size=int(adata.n_obs * (subsample / 100)), replace=False)
#         adata = adata[idx]
#         data = data[idx]
    
#     if verbose:
#         print("Computing LISI scores...")
    
#     from scib.metrics import lisi_graph

#     ilisi_result = lisi_graph(
#         adata,
#         batch_key=batch_key,
#         label_key=label_key,
#         type_=type_,
#         use_rep=use_rep,
#         k0=k0,
#         subsample=subsample,
#         scale=scale,
#         n_cores=n_cores,
#         verbose=verbose
#     )
    
#     # Return the mean of the LISI scores as a single value
#     return np.mean(ilisi_result)

# ilisi_result = ilisi_graph(adata_latent, batch_key=batch_key, label_key=label_key, type_='embed', use_rep='X_pca')

# Store results in a dictionary
results = {
    'kBET': kbet,
    'NMI': nmi,
    'ARI': ari,
    'ASW_label': asw_label,
    'PCR_batch': pcr_batch,
    'Graph Connectivity': graph_conn,
    'Isolated_Label_Silhouette': isolated_label_silhouette,
    'cLISI': clisi,
    # 'iLISI': ilisi_result
}

# Save results to a CSV file
df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])
df.to_csv(output_path)

###############################################################################################################################################################################################################