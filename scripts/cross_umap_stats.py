import numpy as np
import umap
import matplotlib.pyplot as plt
import os
import scipy.sparse as sp
from scipy.stats import pearsonr, ttest_ind

# Snakemake input paths (4 input files from snakemake)
mouse_cis_file = snakemake.input[0]
human_cross_file = snakemake.input[1]
human_cis_file = snakemake.input[2]
mouse_cross_file = snakemake.input[3]

# Snakemake output paths
png_output_mouse_human = snakemake.output[0]                 # Path for the UMAP plot (mouse cis vs human cross)
png_output_mouse_cross_human_cis = snakemake.output[1]       # Path for the UMAP plot (mouse cross vs human cis)
embeddings_mouse_human_output = snakemake.output[2]          # Embeddings for mouse cis vs human cross (.npy)
embeddings_mouse_cross_human_cis_output = snakemake.output[3]  # Embeddings for mouse cross vs human cis (.npy)
stats_output_file = snakemake.output[4]                      # Output file for saving the statistics


# Load data from .npz files
def load_data(file_path):
    mat = sp.load_npz(file_path)
    return mat.toarray()

# UMAP visualization and saving embeddings
def umap_visualization(data1, data2, label1, label2, group_id, png_output, embeddings_output):
    # Ensure both arrays have the same number of columns (axis=1)
    if data1.shape[1] != data2.shape[1]:
        min_cols = min(data1.shape[1], data2.shape[1])
        data1 = data1[:, :min_cols]
        data2 = data2[:, :min_cols]

    reducer = umap.UMAP()
    combined_data = np.concatenate((data1, data2), axis=0)
    embedding = reducer.fit_transform(combined_data)

    print(f"UMAP Visualization for {label1} vs. {label2} in {group_id}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding min: {embedding.min()}, max: {embedding.max()}")

    # Save the UMAP embeddings to a .npy file
    np.save(embeddings_output, embedding)  # Save the embeddings as a .npy file
    print(f"Embeddings saved to {embeddings_output}")

    # Generate and save the UMAP plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], label=label1, alpha=0.5)
    plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], label=label2, alpha=0.5)
    plt.legend()
    plt.title(f'UMAP Visualization for {group_id}')

    plt.savefig(png_output)  # Save the plot to the specified .png file path
    plt.close()  # Close the plot to free memory

    print(f"UMAP plot saved to {png_output}")

# Statistical comparison and saving to a text file
def statistics(data1, data2, label1, label2, group_id, stats_output_file):
    # Ensure both flattened arrays have the same length
    if data1.size != data2.size:
        min_size = min(data1.size, data2.size)
        data1 = data1.flatten()[:min_size]
        data2 = data2.flatten()[:min_size]
    
    correlation, _ = pearsonr(data1, data2)
    fold_change = np.mean(data1) / np.mean(data2)
    t_stat, p_value = ttest_ind(data1, data2)
    
    # Print results to console
    print(f"Group {group_id}:")
    print(f"Pearson correlation between {label1} and {label2}: {correlation}")
    print(f"T-test statistic: {t_stat}")
    print(f"T-test p-value (for difference in means): {p_value}")
    print(f"Fold change: {fold_change}")
    
    # Save results to a text file
    with open(stats_output_file, 'a') as f:  # Use 'a' to append to the file
        f.write(f"Group {group_id}:\n")
        f.write(f"Pearson correlation between {label1} and {label2}: {correlation}\n")
        f.write(f"T-test statistic: {t_stat}\n")
        f.write(f"T-test p-value (for difference in means): {p_value}\n")
        f.write(f"Fold change: {fold_change}\n\n")
    
    return correlation, fold_change, t_stat, p_value

# Load matrices from Snakemake inputs
MM_matrix = load_data(mouse_cis_file)    # mouse_cis
HM_matrix = load_data(human_cross_file)  # human_cross
HH_matrix = load_data(human_cis_file)    # human_cis
MH_matrix = load_data(mouse_cross_file)  # mouse_cross

# Example group_id (could be passed differently)
group_id = "group1"

# Generate UMAP visualizations and save embeddings separately for mouse-human and mouse-cross-human-cis
umap_visualization(MM_matrix, HM_matrix, 'mouse cis', 'human cross', group_id, png_output_mouse_human, embeddings_mouse_human_output)
umap_visualization(MH_matrix, HH_matrix, 'mouse cross', 'human cis', group_id, png_output_mouse_cross_human_cis, embeddings_mouse_cross_human_cis_output)

# Calculate R-squared, fold change, and p-value, and save to text file
statistics(MM_matrix, HM_matrix, 'mouse cis', 'human cross', group_id, stats_output_file)
statistics(MH_matrix, HH_matrix, 'mouse cross', 'human cis', group_id, stats_output_file)





# import numpy as np
# import umap
# import matplotlib.pyplot as plt
# import os
# import scipy.sparse as sp
# from scipy.stats import pearsonr, ttest_ind

# # Snakemake input paths (4 input files from snakemake)
# mouse_cis_file = snakemake.input[0]
# human_cross_file = snakemake.input[1]
# human_cis_file = snakemake.input[2]
# mouse_cross_file = snakemake.input[3]

# # Snakemake output paths
# png_output_path = snakemake.output[0]                 # Path for the UMAP plot
# embeddings_mouse_human_output = snakemake.output[1]    # Embeddings for mouse_cis vs human_cross
# embeddings_mouse_cross_human_cis_output = snakemake.output[2]  # Embeddings for mouse_cross vs human_cis

# # Load data from .npz files
# def load_data(file_path):
#     mat = sp.load_npz(file_path)
#     return mat.toarray()

# # UMAP visualization and saving embeddings
# def umap_visualization(data1, data2, label1, label2, group_id, png_output, embeddings_output):
#     reducer = umap.UMAP()
#     combined_data = np.concatenate((data1, data2), axis=0)
#     embedding = reducer.fit_transform(combined_data)

#     print(f"UMAP Visualization for {label1} vs. {label2} in {group_id}")
#     print(f"Embedding shape: {embedding.shape}")
#     print(f"Embedding min: {embedding.min()}, max: {embedding.max()}")

#     # Save the UMAP embeddings to a file
#     np.save(embeddings_output, embedding)  # Save the embeddings as a .npy file
#     print(f"Embeddings saved to {embeddings_output}")

#     # Generate and save the UMAP plot
#     plt.figure(figsize=(12, 8))
#     plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], label=label1, alpha=0.5)
#     plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], label=label2, alpha=0.5)
#     plt.legend()
#     plt.title(f'UMAP Visualization for {group_id}')

#     plt.savefig(png_output)  # Save the plot to the specified .png file path
#     plt.close()  # Close the plot to free memory

#     print(f"UMAP plot saved to {png_output}")

# # Statistical comparison
# def statistics(data1, data2, label1, label2, group_id):
#     correlation, _ = pearsonr(data1.flatten(), data2.flatten())
#     fold_change = np.mean(data1) / np.mean(data2)
#     t_stat, p_value = ttest_ind(data1.flatten(), data2.flatten())
    
#     print(f"Group {group_id}:")
#     print(f"Pearson correlation between {label1} and {label2}: {correlation}")
#     print(f"T-test statistic: {t_stat}")
#     print(f"T-test p-value (for difference in means): {p_value}")
    
#     return correlation, fold_change, t_stat, p_value

# # Load matrices from Snakemake inputs
# MM_matrix = load_data(mouse_cis_file)    # mouse_cis
# HM_matrix = load_data(human_cross_file)  # human_cross
# HH_matrix = load_data(human_cis_file)    # human_cis
# MH_matrix = load_data(mouse_cross_file)  # mouse_cross

# # Example group_id (could be passed differently)
# group_id = "group1"

# # Generate UMAP visualizations and save embeddings separately for mouse-human and mouse-cross-human-cis
# umap_visualization(MM_matrix, HM_matrix, 'mouse cis', 'human cross', group_id, png_output_path, embeddings_mouse_human_output)
# umap_visualization(MH_matrix, HH_matrix, 'mouse cross', 'human cis', group_id, png_output_path, embeddings_mouse_cross_human_cis_output)

# # Calculate R-squared, fold change, and p-value
# statistics(MM_matrix, HM_matrix, 'mouse cis', 'human cross', group_id)
# statistics(MH_matrix, HH_matrix, 'mouse cross', 'human cis', group_id)
