import numpy as np
import pandas as pd
import pickle
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_and_save_embeddings(npz_file_path, pkl_file_path, embeddings_output_path, n_neighbors=30, n_components=2, metric='cosine', min_dist=0.3, random_state=42, n_jobs=40, n_epochs=200):
    def load_npz_file(npz_path):
        try:
            with np.load(npz_path) as data:
                key = list(data.keys())[0]
                return data[key]
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            return None

    def load_pkl_file(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading PKL file: {e}")
            return None

    def perform_umap(data, n_neighbors, n_components, metric, min_dist, random_state, n_jobs, n_epochs):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            random_state=random_state,
            n_jobs=n_jobs,
            n_epochs=n_epochs
        )
        embedding = reducer.fit_transform(data)
        return embedding

    z_star_data = load_npz_file(npz_file_path)
    metadata = load_pkl_file(pkl_file_path)
    
    if z_star_data is None or metadata is None:
        print("Failed to load data or metadata. Exiting.")
        return None

    if z_star_data.shape[0] != metadata.shape[0]:
        print(f"Mismatch in data length: Data has {z_star_data.shape[0]} rows, metadata has {metadata.shape[0]} rows.")
        return None
    
    metadata.reset_index(drop=True, inplace=True)
    top_n_categories = metadata['cell_type'].value_counts().nlargest(15).index
    filtered_metadata = metadata[metadata['cell_type'].isin(top_n_categories)]
    filtered_data = z_star_data[filtered_metadata.index]

    umap_embedding = perform_umap(filtered_data, n_neighbors, n_components, metric, min_dist, random_state, n_jobs, n_epochs)
    
    np.savez_compressed(embeddings_output_path, embeddings=umap_embedding)
    print(f"Embeddings saved to {embeddings_output_path}")
    
    return embeddings_output_path

def plot_from_embeddings(embeddings_path, pkl_file_path, png_output_path, plot_title, color_by, n_largest=15, plot_all=True):
    def load_pkl_file(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading PKL file: {e}")
            return None

    def plot_category(embedding, metadata, category, png_output_path, plot_title, n_largest, fig_size=(14, 8), point_size=0.5, plot_all=True):
        plt.figure(figsize=fig_size)
        unique_values = metadata[category].value_counts().nlargest(n_largest).index

        cmap = plt.get_cmap('nipy_spectral', len(unique_values))
        color_list = [cmap(i) for i in range(len(unique_values))]

        if plot_all:
            # Plot the rest of the points in grey first
            other_metadata = metadata[~metadata[category].isin(unique_values)]
            other_indices = other_metadata.index
            other_indices = np.intersect1d(other_indices, range(len(embedding)))  # Ensure indices are valid
            other_embedding = embedding[other_indices]
            plt.scatter(other_embedding[:, 0], other_embedding[:, 1], color='grey', label='Other', s=point_size)

        # Collect all points for the categories in a single scatter plot
        all_indices = []
        all_colors = []
        for i, value in enumerate(unique_values):
            idx = metadata[category] == value
            all_indices.extend(metadata[idx].index)
            all_colors.extend([color_list[i]] * sum(idx))

        # Randomize the order of points
        combined = list(zip(all_indices, all_colors))
        np.random.shuffle(combined)
        shuffled_indices, shuffled_colors = zip(*combined)
        shuffled_indices = list(shuffled_indices)
        
        # Ensure all indices are within the valid range of the embedding
        valid_indices = np.intersect1d(shuffled_indices, range(len(embedding)))
        all_embedding = embedding[valid_indices]

        # Match the colors for valid indices
        valid_colors = [shuffled_colors[i] for i, idx in enumerate(shuffled_indices) if idx in valid_indices]

        plt.scatter(all_embedding[:, 0], all_embedding[:, 1], c=valid_colors, s=point_size)

        plt.title(plot_title)
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=label)
                          for i, label in enumerate(unique_values)]
        if plot_all:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Other'))
        plt.legend(handles=legend_handles, title=category, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(png_output_path, bbox_inches='tight')
        plt.close()
        return png_output_path

    embeddings = np.load(embeddings_path)['embeddings']
    metadata = load_pkl_file(pkl_file_path)
    
    if metadata is None:
        print("Failed to load metadata. Exiting.")
        return

    metadata.reset_index(drop=True, inplace=True)
    top_n_categories = metadata[color_by].value_counts().nlargest(n_largest).index

    # Filter metadata for top categories
    filtered_metadata = metadata[metadata[color_by].isin(top_n_categories)]

    image_path = plot_category(embeddings, metadata, color_by, png_output_path, plot_title, n_largest, plot_all=plot_all)

    print(f"UMAP plot from embeddings saved to {png_output_path}")
    return image_path

# File paths and settings
if __name__ == "__main__":
    npz_file_path = snakemake.input[0]
    pkl_file_path = snakemake.input[1]
    png_output_path = snakemake.output[0]
    embeddings_output_path = snakemake.output[1]
    plot_title = "UMAP Projection: Plain VAE colored by cell type"
    

    # Check if embeddings file exists
    if os.path.exists(embeddings_output_path):
        print(f"Embeddings file found at {embeddings_output_path}. Using existing embeddings.")
        plot_from_embeddings(embeddings_output_path, pkl_file_path, png_output_path, plot_title, color_by='cell_type', n_largest=15, plot_all=True)
    else:
        print(f"Embeddings file not found. Computing embeddings and saving to {embeddings_output_path}.")
        embeddings_path = process_and_save_embeddings(npz_file_path, pkl_file_path, embeddings_output_path, n_neighbors=30, n_components=2, metric='cosine', min_dist=0.3, random_state=42, n_jobs=40, n_epochs=200)
        
        if embeddings_path:
            plot_from_embeddings(embeddings_path, pkl_file_path, png_output_path, plot_title, color_by='cell_type', n_largest=15, plot_all=True)
