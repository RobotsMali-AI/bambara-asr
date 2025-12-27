import pandas as pd
import numpy as np
import json
import random
import os
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from typing import Dict, List, Any

# --- Model Definitions (Re-using the Sentence Model) ---
SENTENCE_MODEL_NAME = "dangvantuan/sentence-camembert-base"
MODEL_CACHE = {}

def load_embedding_model(model_name: str):
    """Loads and caches the specified SentenceTransformer model."""
    if model_name not in MODEL_CACHE:
        try:
            model = SentenceTransformer(model_name)
            MODEL_CACHE[model_name] = model
            print(f"✅ Loaded embedding model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading model {model_name}: {e}")
    return MODEL_CACHE[model_name]

# --- Main Evaluation Function ---

def evaluate_clustering_potential(
    manifest_jsonl_path: str,
    model_key: str, # e.g., 'asr_mt', 'lau'
    num_samples_per_cluster: int = 50,
    k_clusters: int = 5,
    output_dir: str = "clustering_results"
) -> str:
    """
    Evaluates the clustering potential of a model's transcription output 
    by comparing K-Means clusters to human-assigned ground truth labels.

    Returns: The path to the generated CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data and Sampling (Imbalanced Setup)
    all_data = []
    with open(manifest_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            all_data.append(data)

    # Separate French (High Resource) and LRL (Model Output) samples
    french_samples = [
        {'text': item['french'], 'cluster': item['cluster'], 'type': 'FR'}
        for item in all_data if item.get('french')
    ]
    
    # We assume 'french' contains the ground-truth transcription of the French audio.
    # The 'model_key' contains the LRL audio transcription.
    model_samples = [
        {'text': item[model_key], 'cluster': item['cluster'], 'type': 'BamFR'}
        for item in all_data if item.get(model_key) and item.get('cluster')
    ]

    # Sample LRL texts uniformly across ground-truth clusters
    clustered_model_samples: Dict[str, List[Dict[str, str]]] = {}
    for item in model_samples:
        clustered_model_samples.setdefault(item['cluster'], []).append(item)
    
    selected_lrl_samples = []
    for cluster, samples in clustered_model_samples.items():
        # Ensure we don't sample more than available
        sample_size = min(num_samples_per_cluster, len(samples)) 
        selected_lrl_samples.extend(random.sample(samples, sample_size))
        
    # Combine French (all) and LRL (sampled) data
    # This creates the imbalanced dataset
    dataset = french_samples + selected_lrl_samples
    
    if not dataset:
        return "Error: No valid data found for clustering."
        
    print(f"Dataset Size: {len(dataset)} (FR: {len(french_samples)}, LRL: {len(selected_lrl_samples)})")
    
    # Prepare inputs for embedding
    texts = [item['text'] for item in dataset]
    ground_truth_labels = [item['cluster'] for item in dataset]
    language_types = [item['type'] for item in dataset]

    # 2. Embedding Generation and Normalization
    
    # Use the sentence model (assuming transcriptions are sentence-level)
    model = load_embedding_model(SENTENCE_MODEL_NAME)
    
    print("Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # CRITICAL STEP: Normalizing vectors makes Euclidean distance proportional 
    # to Cosine Distance (maximizing cosine similarity).
    normalized_embeddings = normalize(embeddings, axis=1)

    # 3. K-Means Clustering
    print(f"Running K-Means clustering with K={k_clusters}...")
    kmeans = KMeans(
        n_clusters=k_clusters, 
        random_state=42, 
        n_init=20, # Run 20 times to find a better local minimum
        verbose=0
    )
    kmeans.fit(normalized_embeddings)
    cluster_labels = kmeans.labels_ # K-Means assigned cluster labels

    # 4. Evaluation: Ground Truth Label Distribution
    
    # Create a DataFrame for easy analysis
    df = pd.DataFrame({
        'kmeans_cluster': cluster_labels,
        'ground_truth_label': ground_truth_labels,
        'language_type': language_types
    })
    
    # Calculate distribution of ground truth labels within each K-Means cluster
    distribution_stats = {}
    ground_truth_names = df['ground_truth_label'].unique()
    
    for i in range(k_clusters):
        cluster_df = df[df['kmeans_cluster'] == i]
        total_in_cluster = len(cluster_df)
        if total_in_cluster == 0:
            continue
            
        label_counts = cluster_df['ground_truth_label'].value_counts()
        cluster_stats = {'KMeans Cluster': f'Cluster {i}'}
        
        # Calculate percentage for each ground truth label
        for gt_label in ground_truth_names:
            count = label_counts.get(gt_label, 0)
            percentage = (count / total_in_cluster) * 100
            cluster_stats[gt_label] = f"{percentage:.1f}% ({count})"
        
        distribution_stats[i] = cluster_stats

    # 5. Save Results to CSV
    csv_filename = os.path.join(output_dir, f"clustering_stats_{model_key}.csv")
    stats_df = pd.DataFrame(list(distribution_stats.values()))
    stats_df.to_csv(csv_filename, index=False)
    print(f"✅ Clustering statistics saved to {csv_filename}")

    # 6. Visualization (Dimensionality Reduction and Plotting)
    
    # Reduce dimensionality to 2D for plotting using t-SNE
    print("Reducing dimensionality with t-SNE for visualization...")
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        metric='cosine', # Use cosine metric in t-SNE for consistency
        init='pca' 
    )
    # Fit t-SNE on the normalized embeddings
    embeddings_2d = tsne.fit_transform(normalized_embeddings)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Define colors for ground truth labels
    unique_labels = sorted(df['ground_truth_label'].unique())
    num_labels = len(unique_labels)
    cmap = plt.cm.get_cmap('Spectral', num_labels)
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    # Scatter plot, colored by GROUND TRUTH label
    for i, label in enumerate(unique_labels):
        indices = df[df['ground_truth_label'] == label].index
        
        # Differentiate between French (FR) and LRL (LRL) samples by marker shape
        fr_indices = [idx for idx in indices if language_types[idx] == 'FR']
        lrl_indices = [idx for idx in indices if language_types[idx] == 'BamFR']
        
        # Plot FR samples (larger circle)
        plt.scatter(
            embeddings_2d[fr_indices, 0], 
            embeddings_2d[fr_indices, 1], 
            color=label_to_color[label], 
            marker='o', 
            s=100, 
            alpha=0.6, 
            label=f'{label} (FR)'
        )
        
        # Plot LRL samples (smaller cross)
        plt.scatter(
            embeddings_2d[lrl_indices, 0], 
            embeddings_2d[lrl_indices, 1], 
            color=label_to_color[label], 
            marker='x', 
            s=50, 
            alpha=0.9, 
            label=f'{label} (BamFR)' if not f'{label} (FR)' in plt.gca().get_legend_handles_labels()[1] else None # Avoid duplicate labels in legend
        )
        
    plt.title(f'K-Means Clusters (K={k_clusters}) on Normalized Embeddings for Model: {model_key}\nColored by Ground Truth Topic Label', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title="Ground Truth Topic", loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(True)
    
    plot_filename = os.path.join(output_dir, f"clustering_plot_{model_key}.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close() # Close plot to free up memory
    print(f"✅ Visualization plot saved to {plot_filename}")

    return csv_filename