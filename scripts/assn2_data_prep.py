# ==============================================================================
# CGS4144 - Assignment 2: Data Preparation, Density Plot & PCA Analysis (Python)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import mygene

PATH_TO_REFINEBIO_COUNTS = rf"../data/SRP064298/SRP064298.tsv"
PATH_TO_METADATA = rf"../data/SRP064298/metadata_SRP064298.tsv"

# 1. Load dataset files from refine.bio
# Replace file paths with your actual dataset filenames from Assignment 1
counts_matrix = pd.read_csv(PATH_TO_REFINEBIO_COUNTS, sep="\t", index_col=0)
metadata = pd.read_csv(PATH_TO_METADATA, sep="\t", index_col=0)

print("=== METADATA COLUMNS ===")
print(metadata.columns.tolist())

print("\n=== METADATA SHAPE ===")
print(metadata.shape)

print("\n=== COUNTS SHAPE ===")
print(counts_matrix.shape)

print("\n=== FIRST METADATA ROWS ===")
print(metadata.head())

print("\n=== FIRST 10 METADATA INDEX VALUES ===")
print(metadata.index[:10].tolist())

print("\n=== FIRST 10 COUNT MATRIX COLUMNS ===")
print(counts_matrix.columns[:10].tolist())

if 'refinebio_title' in metadata.columns:
    print("\n=== FIRST 30 SAMPLE TITLES ===")
    print(metadata['refinebio_title'].head(30).to_string())

# Print available metadata columns to help identify the grouping variable
print("--- Available Metadata Columns ---")
print(metadata.columns.tolist())
print("----------------------------------\n")

# 2. Convert Ensembl IDs (e.g., ENSMUSG00000000001) to official Gene Symbols
# Extract clean Ensembl IDs by selecting the first element after splitting on '.'
clean_ensembl = [str(gene_id).split('.') for gene_id in counts_matrix.index]

# Auto-detect species: mouse (ENSMUSG) vs human (ENSG)
species_type = 'mouse' if any('ENSMUSG' in gid for gid in clean_ensembl[:50]) else 'human'
print(f"Detected species for annotation: {species_type}")

mg = mygene.MyGeneInfo()
res = mg.querymany(clean_ensembl, scopes='ensembl.gene', fields='symbol', species=species_type, as_dataframe=True)

# Map gene symbols and filter out unmapped or duplicate gene names
if 'symbol' in res.columns:
    symbol_map = res['symbol'].dropna().to_dict()
    mapped_index = [symbol_map.get(gid, np.nan) for gid in clean_ensembl]
    counts_matrix.index = mapped_index
    counts_clean = counts_matrix[counts_matrix.index.notna()]
    counts_clean = counts_clean[~counts_clean.index.duplicated(keep='first')]
else:
    counts_clean = counts_matrix

# Print matrix dimensions for the report summary (Section 1 requirement)
print("\n--- Expression Matrix Summary ---")
print(f"Total Genes (Rows): {counts_clean.shape}")
print(f"Total Samples (Columns): {counts_clean.shape}")

# 3. Log transformation and median expression calculation (Section 1)
log_counts = np.log2(counts_clean + 1)

# Calculate median expression across samples for each gene
gene_medians = log_counts.median(axis=1)

# Generate and save Density Plot
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(6, 4))
sns.kdeplot(gene_medians, fill=True, color="#69b3a2", alpha=0.8)
plt.title("Density Plot of Per-Gene Median Expression", fontsize=12, fontweight='bold')
plt.xlabel("Median Expression (log2 Counts + 1)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("results/density_plot.png", dpi=300)
plt.close()

# 4. Principal Component Analysis (Section 2)
# Transpose matrix (.T) so samples are rows and genes are columns
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(log_counts.T)

# Calculate percentage of variance explained by PC1 and PC2
var_explained = pca.explained_variance_ratio_ * 100

# Create DataFrame for PCA plot combined with metadata
pca_df = pd.DataFrame(pca_transformed, index=log_counts.columns, columns=['PC1', 'PC2'])

# Select treatment / eating frequency column from metadata
target_group_col = 'refinebio_treatment'

if target_group_col in metadata.columns:
    pca_df['Group'] = metadata.loc[pca_df.index, target_group_col]
else:
    fallback_col = metadata.columns
    pca_df['Group'] = metadata.loc[pca_df.index, fallback_col]
    print(f"\n[Note] '{target_group_col}' not found. Using '{fallback_col}' for plot coloring.")

# Generate and save PCA Plot with legend and labeled axes
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Group',
    s=80,
    alpha=0.8
)
plt.title("PCA Plot of RNA-Seq Samples", fontsize=12, fontweight='bold')
plt.xlabel(f"PC1 ({var_explained:.2f}% Variance)")
plt.ylabel(f"PC2 ({var_explained:.2f}% Variance)")
plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("results/pca_plot.png", dpi=300)
plt.close()

print("\n--- Processing complete! Images saved in 'results/' directory. ---")