# ==============================================================================
# CGS4144 - Assignment 2
# Data Preparation, Density Plot & PCA Analysis
#
# Dataset: SRP064298
# Scientific Question:
# Can liver gene expression distinguish wild-type mice fed ad libitum
# from wild-type mice under night-restricted feeding?
# ==============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import mygene


# ==============================================================================
# 0. Paths
# ==============================================================================

# Project structure expected:
#
# CGS4144_Assn1/
# ├── data/
# │   └── SRP064298/
# │       ├── SRP064298.tsv
# │       └── metadata_SRP064298.tsv
# ├── scripts/
# │   └── assn2_data_prep.py
# └── results/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATH_TO_REFINEBIO_COUNTS = (
    PROJECT_ROOT / "data" / "SRP064298" / "SRP064298.tsv"
)

PATH_TO_METADATA = (
    PROJECT_ROOT / "data" / "SRP064298" / "metadata_SRP064298.tsv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==============================================================================
# 1. Load refine.bio data
# ==============================================================================

print("\n==================================================")
print("1. LOADING DATA")
print("==================================================")

counts_matrix = pd.read_csv(
    PATH_TO_REFINEBIO_COUNTS,
    sep="\t",
    index_col=0
)

metadata = pd.read_csv(
    PATH_TO_METADATA,
    sep="\t",
    index_col=0
)

print(f"Original expression matrix shape: {counts_matrix.shape}")
print(f"Metadata shape: {metadata.shape}")

print(f"\nGenes: {counts_matrix.shape[0]}")
print(f"Sequencing runs: {counts_matrix.shape[1]}")


# ==============================================================================
# 2. Verify that sequencing runs match between counts and metadata
# ==============================================================================

print("\n==================================================")
print("2. CHECKING SAMPLE IDS")
print("==================================================")

count_runs = set(counts_matrix.columns)
metadata_runs = set(metadata.index)

missing_in_metadata = count_runs - metadata_runs
missing_in_counts = metadata_runs - count_runs

print(
    f"Count-matrix runs missing from metadata: "
    f"{len(missing_in_metadata)}"
)

print(
    f"Metadata runs missing from count matrix: "
    f"{len(missing_in_counts)}"
)

if missing_in_metadata:
    raise ValueError(
        "Some columns in the expression matrix are missing "
        "from the metadata."
    )


# ==============================================================================
# 3. Combine sequencing runs belonging to the same biological sample
# ==============================================================================

print("\n==================================================")
print("3. COMBINING TECHNICAL SEQUENCING RUNS")
print("==================================================")

if "refinebio_title" not in metadata.columns:
    raise ValueError(
        "'refinebio_title' was not found in the metadata."
    )

number_of_biological_samples = metadata["refinebio_title"].nunique()

print(
    f"Sequencing runs before combining: "
    f"{counts_matrix.shape[1]}"
)

print(
    f"Unique biological samples: "
    f"{number_of_biological_samples}"
)

# Map:
# SRR2534402 -> WT_AL_ZT00_A_RNA
# SRR2534403 -> WT_AL_ZT00_A_RNA
# etc.
run_to_sample = metadata["refinebio_title"].to_dict()

# Rename columns from sequencing-run IDs to biological-sample names.
counts_by_sample = counts_matrix.rename(columns=run_to_sample)

# Multiple sequencing runs can belong to the same biological sample.
# Since these are technical sequencing runs from the same sample,
# combine their counts by summing them.
counts_by_sample = (
    counts_by_sample
    .T
    .groupby(level=0)
    .sum()
    .T
)

print("\nAfter combining sequencing runs:")
print(f"Genes: {counts_by_sample.shape[0]}")
print(f"Biological samples: {counts_by_sample.shape[1]}")

print("\nFirst biological sample names:")
print(counts_by_sample.columns[:10].tolist())


# ==============================================================================
# 4. Build biological-sample metadata
# ==============================================================================

print("\n==================================================")
print("4. PREPARING SAMPLE METADATA")
print("==================================================")

# Each biological sample has several sequencing runs.
# Keep one metadata row for each biological sample.
sample_metadata = (
    metadata
    .reset_index()
    .drop_duplicates(subset="refinebio_title")
    .set_index("refinebio_title")
)

# Identify the two experimental groups relevant to our question.
sample_metadata["Group"] = np.nan

sample_metadata.loc[
    sample_metadata.index.str.contains(
        r"^WT_AL_",
        regex=True,
        na=False
    ),
    "Group"
] = "WT_AL"

sample_metadata.loc[
    sample_metadata.index.str.contains(
        r"^WT_RF_",
        regex=True,
        na=False
    ),
    "Group"
] = "WT_RF"

# Extract Zeitgeber Time:
# WT_AL_ZT02_A_RNA -> ZT02
sample_metadata["ZT"] = (
    sample_metadata
    .index
    .to_series()
    .str.extract(r"(ZT\d{2})", expand=False)
)

print("\nGroups detected:")
print(sample_metadata["Group"].value_counts(dropna=False))

print("\nZT values detected:")
print(
    sorted(
        sample_metadata["ZT"]
        .dropna()
        .unique()
    )
)


# ==============================================================================
# 5. Select WT_AL and WT_RF at matching time points
# ==============================================================================

print("\n==================================================")
print("5. SELECTING ANALYSIS SAMPLES")
print("==================================================")

# Find which time points are available in WT_AL.
wt_al_times = set(
    sample_metadata.loc[
        sample_metadata["Group"] == "WT_AL",
        "ZT"
    ].dropna()
)

# Find which time points are available in WT_RF.
wt_rf_times = set(
    sample_metadata.loc[
        sample_metadata["Group"] == "WT_RF",
        "ZT"
    ].dropna()
)

# Keep only ZT values that appear in BOTH groups.
common_zt = sorted(
    wt_al_times.intersection(wt_rf_times),
    key=lambda value: int(value.replace("ZT", ""))
)

print(f"WT_AL time points: {sorted(wt_al_times)}")
print(f"WT_RF time points: {sorted(wt_rf_times)}")
print(f"\nCommon time points: {common_zt}")

analysis_metadata = sample_metadata[
    sample_metadata["Group"].isin(["WT_AL", "WT_RF"])
    & sample_metadata["ZT"].isin(common_zt)
].copy()

# Make sure selected samples also exist in the expression matrix.
analysis_samples = [
    sample
    for sample in analysis_metadata.index
    if sample in counts_by_sample.columns
]

analysis_metadata = analysis_metadata.loc[analysis_samples]

counts_analysis_ensembl = counts_by_sample[analysis_samples].copy()

print("\nSamples selected:")
print(analysis_metadata["Group"].value_counts())

print("\nSamples by ZT and group:")
print(
    pd.crosstab(
        analysis_metadata["ZT"],
        analysis_metadata["Group"]
    )
)

print(
    f"\nTotal biological samples used for analysis: "
    f"{len(analysis_samples)}"
)


# ==============================================================================
# 6. Convert mouse Ensembl gene IDs to gene symbols
# ==============================================================================

print("\n==================================================")
print("6. CONVERTING ENSEMBL IDS TO MOUSE GENE SYMBOLS")
print("==================================================")

# Example:
# ENSMUSG00000000001.5 -> ENSMUSG00000000001
clean_ensembl = [
    str(gene_id).split(".")[0]
    for gene_id in counts_analysis_ensembl.index
]

print(
    f"Number of Ensembl IDs to query: "
    f"{len(clean_ensembl)}"
)

mg = mygene.MyGeneInfo()

query_results = mg.querymany(
    clean_ensembl,
    scopes="ensembl.gene",
    fields="symbol",
    species="mouse",
    verbose=False
)

# Build mapping:
# ENSMUSG00000000001 -> Gnai3
symbol_map = {}

for item in query_results:

    query = item.get("query")
    symbol = item.get("symbol")

    if query is not None and symbol is not None:
        if query not in symbol_map:
            symbol_map[query] = symbol


mapped_symbols = [
    symbol_map.get(ensembl_id, np.nan)
    for ensembl_id in clean_ensembl
]

number_mapped = pd.Series(mapped_symbols).notna().sum()
number_unmapped = pd.Series(mapped_symbols).isna().sum()

print(f"Successfully mapped IDs: {number_mapped}")
print(f"Unmapped IDs: {number_unmapped}")


# Add gene symbols as a temporary column.
counts_analysis = counts_analysis_ensembl.copy()
counts_analysis["GeneSymbol"] = mapped_symbols

# Remove genes for which no gene symbol was found.
counts_analysis = counts_analysis.dropna(
    subset=["GeneSymbol"]
)

# Some Ensembl IDs may map to the same official gene symbol.
# Combine those genes by summing their expression values.
counts_analysis = (
    counts_analysis
    .groupby("GeneSymbol")
    .sum()
)

print("\nExpression matrix after gene-symbol conversion:")
print(f"Genes: {counts_analysis.shape[0]}")
print(f"Samples: {counts_analysis.shape[1]}")


# ==============================================================================
# 7. Save cleaned data and metadata
# ==============================================================================

print("\n==================================================")
print("7. SAVING CLEANED DATA")
print("==================================================")

counts_analysis.to_csv(
    RESULTS_DIR / "analysis_expression_matrix.tsv",
    sep="\t"
)

analysis_metadata.to_csv(
    RESULTS_DIR / "analysis_metadata.tsv",
    sep="\t"
)

print(
    "Saved cleaned expression matrix to:",
    RESULTS_DIR / "analysis_expression_matrix.tsv"
)

print(
    "Saved analysis metadata to:",
    RESULTS_DIR / "analysis_metadata.tsv"
)


# ==============================================================================
# 8. Log2 transformation
# ==============================================================================

print("\n==================================================")
print("8. LOG TRANSFORMATION")
print("==================================================")

# log2(count + 1) prevents log(0).
log_counts = np.log2(counts_analysis + 1)

print(
    f"Log-transformed matrix shape: "
    f"{log_counts.shape}"
)


# ==============================================================================
# 9. Calculate per-gene median expression
# ==============================================================================

print("\n==================================================")
print("9. PER-GENE MEDIAN EXPRESSION")
print("==================================================")

gene_medians = log_counts.median(axis=1)

print(gene_medians.describe())

# Save values so they are reproducible and can be examined later.
gene_medians.to_csv(
    RESULTS_DIR / "gene_median_expression.tsv",
    sep="\t",
    header=["median_log2_expression"]
)


# ==============================================================================
# 10. Density plot
# ==============================================================================

print("\n==================================================")
print("10. CREATING DENSITY PLOT")
print("==================================================")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.kdeplot(
    gene_medians,
    fill=True,
    alpha=0.7
)

plt.title(
    "Density of Per-Gene Median Expression",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel(
    "Median Gene Expression [log2(count + 1)]"
)

plt.ylabel("Density")

plt.tight_layout()

density_path = RESULTS_DIR / "density_plot.png"

plt.savefig(
    density_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Density plot saved to: {density_path}")


# ==============================================================================
# 11. PCA
# ==============================================================================

print("\n==================================================")
print("11. PRINCIPAL COMPONENT ANALYSIS")
print("==================================================")

# Remove genes with zero variance.
# Such genes provide no information to PCA.
gene_variance = log_counts.var(axis=1)

pca_counts = log_counts.loc[
    gene_variance > 0
]

print(
    f"Genes used for PCA after removing "
    f"zero-variance genes: {pca_counts.shape[0]}"
)

# PCA expects:
# rows    = observations/samples
# columns = features/genes
#
# Our matrix is genes x samples, so transpose it.
pca_input = pca_counts.T

pca = PCA(
    n_components=2
)

pca_transformed = pca.fit_transform(
    pca_input
)

variance_explained = (
    pca.explained_variance_ratio_ * 100
)

print(
    f"PC1 explained variance: "
    f"{variance_explained[0]:.2f}%"
)

print(
    f"PC2 explained variance: "
    f"{variance_explained[1]:.2f}%"
)


# ==============================================================================
# 12. Create PCA dataframe
# ==============================================================================

pca_df = pd.DataFrame(
    pca_transformed,
    index=pca_input.index,
    columns=["PC1", "PC2"]
)

pca_df["Group"] = analysis_metadata.loc[
    pca_df.index,
    "Group"
]

pca_df["ZT"] = analysis_metadata.loc[
    pca_df.index,
    "ZT"
]

# Save PCA coordinates for later analysis.
pca_df.to_csv(
    RESULTS_DIR / "pca_coordinates.tsv",
    sep="\t"
)


# ==============================================================================
# 13. PCA plot colored by feeding group
# ==============================================================================

print("\n==================================================")
print("13. CREATING PCA PLOT")
print("==================================================")

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Group",
    s=90,
    alpha=0.85
)

plt.title(
    "PCA of Mouse Liver Gene Expression",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel(
    f"PC1 ({variance_explained[0]:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({variance_explained[1]:.2f}% variance)"
)

plt.legend(
    title="Feeding Regimen",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

pca_path = RESULTS_DIR / "pca_plot.png"

plt.savefig(
    pca_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"PCA plot saved to: {pca_path}")


# ==============================================================================
# 14. OPTIONAL DIAGNOSTIC PCA PLOT COLORED BY TIME
#
# This is useful for us to determine whether circadian time is strongly
# influencing the PCA. It is not a replacement for the required group-colored
# PCA plot.
# ==============================================================================

print("\n==================================================")
print("14. CREATING DIAGNOSTIC PCA PLOT BY ZT")
print("==================================================")

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="ZT",
    s=90,
    alpha=0.85
)

plt.title(
    "PCA of Mouse Liver Gene Expression by ZT",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel(
    f"PC1 ({variance_explained[0]:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({variance_explained[1]:.2f}% variance)"
)

plt.legend(
    title="Zeitgeber Time",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

pca_zt_path = RESULTS_DIR / "pca_plot_by_ZT.png"

plt.savefig(
    pca_zt_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Diagnostic PCA plot saved to: "
    f"{pca_zt_path}"
)


# ==============================================================================
# 15. Final summary
# ==============================================================================

print("\n==================================================")
print("ANALYSIS COMPLETE")
print("==================================================")

print(
    f"Original sequencing runs: "
    f"{counts_matrix.shape[1]}"
)

print(
    f"Biological samples in complete experiment: "
    f"{counts_by_sample.shape[1]}"
)

print(
    f"Biological samples used for WT_AL vs WT_RF analysis: "
    f"{counts_analysis.shape[1]}"
)

print(
    f"Mapped genes used in analysis: "
    f"{counts_analysis.shape[0]}"
)

print(
    f"Matching ZT time points: "
    f"{', '.join(common_zt)}"
)

print("\nGroup counts:")
print(
    analysis_metadata["Group"]
    .value_counts()
)

print("\nSamples by time point:")
print(
    pd.crosstab(
        analysis_metadata["ZT"],
        analysis_metadata["Group"]
    )
)

print("\nGenerated files:")

for path in sorted(RESULTS_DIR.iterdir()):
    print(f" - {path.name}")

print("\nDone.")