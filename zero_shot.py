# Python built-in libraries
import os
import pickle
import random
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np
from tqdm import tqdm

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data

def prepare_adata(adata, home_cluster_label, tissue_label, organism_label):
    """
    Annotate AnnData for C2S: define cell_type, tissue, organism in .obs
    """
    # Map clusters to cell_type labels
    adata.obs['cell_type'] = adata.obs['State'].apply(
        lambda x: 'homeostatic' if x == home_cluster_label else f'cluster_{x}'
    )
    adata.obs['tissue'] = tissue_label
    adata.obs['organism'] = organism_label
    return adata

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

home_cluster_label = 'MG0'  # replace with your homeostatic cluster ID or label
tissue_label = 'microglia'
organism_label = 'human'

DATA_PATH = "/home/ec2-user/c2s/ad_data.h5ad"

adata = anndata.read_h5ad(DATA_PATH)

adata = prepare_adata(adata, home_cluster_label, tissue_label, organism_label)

adata.obs = adata.obs[["cell_type", "tissue", "organism"]]

adata_obs_cols_to_keep = adata.obs.columns.tolist()

# Create CSData object
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=SEED, 
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)

# Define CSModel object
cell_type_prediction_model_path = "vandijklab/C2S-Scale-Pythia-1b-pt"
save_dir = "./c2s_zero_shot_microglia"
save_name = "zero_shot_model"
csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)

c2s_save_dir = "./c2s_zero_shot_microglia"  # C2S dataset will be saved into this directory
c2s_save_name = "zero_shot_data"  # This will be the name of our C2S dataset on disk

# Sample 10,000 indices
num_cells = 10000
total_cells = len(arrow_ds["cell_type"])
sample_indices = np.random.choice(total_cells, size=num_cells, replace=False)

# Subset Arrow dataset
arrow_ds_subset = arrow_ds.select(sample_indices)

# Wrap in CSData object
csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds_subset, 
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)

predicted_cell_types = predict_cell_types_of_data(
    csdata=csdata,
    csmodel=csmodel,
    n_genes=200
)

# Collect predictions and ground truths
all_preds = []
all_labels = []

for model_pred, gt_label in zip(predicted_cell_types, arrow_ds["cell_type"]):
    # Remove trailing period if present
    if model_pred.endswith('.'):
        model_pred = model_pred[:-1]
    all_preds.append(model_pred)
    all_labels.append(gt_label)

# Compute accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"\nOverall accuracy: {accuracy:.4f}")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=np.unique(all_labels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_labels))

# Plot
plt.figure(figsize=(10, 8))
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
