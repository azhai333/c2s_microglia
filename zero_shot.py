#!/usr/bin/env python
"""
Script for in-silico perturbation of microglia scRNA-seq dataset
using Google's Cell2Sentence LLM (C2S).

This version includes a zero-shot benchmark that evaluates how well
C2S can classify cluster-level profiles before fine-tuning.
"""
import os
import random
import numpy as np
import scanpy as sc
import anndata
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.prompt_formatter import PromptFormatter

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# Custom prompt formatter
# ----------------------------------------------------------------------------
class CustomPromptFormatter(PromptFormatter):
    def __init__(self, input_prompt, answer_template, top_k_genes):
        super().__init__()
        self.task_name = "cell_type_prediction"
        self.input_prompt = input_prompt
        self.answer_template = answer_template
        self.top_k_genes = top_k_genes

    def format_hf_ds(self, hf_ds):
        model_inputs_list = []
        responses_list = []
        for rec in hf_ds:
            inp = self.input_prompt.format(
                num_genes=len(rec['cell_sentence'].split()),
                organism=rec['organism'],
                tissue_type=rec['tissue'],
                cell_sentence=rec['cell_sentence']
            )
            out = self.answer_template.format(cell_type=rec['cell_type'])
            model_inputs_list.append(inp)
            responses_list.append(out)
        ds_dict = {
            'sample_type': ['cell_type_prediction'] * len(model_inputs_list),
            'model_input': model_inputs_list,
            'response': responses_list
        }
        return Dataset.from_dict(ds_dict)

# ----------------------------------------------------------------------------
# Benchmarking function for zero-shot accuracy
# ----------------------------------------------------------------------------
def run_zero_shot_benchmark(adata, csmodel, prompt_formatter, top_k_genes, tissue_label, organism_label):
    print("Running zero-shot classification benchmark...\n")

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    df_expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Averaging expression per cell type gives a stable, representative profile
    # of each class for a more controlled comparison
    avg_exp = df_expr.groupby(adata.obs['State']).mean()

    true_labels = []
    predicted_labels = []

    for label, expr in avg_exp.iterrows():
        top_genes = expr.sort_values(ascending=False).head(top_k_genes).index.tolist()
        sentence = ' '.join(top_genes)
        ds = Dataset.from_dict({
            'cell_sentence': [sentence],
            'tissue': [tissue_label],
            'organism': [organism_label],
            'cell_type': [label]  # ground truth for benchmark
        })
        formatted_ds = prompt_formatter.format_hf_ds(ds)
        pred = csmodel.predict(formatted_ds['model_input'])[0]

        true_labels.append(label)
        predicted_labels.append(pred)

        print(f"True: {label:20s} | Predicted: {pred}")

    results = pd.DataFrame({
        'true': true_labels,
        'pred': predicted_labels
    })
    accuracy = (results['true'].str.lower() == results['pred'].str.lower()).mean()
    print(f"\n✅ Zero-shot accuracy: {accuracy:.2%}")

    # Plot confusion matrix
    print("\nConfusion Matrix:")
    labels = sorted(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Zero-Shot Cell Type Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig("zero_shot_confusion_matrix.png")
    print("Confusion matrix saved to zero_shot_confusion_matrix.png")

    return results

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    adata_path = '/home/ec2-user/c2s/ad_data.h5ad'
    pretrained_model_path = 'vandijklab/C2S-Scale-Pythia-1b-pt'
    tissue_label = 'brain'
    organism_label = 'human'
    top_k_genes = 100

    adata = sc.read_h5ad(adata_path)

    custom_input_prompt_template = (
        """
Given below is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell.
Given this expression representation as well as the tissue which this cell originates from, your task is to give the cell type which this cell belongs to.
Tissue type: {tissue_type}
Cell sentence: {cell_sentence}.
The cell type corresponding to these genes is:"""
    )
    answer_template = "{cell_type}"

    prompt_formatter = CustomPromptFormatter(
        input_prompt=custom_input_prompt_template,
        answer_template=answer_template,
        top_k_genes=top_k_genes
    )

    save_dir = './c2s_zero_shot_microglia'
    save_name = 'microglia_homeostasis'

    csmodel = cs.CSModel(model_name_or_path=pretrained_model_path,save_dir=save_dir,
    save_name=save_name)
    
    run_zero_shot_benchmark(
        adata,
        csmodel,
        prompt_formatter,
        top_k_genes,
        tissue_label,
        organism_label
    )


if __name__ == '__main__':
    main()
