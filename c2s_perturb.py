#!/usr/bin/env python
"""
Script for in-silico perturbation of microglia scRNA-seq dataset
using Google's Cell2Sentence LLM (C2S).

This follows the C2S tutorials for fine-tuning on a new dataset
and using a custom prompt formatter for cell-type prediction.
"""
import os
import random
import numpy as np
import scanpy as sc
import anndata
import pandas as pd
from datasets import Dataset
from datetime import datetime
from transformers import TrainingArguments
import subprocess

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.prompt_formatter import PromptFormatter

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
import requests
import boto3

# ----------------------------------------------------------------------------
# Custom prompt formatter
# ----------------------------------------------------------------------------
class CustomPromptFormatter(PromptFormatter):
    def __init__(self, task_name, input_prompt, answer_template, top_k_genes):
        super().__init__()
        self.task_name = task_name
        self.input_prompt = input_prompt
        self.answer_template = answer_template
        self.top_k_genes = top_k_genes
        assert isinstance(top_k_genes, int) and top_k_genes > 0, "'top_k_genes' must be an integer > 0"

    def format_hf_ds(self, hf_ds):
        # Build prompt strings and responses for each sample in the HF dataset
        model_inputs_list = []
        responses_list = []
        for rec in hf_ds:
            # rec must contain: 'cell_sentence', 'tissue', 'organism', 'cell_type'
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
            'sample_type': [self.task_name] * len(model_inputs_list),
            'model_input': model_inputs_list,
            'response': responses_list
        }
        return Dataset.from_dict(ds_dict)

# ----------------------------------------------------------------------------
# Functions for C2S fine-tuning and perturbation analysis
# ----------------------------------------------------------------------------
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


def finetune_c2s(
    adata,
    save_dir,
    save_name,
    pretrained_model_path,
    top_k_genes=2000,
    max_eval_samples=500
):
    """
    Fine-tune a C2S CSModel on the provided AnnData.
    Returns the fine-tuned csmodel and its prompt_formatter.
    """
    # 1. Convert AnnData to Huggingface Arrow dataset
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=SEED,
        sentence_delimiter=' ',
        label_col_names=['cell_type', 'tissue', 'organism']
    )

    # Sample 10,000 indices
    num_cells = 10000
    total_cells = len(arrow_ds["cell_type"])
    sample_indices = np.random.choice(total_cells, size=num_cells, replace=False)

    # Subset Arrow dataset
    arrow_ds_subset = arrow_ds.select(sample_indices)

    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds_subset,
        vocabulary=vocabulary,
        save_dir=save_dir,
        save_name=save_name,
        dataset_backend='arrow'
    )

    # 2. Define and initialize the C2S model
    csmodel = cs.CSModel(
        model_name_or_path=pretrained_model_path,
        save_dir=save_dir,
        save_name=save_name
    )

    # 3. Training arguments
    train_args = TrainingArguments(
        output_dir=os.path.join(save_dir, 'finetune_output'),
        num_train_epochs=3,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        logging_steps=50,
        save_steps=200,
        eval_strategy='steps',
        fp16=True,
        gradient_checkpointing=True
    )

    # 4. Custom prompt formatter
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
        task_name='cell_type_pred_given_tissue',
        input_prompt=custom_input_prompt_template,
        answer_template=answer_template,
        top_k_genes=top_k_genes
    )

    # 5. Fine-tune
    csmodel.fine_tune(
        csdata=csdata,
        task='cell_type_pred_given_tissue',
        train_args=train_args,
        loss_on_response_only=False,
        top_k_genes=top_k_genes,
        max_eval_samples=max_eval_samples,
        prompt_formatter=prompt_formatter
    )
    return csmodel, prompt_formatter


def compute_top_variable_genes(adata, top_n=500, force_include_genes=None):
    """
    Identify top_n highly variable genes using Scanpy's method.
    """
    # 1. Filter low-quality cells
    sc.pp.filter_cells(adata, min_genes=200) 

    # 2. Filter lowly expressed genes
    sc.pp.filter_genes(adata, min_cells=3) 

    # 3. Mitochondrial gene filtering (optional for human)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # if using human Ensembl/HGNC
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]  # Remove cells with >5% mito counts

    # 4. Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)   # Normalize counts per cell

    # 5. Log-transform
    sc.pp.log1p(adata)

    # 6. Scale (zero mean, unit variance per gene)
    sc.pp.scale(adata, max_value=10)

    sc.pp.highly_variable_genes(
        adata,
        flavor='seurat_v3',
        n_top_genes=top_n,
        subset=False
    )
    hvg = adata.var_names[adata.var['highly_variable']].tolist()
    if force_include_genes:
        hvg = list(set(hvg).union(force_include_genes))

    return hvg


def perturbation_analysis(
    adata,
    csmodel,
    prompt_formatter,
    hvgs,
    home_cluster_label,
    tissue_label,
    organism_label
):
    """
    For each non-homeostatic cluster, generate baseline and perturbed gene lists,
    query the fine-tuned model, and record whether the perturbation moves
    the prediction to 'homeostatic'.
    """
    # Build expression DataFrame
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    df_expr = pd.DataFrame(
        X,
        index=adata.obs_names,
        columns=adata.var_names
    )
    avg_exp = df_expr.groupby(adata.obs['cluster']).mean()

    results = []
    for cluster in avg_exp.index:
        if cluster == home_cluster_label:
            continue
        exp_vals = avg_exp.loc[cluster]
        # baseline gene list
        base_genes = exp_vals.sort_values(ascending=False).head(prompt_formatter.top_k_genes).index.tolist()
        # baseline prompt
        baseline_sentence = ' '.join(base_genes)
        baseline_ds = Dataset.from_dict({
            'cell_sentence': [baseline_sentence],
            'tissue': [tissue_label],
            'organism': [organism_label],
            'cell_type': ['']
        })
        formatted_base = prompt_formatter.format_hf_ds(baseline_ds)
        baseline_pred = csmodel.predict(formatted_base['model_input'])[0]

        # perturb each gene
        for gene in hvgs:
            if gene in base_genes:
                pert_genes = [gene] + [g for g in base_genes if g != gene]
            else:
                pert_genes = [gene] + base_genes[:-1]
            sentence = ' '.join(pert_genes)
            ds = Dataset.from_dict({
                'cell_sentence': [sentence],
                'tissue': [tissue_label],
                'organism': [organism_label],
                'cell_type': ['']
            })
            formatted_ds = prompt_formatter.format_hf_ds(ds)
            pred = csmodel.predict(formatted_ds['model_input'])[0]
            results.append({
                'cluster': cluster,
                'gene': gene,
                'baseline_pred': baseline_pred,
                'pert_pred': pred,
                'promoted_to_homeo': int('homeostatic' in pred.lower())
            })
    return pd.DataFrame(results)

def push_to_github(results_csv):
    """Commit and push results to GitHub."""
    try:
        subprocess.run(['git', 'add', results_csv], check=True)
        subprocess.run(['git', 'commit', '-m', 'Add perturbation results'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("✓ Results pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print("❌ Git push failed:", e)

def shutdown_instance():
    instance_id = requests.get('http://169.254.169.254/latest/meta-data/instance-id').text
    ec2 = boto3.client('ec2', region_name='us-east-1')
    ec2.terminate_instances(InstanceIds=[instance_id])

def main():
    # User parameters (modify as needed)
    adata_path = '/home/ec2-user/c2s/ad_data.h5ad'
    pretrained_model_path = 'vandijklab/pythia-160m-c2s'
    save_dir = './c2s_finetune_microglia'
    save_name = 'microglia_homeostasis'
    home_cluster_label = 'MG0'  # replace with your homeostatic cluster ID or label
    tissue_label = 'microglia'
    organism_label = 'human'
    top_n_hvgs = 2000
    top_k_genes = 100
    force_include_genes = ['SALL1', 'SMAD4']

    os.makedirs(save_dir, exist_ok=True)

    # Load and prepare data
    adata = sc.read_h5ad(adata_path)
    adata = prepare_adata(adata, home_cluster_label, tissue_label, organism_label)

    # Fine-tune the C2S model
    csmodel, prompt_formatter = finetune_c2s(
        adata,
        save_dir,
        save_name,
        pretrained_model_path,
        top_k_genes=top_k_genes
    )

    # Identify highly variable genes
    hvgs = compute_top_variable_genes(adata, top_n=top_n_hvgs, force_include_genes=force_include_genes)

    # Perform perturbation analysis
    results_df = perturbation_analysis(
        adata,
        csmodel,
        prompt_formatter,
        hvgs,
        home_cluster_label,
        tissue_label,
        organism_label
    )

    # Save results
    results_csv = os.path.join(save_dir, 'perturbation_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Analysis complete. Results saved to {results_csv}")
    
    push_to_github(results_csv)
    # shutdown_instance()

if __name__ == '__main__':
    main()
