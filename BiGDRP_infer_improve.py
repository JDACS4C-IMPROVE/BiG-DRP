import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from argparse import Namespace
from utils.utils import mkdir, reindex_tuples, moving_average, reset_seed, create_fold_mask
from utils.data_initializer import initialize
#from utils.data_initializer import initialize_crossstudy
from utils.network_gen import create_network
import json
from pathlib import Path
from typing import Dict
import torch
#from torch_geometric.data import DataLoader
from improve import framework as frm
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
from  bigdrp.model import BiGDRP
#import train as bmk
#import candle
import torch
from torch.utils.data import DataLoader, TensorDataset
from BiGDRP_train_improve import metrics_list, model_train_params
import sys
#from BiGDRP.model import BiGDRP
from bigdrp.trainer import Trainer
import pandas as pd
from improve import framework as frm
from improve.metrics import compute_metrics
# Model-specifc imports
from model_utils.utils import extract_subset_fea
# [Req] Imports from preprocess and train scripts
from BiGDRP_preprocess_improve import preprocess_params
from BiGDRP_train_improve import create_dataset_anl
from BiGDRP_train_improve import metrics_list, model_train_params

# Just because the tensorflow warnings are a bit verbose
filepath = Path(__file__).resolve().parent
app_infer_params = []
model_infer_params = []
infer_params = app_infer_params + model_infer_params



def load_trainer(n_genes, cell_lines, drug_feats, network, hyperparams, filepath):
    drug_feats_tensor = torch.tensor(drug_feats.values, dtype=torch.float32)
    cell_lines_tensor = torch.tensor(cell_lines.values, dtype=torch.float32)
    trainer = Trainer(n_genes, cell_lines_tensor, drug_feats_tensor, network, hyperparams)
    trainer.load_state_dict(torch.load(filepath))
    return trainer


def run_infer(percentile, drug_feats, cell_lines, labels, label_matrix, normalizer, params):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # import ipdb; ipdb.set_trace()

    frm.create_outdir(outdir=params["infer_outdir"])
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    learning_rate = params['learning_rate']
    epoch = params['epochs']
    infer_batch_size = params['infer_batch_size']

    train_tuples = labels.loc[labels['cl_fold'] == 1]
    train_samples = list(train_tuples['cell_line'].unique())
    train_x = cell_lines.loc[train_samples].values
    train_y = label_matrix.loc[train_samples].values
    drug_list = list(drug_feats.index)
    test_tuples = labels.loc[labels['cl_fold'] == 3]
    test_samples = list(test_tuples['cell_line'].unique())
    test_x = cell_lines.loc[test_samples].values
    test_y = label_matrix.loc[test_samples].values
    n_genes = cell_lines.shape[1]
    train_tuples = train_tuples[['drug', 'cell_line', 'response']]
    train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds
    train_x, test_x = normalizer(train_x, test_x)
#    print(train_tuples, percentile)
    network = create_network(train_tuples, percentile)
    hyperparams = {
        'learning_rate': learning_rate,
        'num_epoch': epoch,
        'batch_size': infer_batch_size,
        'common_dim': 512,
        'expr_enc': 1024,
        'conv1': 512,
        'conv2': 512,
        'mid': 512,
        'drop': 1}
    filepath = params['infer_model_dir'] + "/results/model_weights_fold_anl.pt"  
    model = load_trainer(n_genes, cell_lines, drug_feats, network, hyperparams, filepath)
    test_data = TensorDataset(torch.FloatTensor(test_x))
    test_data = DataLoader(test_data, batch_size=infer_batch_size, shuffle=False)
    drug_enc = model.get_drug_encoding().cpu().detach().numpy()
    prediction_matrix = model.predict_matrix(test_data, drug_encoding=torch.Tensor(drug_enc))
    print(prediction_matrix)


def launch(params, drp_params):
    frm.create_outdir(outdir=params["infer_outdir"])
    print("\nTest data:")
    print(f"infer_ml_data_dir: {params['infer_ml_data_dir']}")
#    print(f"test_batch: {params['test_batch']}")
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    test_data_fname = test_data_fname.split(params["data_format"])[0]
 #    test_loader = build_GraphDRP_dataloader(params["test_ml_data_dir"],
#                                             test_data_fname,
#                                             params["test_batch"],
#                                             shuffle=False)
#    device = determine_device(params["cuda_name"])
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"])
    ns = Namespace(**drp_params)
    FLAGS = ns
    tuples_label_fold_out = params['infer_ml_data_dir'] + "/" +  params['tuples_label_fold_out']
    expression_out = params['infer_ml_data_dir'] + "/" +  params['expression_out']
    data_cleaned_out = params['infer_ml_data_dir'] + "/" +  params['data_cleaned_out']
    descriptor_out = params['infer_ml_data_dir'] + "/" +  params['descriptor_out']
    morgan_out = params['infer_ml_data_dir'] + "/" +  params['morgan_out']    
    drug_feats, cell_lines, labels, label_matrix, normalizer = initialize(FLAGS,
                                                                          tuples_label_fold_out,
                                                                          expression_out,
                                                                          data_cleaned_out,
                                                                          descriptor_out,
                                                                          morgan_out)
    run_infer(FLAGS.network_percentile, drug_feats, cell_lines, labels, label_matrix, normalizer, params)
#    print(labels)FLAGS.network_percentile
#    print(lab


def main(args):
    additional_definitions = preprocess_params + model_train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="BiG_DRP_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    drp_params = dict((k, params[k]) for k in ('descriptor_out', 'expression_out','morgan_out','data_cleaned_out',
                                               'labels', 'tuples_label_fold_out', 'infer_ml_data_dir',
                                               'infer_model_dir', 'infer_outdir',  
                                               'dataroot', 'drug_feat',
                                               'model_outdir', 'mode', 'network_percentile',
                                               'normalize_response', 'seed',
                                               'split', 'weight_folder', 'epochs',
                                               'infer_batch_size', 'learning_rate'))
    test_scores = launch(params, drp_params)
    print("\nFinished model inference.")

if __name__ == "__main__":
    main(sys.argv[1:])

