import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from argparse import Namespace
from utils.data_initializer import initialize
#from utils.data_initializer import initialize_crossstudy
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
import pandas as pd
from improve import framework as frm
from improve.metrics import compute_metrics
# Model-specifc imports
from model_utils.utils import extract_subset_fea
# [Req] Imports from preprocess and train scripts
from BiGDRP_preprocess_improve import preprocess_params

#from bigdrp.trainer import Trainer
#from utils.tuple_dataset import TupleMatrixDataset

# Just because the tensorflow warnings are a bit verbose
filepath = Path(__file__).resolve().parent


app_infer_params = []


model_infer_params = []


infer_params = app_infer_params + model_infer_params

req_preprocess_args = [ll["name"] for ll in model_train_params]

def load_predictions(folder, split, fold_mask, file_prefix='val_prediction_fold'):
    preds = []
    for i in range(5):
        x = pd.read_csv(folder+'/%s_%d.csv'%(file_prefix, i), index_col=0)
        x.index = x.index.astype(str)
        preds.append(x)
    
    if split=='lco': # leave cell out
        preds_df = pd.DataFrame()
        for i in range(5):
            preds_df = pd.concat([preds_df, preds[i]])
        preds_df = preds_df.sort_index()

    else:
        if fold_mask is None:
            print("fold mask should not be None when loading for leave-pairs-out")

        drugs = preds[0].columns

        if len(drugs) > len(fold_mask.columns):
            drugs = list(fold_mask.columns)

        samples = set()
        for i in range(5):
            samples = samples.union(set(preds[i].index))
        samples = sorted(list(samples)) # fix the order

        preds_df = pd.DataFrame(np.zeros((len(samples), len(drugs))), index=samples, columns=drugs)
        for i in range(5):
            temp = preds[i][drugs].replace(np.nan, 0)
            missing = set(samples) - set(temp.index) # the fold doesn't have these samples
            if len(missing) > 0:
                # print('fold %d does not have samples: '%i, missing)
                for m in missing:
                    temp.loc[m] = np.zeros(len(drugs))

            fm = ((fold_mask == i)*1).loc[samples, drugs]
            preds_df += temp.loc[samples, drugs]*fm # make sure that only those in the fold are added

    return preds, preds_df


def get_per_drug_metric(df, y, y_bin=None):
    """
        df: DataFrame containing the predictions with drug as columns and CCLs as rows
        y: DataFrame containing the true responses
        y_bin: DataFrame containing the true responses in binary
    """

    y0 = y.replace(np.nan, 0)
    drugs = df.columns
    if y_bin is not None:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE', 'R2', 'AUROC'])
        calc_auroc = True
    else:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE', 'R2'])

    for drug in drugs:
        mask = y0[drug].values.nonzero()
        prediction = df[drug].values[mask]
        true_label = y[drug].values[mask]
        
        rmse = np.sqrt(((prediction-true_label)**2).mean())
        scc = spearmanr(true_label, prediction)[0]
        pcc = pearsonr(true_label, prediction)[0]
        r2 = r2_score(true_label, prediction)
        
        if calc_auroc:
            true_bin = y_bin[drug].values[mask]
            true_bin = true_bin.astype(int)
            if true_bin.mean() != 1:
                try:
                    auroc = roc_auc_score(true_bin, prediction)
                except ValueError:
                    pass
            else:
                auroc = np.nan
            metrics.loc[drug] = [scc,pcc,rmse,r2,auroc]
        else:
            metrics.loc[drug] = [scc,pcc,rmse,r2]

    return metrics

def get_per_drug_fold_metric(df, y, fold_mask, y_bin=None):
    """
        df: DataFrame containing the predictions with drug as columns and CCLs as rows
        y: DataFrame containing the true responses
        fold_mask: DataFrame containing the designated folds
        y_bin: DataFrame containing the true responses in binary
    """

    drugs = df.columns

    if y_bin is not None:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE', 'AUROC'])
        calc_auroc = True
    else:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE'])

    for drug in tqdm(drugs):

        temp = np.zeros((5, len(metrics.columns)))
        for i in range(5):
            mask = ((fold_mask[drug] == i)*1).values.nonzero()
            prediction = df[drug].values[mask]
            true_label = y[drug].values[mask]

            rmse = np.sqrt(((prediction-true_label)**2).mean())
            scc = spearmanr(true_label, prediction)[0]
            pcc = pearsonr(true_label, prediction)[0]
            r2 = r2_score(true_label, prediction)
            
            if calc_auroc:
                true_bin = y_bin[drug].values[mask]
                true_bin = true_bin.astype(int)
                if true_bin.mean() != 1:
                    auroc = roc_auc_score(true_bin, prediction)
                else:
                    auroc = np.nan
                temp[i] = [scc,pcc,rmse,r2,auroc]
            else:
                temp[i] = [scc,pcc,rmse,r2]

        metrics.loc[drug] = temp.mean(axis=0)
    return metrics


def str2Class(str):
    """ Get model class from model name (str) """
    return globals()[str]()

def load_BiGDRP(params, modelpath, device):
    """ Load GraphDRP """
    if modelpath.exists() == False:
        raise Exception(f"ERROR ! modelpath not found {modelpath}\n")
    model = BiGDRP()
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    return model


def launch(args):
    frm.create_outdir(outdir=params["infer_outdir"])
    print("\nTest data:")
    print(f"train_ml_data_dir: {params['train_ml_data_dir']}")
    print(f"test_batch: {params['test_batch']}")
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    test_data_fname = test_data_fname.split(params["data_format"])[0]
    print("\nTest data:")
    print(f"train_ml_data_dir: {params['train_ml_data_dir']}")
    print(f"test_batch: {params['test_batch']}")
#    test_loader = build_GraphDRP_dataloader(params["test_ml_data_dir"],
#                                             test_data_fname,
#                                             params["test_batch"],
#                                             shuffle=False)
    device = determine_device(params["cuda_name"])
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]                                                         
    model = load_BiGDRP(params, modelpath, device)
    model.eval()
    test_true, test_pred = predicting(model, device, data_loader=test_loader)                               
    frm.store_predictions_df(
        params, y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )
    test_scores = frm.compute_performace_scores(
        params, y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores
        
    
def run(gParameters):
    print("In Run Function:\n")
#    args = candle.ArgumentStruct(**gParameters)
    # Call launch() with specific model arch and args with all HPs
    scores = launch(args)
    print('printing scores ...')
    print(scores)
    # Supervisor HPO
    with open(args.output_dir + "/scores_infer.json", "w", encoding="utf-8") as f:
        json.dump(scores.to_json(), f, ensure_ascii=False, indent=4)
    return scores

    
def initialize_parameters():
    params = frm.initialize_parameters(
        filepath,
        default_model="BiG_DRP_model.txt",
        additional_definitions=model_train_params,
        required=req_preprocess_args,
    )
    return params

   
def run_cross_study(params, study, gene_expression, label_matrix, test_data_tup):
    ns = Namespace(**params)
    FLAGS = ns
    cell_lines = pd.read_csv(gene_expression)
    cell_lines =  cell_lines.T
    cell_lines.columns =  cell_lines.iloc[0]
    cell_lines = cell_lines[1:]
    label_matrix = pd.read_csv(label_matrix)
    label_matrix =  label_matrix.T
    label_matrix.columns =  label_matrix.iloc[0]
    label_matrix = label_matrix[1:]
    run_infer(study, label_matrix, test_data_tup)


def run_infer(study, study_label, study_matrix):
    study_drug_feats, study_cell_lines, study_labels, study_label_matrix, standardize= initialize_crossstudy(study,drug_feat,
                                                                                                     normalize_response, 
                                                                                                     study_label, 
                                                                                                     GENE_EXPRESSION_FILE, 
                                                                                                     study_matrix,
                                                                                                     DRUG_DESCRIPTOR_FILE,
                                                                                                     MORGAN_FP_FILE)


    study_tuples = pd.read_csv(study_label)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    study_label_matrix = study_label_matrix.replace(np.nan, 0)
    study_drug_list = list(study_drug_feats.index)
    study_samples = list(study_labels['cell_line'].unique())
    study_tuples = study_tuples[['drug', 'cell_line', 'response']]
    study_tuples = reindex_tuples(study_tuples, study_drug_list, study_samples)
    study_x = study_cell_lines.loc[study_samples].values
    study_y = study_label_matrix.loc[study_samples].values
    ss = StandardScaler()
    study_x = StandardScaler().fit_transform(study_x)
    study_y = StandardScaler().fit_transform(study_y)
    graph_sampler = MultiLayerFullNeighborSampler(2)
    study_network = create_network(study_tuples, network_percentile)
    study_drug_feats_tensor = torch.FloatTensor(study_drug_feats.values).to(device)
    study_cell_lines_tensor = torch.FloatTensor(study_cell_lines.values).to(device)
    study_network = study_network.to(device)
    study_network.ndata['features'] = {'drug': study_drug_feats_tensor, 'cell_line': study_cell_lines_tensor}
    _,_, study_blocks = graph_sampler.sample_blocks(study_network, {'drug': range(len(study_drug_feats)), 
                                                              'cell_line': range(len(study_cell_lines))})           
    dtudy_blocks = [b.to(device) for b in study_blocks]
    hyp = hyperparams.copy()
    n_genes = study_cell_lines.shape[1]
    n_drug_feats = study_drug_feats.shape[1]
    #n_cell_feats = cell_lines.shape[1]
    n_cell_feats = study_cell_lines.shape[1]
    drug_feats_tensor = torch.FloatTensor(study_drug_feats.values)
    cell_line_feats_tensor = torch.FloatTensor(study_x)
    study_data = TensorDataset(torch.FloatTensor(study_x),torch.FloatTensor(study_y))
    study_loader = DataLoader(study_data, batch_size=hyp['batch_size'], shuffle=False)
    model = BiGDRP(n_genes, n_cell_feats, n_drug_feats, study_network.etypes, hyp)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    drug_enco = model.get_drug_encoding(study_blocks, study_drug_feats_tensor, study_cell_lines_tensor)
    study_results = []
    study_ys = []
    with torch.no_grad():
        for (x, y) in study_loader:
            x = x.to(device)
            y = y.to(device)
            drug_encoding = drug_enco.to(device)
            pred = model.predict_response_matrix(x, drug_encoding)
            study_ys.append(y.cpu().detach().numpy())
            study_results.append(pred.cpu().detach().numpy())
            
    study_ys = np.concatenate(study_ys)
    study_results = np.concatenate(study_results)
    min_rows = min(study_ys.shape[0], study_results.shape[0])
    study_results = study_results[:min_rows]
    study_ys = study_ys[:min_rows]
    study_r2 = r2_score(study_ys, study_results)
    print(study_r2)


def main(args):
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        # default_model="lgbm_params_ws.txt",
        # default_model="lgbm_params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_params,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")

if __name__ == "__main__":
    main(sys.argv[1:])
#def candle_main(ANL=True):
#    params = initialize_parameters()
#    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
#    data_dir = os.environ['CANDLE_DATA_DIR'] + "/Data/"
#    cross_study_dir = data_dir + "/" + params['cross_study_dir']
#    data_type_list = params['data_type'].split(',')
#    gene_expression = data_dir + "/drp-data/grl-preprocessed/sanger_tcga/BiG_DRP_fpkm.csv"
#    label_matrix = data_dir + "/drp-data/grl-preprocessed//drug_response/BiG_DRP_data_cleaned.csv"
#    if ANL:
#        for dt in data_type_list:
#            print(dt)
#            test_data_tup = cross_study_dir + "/" + dt + "_tuples_test.csv"
#            print(test_data_tup)
#            test_data_cleaned = cross_study_dir + "/" + dt + "_cleaned_test.csv"
#            assert(os.path.isfile(test_data_tup))
#            cross_study_scores = run_cross_study(params, dt, gene_expression, test_data_cleaned, test_data_tup)
#    else:
#        run_infer(study, study_label, study_matrix)
#        print("Done inference.")
#
#    
#if __name__ == "__main__":
#    candle_main()
