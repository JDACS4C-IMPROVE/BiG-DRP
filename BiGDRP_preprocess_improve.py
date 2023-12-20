import sys
#import candle
import os
import json
import shutil
from typing import Dict
from typing import List
from pathlib import Path
from json import JSONEncoder
from utils.utils import mkdir
#from Big_DRP_train import main
os.environ['NUMEXPR_MAX_THREADS']='6'
from pathlib import Path
import numexpr as ne
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from torch.utils.data import DataLoader, TensorDataset
from dgl.dataloading import MultiLayerFullNeighborSampler
import random
import torch
from torch.utils.data import Dataset
import dgl
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import HeteroGraphConv, GraphConv
import os
import argparse
import time
from scipy.stats import pearsonr, spearmanr
from collections import deque
from improve import framework as frm
# from improve import dataloader as dtl  # This is replaced with drug_resp_pred
from improve import drug_resp_pred as drp  # some funcs from dataloader.py were copied to drp
from sklearn.metrics import r2_score
#import improve_utils
#file_path = os.path.dirname(os.path.realpath(__file__))
filepath = Path(__file__).resolve().parent
required=None
additional_definitions=None


# This should be set outside as a user environment variable
#os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/BiG-DRP'

#parent path
fdir = Path('__file__').resolve().parent
#source = "csa_data/raw_data/splits/"
auc_threshold=0.5


drp_preproc_params = [
    {"name": "x_data_canc_files",  # app;                                                                                            
     #"nargs": "+",                                                                                                                 
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "x_data_drug_files",  # app;                                                                                            
     #"nargs": "+",                                                                                                                 
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "y_data_files",  # imp;                                                                                                 
     #"nargs": "+",                                                                                                                 
     "type": str,
     "help": "List of output files.",
    },
    {"name": "expression_out",
     "default" : "BiG_DRP_fpkm.csv",
      "type": str,                                                                                                                  
      "help": "Data set to preprocess.",                                                                                           
     },                                                                                 
    {"name": "split_id",                                                                                                           
      "type": int,                                                                                                                  
      "default": 0,                                                                                                                 
      "help": "ID of split to read. This is used to find training/validation/testingpartitions and read lists of data samples to use for preprocessing.",
     },                                                                                                                             
    # {"name": "response_file",                                                                                                      
    #  "type": str,                                                                                                                  
    #  "default": "response.tsv",                                                                                                    
    #  "help": "File with response data",             
    # },                                                           
    {"name": "canc_col_name",  # app;                                                  
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",  # app;                                                                          
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
    # {"name": "gene_system_identifier",  # app;                                                                                     
    #  "nargs": "+",                                                                                                                
    #  "type": str,                                                                                                                  
    #  "help": "Gene identifier system to use. Options: 'Entrez', 'Gene_Symbol',\                                                    
    #          'Ensembl', 'all', or any list combination.",                                                                          
    # },
]
# gdrp_data_conf = []  # replaced with model_conf_params + drp_conf_params                                                          
# preprocess_params = model_conf_params + drp_conf_params                                                                           
preprocess_params = drp_preproc_params
req_preprocess_args = [ll["name"] for ll in preprocess_params]  # TODO: it seems that all args specifiied to be 'req'. Why?          
req_preprocess_args.extend(["y_col_name", "model_outdir"])    

def initialize_parameters():
    params = frm.initialize_parameters(
        filepath,
        default_model="BiG_DRP_model.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    return params

def mv_file(download_file, req_file):
    if os.path.isfile(req_file):
        pass
    else:
        shutil.move(download_file, req_file)


def raw_data_available(params: Dict) -> frm.DataPathDict:
    """                                                                                                                             
    Sweep the expected raw data folder and check that files needed for cross-study analysis (CSA) are available.                         :params: Dict params: Dictionary of parameters read                                                                                  :return: Path to directories requested stored in dictionary with str key str and Path value.                                   
    :rtype: DataPathDict                                                                                                             
    """
    # Expected                                                                                                                     
    # raw_data -> {splits, x_data, y_data}                                                                                     
    # Make sure that main path exists                                                                                            
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"])
    if mainpath.exists() == False:
        raise Exception(f"ERROR ! {mainpath} not found.\n")

    inpath = mainpath / params["raw_data_dir"]
    print(inpath)
    if inpath.exists() == False:
        raise Exception(f"ERROR ! {inpath} not found.\n")

    x_data_dir = inpath / params["x_data_dir"]
    y_data_dir = inpath / params["y_data_dir"]
    splits_dir = inpath / "splits"
    #xpath = frm.check_path_and_files(x_data_dir, params["x_data_files"], inpath)
    ypath = frm.check_path_and_files(y_data_dir, params["y_data_files"][0], inpath)
    spath = frm.check_path_and_files('raw_data/splits', [], inpath)
    return {"y_data_path": ypath, "splits_path": spath}
#    return {"x_data_path": xpath, "y_data_path": ypath, "splits_path": spath}


def preprocess_param_inputs(params):
    supplementary_dir = os.path.join(os.environ['IMPROVE_DATA_DIR'], 'BiG-DRP/supplementary/')
    preprocessed_dir = os.path.join(os.environ['IMPROVE_DATA_DIR'], 'BiG-DRP/preprocessed/')
    params = frm.build_paths(params)  
#    processed_outdir = frm.create_outdir(params)
    drug_synonym_file = params['train_ml_data_dir'] + "/" + params['drug_synonyms']
    gene_expression_file = params['train_ml_data_dir'] + "/" + params['expression_out']
    ln50_file = params['train_ml_data_dir'] + "/" + params['data_file']
    model_label_file = params['train_ml_data_dir'] + "/" + params['binary_file']
    tcga_file =  supplementary_dir + params['tcga_file']
    data_bin_cleaned_out = params['train_ml_data_dir'] + '/' +params['data_bin_cleaned_out']
    data_cleaned_out = params['train_ml_data_dir'] + '/' + params['data_cleaned_out']
    data_tuples_out = params['train_ml_data_dir'] + '/' + params['data_tuples_out']
    tuples_label_fold_out = params['train_ml_data_dir'] + '/' + params['labels']
    smiles_file = params['train_ml_data_dir'] + '/' +params['smiles_file']
    params['data_bin_cleaned_out'] = data_bin_cleaned_out
    params['data_input'] = params['train_ml_data_dir'] + "/" + params['data_file']
    params['binary_input'] = params['train_ml_data_dir'] + "/" + params['binary_file']
    params['drug_out'] = params['train_ml_data_dir'] + '/' + params['drugset']
    params['fpkm_file'] = gene_expression_file
    params['descriptor_out'] = params['train_ml_data_dir'] + "/" + params['descriptor_out'] 
    params['morgan_data_out'] = params['train_ml_data_dir'] + "/" + params['morgan_out']
    params['model_label_file'] = model_label_file
    params['smiles_file'] =  smiles_file
    params['model_label_file'] = model_label_file
    params['tuples_label_out'] = params['train_ml_data_dir'] + "/" + params['data_tuples_out']
    params['tuples_label_fold_out'] = params['train_ml_data_dir'] + "/" + params['labels']
    params['tcga_file'] = tcga_file
    params['raw_data_dir'] = '.'
    params['x_data_dir'] = 'raw_data/x_data'
    params['y_data_dir'] = 'raw_data/y_data'
    params['dataroot'] = os.environ['IMPROVE_DATA_DIR']
    params['folder'] = params['model_outdir']
    params['outroot'] = params['model_outdir']
    params['network_perc'] = params['network_percentile']
    params['drug_feat'] = params['drug_feat']
    params['drug_synonyms'] = drug_synonym_file
    params['data_bin_cleaned_out'] = data_bin_cleaned_out
    params['data_cleaned_out'] = data_cleaned_out
    params['data_tuples_out'] = data_tuples_out
    params['tuples_label_fold_out'] = tuples_label_fold_out
    return(params)


def check_data_available(params: Dict) -> frm.DataPathDict:
    """                                                                                                                              
    Sweep the expected input paths and check that raw data files needed for preprocessing are available.                                                                                                            
    :params: Dict params: Dictionary of parameters read                                                                          
    :return: Path to directories requested stored in dictionary with str key str and Path value.                                 
    :rtype: DataPathDict                                                                                                             
    """
    # Check that raw data is available                                                                                               
    # Expected                                                                                                                       
    # raw_data -> {splits, x_data, y_data}                                                                                           
    ipathd = raw_data_available(params)
    # Create output directory. Do not complain if it exists.                                                                         
    opath = Path(params["model_outdir"]) # this was originally called ml_data                                                        
    os.makedirs(opath, exist_ok=True)
    # Return in DataPathDict structure
    inputdtd = {"y_data_path": ipathd["y_data_path"],"splits_path": ipathd["splits_path"]}
    outputdtd = {"preprocess_path": opath}
    return inputdtd, outputdtd
    
    
#def download_author_data(params, data_dir):
#    print("downloading file: %s"%params['data_url'])
#    data_download_filepath = candle.get_file(params['original_data'], params['data_url'],
#                                             datadir = data_dir,
#                                             cache_subdir = None)
#    return(params)


class MyEncoder(JSONEncoder):
    def default(self, obj):
        return obj.__dict__ 


def convert_to_binary(x):
    if not np.isnan(x):
        if x > auc_threshold:
            return "S"
        else:
            return "R"
    else:
        return ""


def create_big_drp_data(df, split_type, params):
    metric = params['metric']
    rs_df = df[['improve_sample_id','improve_chem_id',metric]]
    rs_df = df.pivot_table(index='improve_chem_id', columns='improve_sample_id', values=metric)
#    rs_df = df.pivot_table(index='improve_chem_id', columns='improve_sample_id', values=metric, aggfunc='mean')

    rs_df = rs_df.reset_index()
    rs_tdf = rs_df.set_index("improve_chem_id")
    rs_tdf = rs_tdf.T
    rs_binary_df = rs_tdf.applymap(convert_to_binary)
    rep = len(rs_binary_df.columns)
    rs_binary_df.index.names = ['compounds']
    thesholds = np.repeat([auc_threshold],rep)
    thesholds = list(thesholds)
    rs_binary_df.loc['threshold'] = thesholds
    rs_binary_df = rs_binary_df.reset_index()
    rs_binary_df = rs_binary_df.apply(np.roll, shift=1)
    binary_file = 'binary_' + split_type + "_file"
    binary_input= params['train_ml_data_dir'] + "/" + params[binary_file]
    print("writing out file {0}".format(binary_input))
    rs_binary_df.to_csv(binary_input, index=None)
    rs_tdf = rs_tdf.reset_index()
    rs_tdf = rs_tdf.rename({'compounds': "sample_names"},axis=1)
    ic50_file = 'data_' + split_type + "_file"    
    data_input = params['train_ml_data_dir'] + "/" + params[ic50_file]
    print("writing out file {0}".format(data_input))
    rs_tdf.to_csv(data_input, index_label="improve_id")

    
#def create_data_inputs(params, data_dir):
#    data_input = params['data_input']
#    binary_input = params['binary_input']
#    train_data_type = params['train_data_type']
#    metric = params['metric']
#    auc_threshold = params['auc_threshold']
#    data_type =  params['train_data_type']
#    rs = improve_utils.load_single_drug_response_data(source=data_type, split=0,
#                                                      split_type=["train", "val", "test"],
#                                                      y_col_name=metric)

#    rs_train = improve_utils.load_single_drug_response_data(source=data_type, split=0,
#                                                            split_type=["train"],
#                                                            y_col_name=metric)
    
#    rs_val = improve_utils.load_single_drug_response_data(source=data_type, split=0,
#                                                          split_type=["val"],
#                                                          y_col_name=metric)
    
#    rs_test = improve_utils.load_single_drug_response_data(source=data_type, split=0,
#                                                           split_type=["test"],
#                                                           y_col_name=metric)
#    create_big_drp_data(rs_train, data_dir, "train", params)
#    create_big_drp_data(rs_val, data_dir, "val", params)
#    create_big_drp_data(rs_test, data_dir, "test", params)    
    

    
def cross_study_test_data(params):
    data_input = params['data_input']
    binary_input = params['binary_input']
    data_type = params['data_type']
    metric = params['metric']
    auc_threshold = params['auc_threshold']
    cross_study_dir = params['cross_study']
    metric = params['metric']
    auc_threshold = params['auc_threshold']
    cross_study_list =  data_type.split(',')
    
    for i in cross_study_list:
        data_out = cross_study_dir + '/' + i + "_test.csv"
        binary_out =  cross_study_dir + '/' + i + "binary_test.csv"
        rs = improve_utils.load_single_drug_response_data(source=i, split=0,
                                                      split_type=["test"],
                                                      y_col_name=metric)
        rs = rs.drop_duplicates()
        rs = rs.reset_index(drop=True)
        rs = rs.groupby(['improve_chem_id', 'improve_sample_id']).mean().reset_index()
        rs_df = rs.pivot(index='improve_chem_id', columns='improve_sample_id', values=metric)
        rs_df = rs_df.reset_index()
        rs_tdf = rs_df.set_index("improve_chem_id")
        rs_tdf = rs_tdf.T
        rs_binary_df = rs_tdf.applymap(convert_to_binary)
        rep = len(rs_binary_df.columns)
        rs_binary_df.index.names = ['compounds']
        thesholds = np.repeat([auc_threshold],rep)
        thesholds = list(thesholds)
        rs_binary_df.loc['threshold'] = thesholds
        rs_binary_df = rs_binary_df.reset_index()
        rs_binary_df = rs_binary_df.apply(np.roll, shift=1)
        rs_binary_df.to_csv(binary_out, index=None)
        rs_tdf = rs_tdf.reset_index()
        rs_tdf = rs_tdf.rename({'compounds': "sample_names"},axis=1)
        rs_tdf.to_csv(data_out, index_label="improve_id")

        
def process_expression(df, params):
#    ge = improve_utils.load_gene_expression_data(gene_system_identifier="Ensembl")
    tcga_file = params['tcga_file']
    expression_out = params['fpkm_file']
    expression_tdf = df.dropna()
    expression_tdf = expression_tdf.set_index('improve_sample_id')
    expression_threshold = np.floor(0.1*len(expression_tdf))
    print('expression threshold is:')
    print(expression_threshold)
#    print((expression_df.astype(float) > 1).sum())
    to_keep = (expression_tdf.astype(float) > 1).sum() > expression_threshold
#    print(to_keep)
    f_log = (expression_tdf.loc[:,to_keep].astype(float) +1).apply(np.log2)
    f_log = f_log.loc[:, f_log.std() > 0.1]
    tcga = pd.read_csv(tcga_file, index_col=0)
    common = tcga.index.intersection(f_log.columns)
    fpkm_df = f_log[common].T
    fpkm_df.index.names = ['ensembl_gene_id']
    fpkm_df= fpkm_df.reset_index()
    fpkm_df.to_csv(expression_out, index=None)
    

def creating_drug_and_smiles_input(df, params):
    #se = improve_utils.load_smiles_data()
    smiles_out = params['smiles_file']
    drug_synonyms_out = params['drug_synonyms']
#    print(df)
#    se = df.set_index("improve_chem_id")
    df.to_csv(smiles_out, index_label=None)
    #creating the synonmys file
    improve_chem_list = df.index.tolist()
    syn_list = []
    for x in improve_chem_list:
        str_x = "imporve_drug_for_" + str(x)
        syn_list.append(str_x)
    df['syn'] = syn_list
    drug_syn_df = df.drop(['canSMILES'], axis=1)
    drug_syn_df.to_csv(drug_synonyms_out, header=None)
    print("smiles file downloaded and reformatted using improve utils {0} {1}".format(smiles_out,
                                                                                      drug_synonyms_out)) 

def filter_labels(df, syn, cells, drug_col):
    ## Check for rescreeens
    to_check = []
    for i in df[drug_col].unique():
        if 'rescreen' in str(i):
            to_check.append(i)
            to_check.append(i.split(' ')[0])
    to_check

    ## remove those with less labels
    to_remove = []
    for i in range(0, len(to_check), 2):
        d1 = df.loc[df[drug_col] == to_check[i]]
        d2 = df.loc[df[drug_col] == to_check[i+1]]
        common = list(set(d1.T.dropna().index.intersection(d2.T.dropna().index)) - set([drug_col]))
        if d1.iloc[0].notna().sum() > d2.iloc[0].notna().sum():
            x = d2.index[0]
            df.at[d1.index[0], drug_col] = df.loc[d1.index[0], drug_col].split(' ')[0]
        else:
            x = d1.index[0]
            df.at[d2.index[0], drug_col] = df.loc[d2.index[0], drug_col].split(' ')[0]
        to_remove.append(x)

    df = df.loc[~df.index.isin(to_remove)]

    dups = df.loc[df.duplicated(drug_col, keep=False)]
    not_dups = df.loc[~df.index.isin(dups.index)].index

    to_keep = []
    for d in dups[drug_col].unique():
        d = df.loc[df[drug_col] == d]
        d1 = d.iloc[0][cells]
        d2 = d.iloc[1][cells]
        if d1.notna().sum() > d2.notna().sum():
            to_keep.append(d.index[0])
        else:
            to_keep.append(d.index[1])

    not_dups = list(not_dups) + to_keep
    df = df.loc[not_dups]
    df.index = df[drug_col]
    df = df.loc[df.index!='improve_id']
    return df

def preprocess_data(params):
#    ic50_file = params['data_input']
#    binary_file = params['binary_input']
    split_type = ['train', 'val', 'test']
    drug_synonyms = params['drug_synonyms']
    fpkm_file = params['fpkm_file']
    syn = pd.read_csv(drug_synonyms, header=None)
    fpkm = pd.read_csv(fpkm_file, index_col=0).columns
    data_clean=[]
    data_clean_out = params['data_cleaned_out']
    for dt in split_type:
        binary_data = 'binary_' + dt + "_file"
        binary_file = params['train_ml_data_dir'] + '/' + params[binary_data]
        ic50_data = 'data_' + dt + "_file"
        ic50_file = params['train_ml_data_dir'] + '/' + params[ic50_data]
        data_cleaned_data = "data_cleaned_" + dt + "_out" 
        data_cleaned_out = params[data_cleaned_data]
        data_bin_cleaned_data = "data_bin_cleaned_" + dt + "_out"
        data_bin_cleaned_out = params['train_ml_data_dir'] + '/' + params[data_bin_cleaned_data]
        data_tuples_data = "data_tuples_" + dt + "_out"
        data_tuples_out = params['train_ml_data_dir'] + '/' + params[data_tuples_data]
        lnic50 = pd.read_csv(ic50_file, index_col=1, header=None)                                                 
        cells = lnic50.index[1:]
        df = lnic50.T
        df = filter_labels(df, syn, cells, drug_col='sample_names')
        df = df.sort_index()[cells]
#        df.to_csv(data_cleaned_out)
        data_clean.append(df)
        print("Data matrix size:", df.shape)
        drugs = list(df.index)

        # BINARIZED DATA                                                                                                                        
        print("\n\nProcessing binarized data...")
        bin_data = pd.read_csv(binary_file, index_col=0, header=None)
        bin_data.loc['compounds'] = bin_data.loc['compounds']#.str.lower()
        bin_data = bin_data.T
        print("BIN: cells match?", len(cells.intersection(bin_data.columns))==len(cells))
        bin_data = filter_labels(bin_data, syn, cells, drug_col='compounds')
        reps = {
            'R': 1,
            'R ': 1,
            'S': 0,
            'S ': 0
        }
        bin_data = bin_data.replace(reps)
        bin_data = bin_data.sort_index()[cells]
        OUTFILE = data_bin_cleaned_out
        bin_data.to_csv(OUTFILE)
        print("Generated cleaned bin file {0}".format(OUTFILE))
        print("Binarized matrix size:", bin_data.shape)
        

        # TUPLE DATA                                                                                                                            
        print("Processing tuple data...")
        tuples = pd.DataFrame(columns=['drug', 'improve_id', 'cell_line', 'response', 'resistant'])
        idx = np.transpose((1-np.isnan(np.asarray(df.values, dtype=float))).nonzero())
        
        print("num tuples:",  len(idx))
        i = 0
        for drug, cl in tqdm(idx):
            x = {'drug': drugs[drug],
                 'improve_id': lnic50.loc[cells[cl]][0],
                 'cell_line': cells[cl],
                 'response': df.loc[drugs[drug], cells[cl]],
                 'resistant': bin_data.loc[drugs[drug], cells[cl]]}
            tuples.loc[i] = x
            i += 1
        OUTFILE = data_tuples_out
#        print(tuples)
        tuples.to_csv(OUTFILE)
        print("Generated tuple labels {0}".format(OUTFILE))
    final_dataframe = pd.concat(data_clean)
    final_dataframe.to_csv(data_clean_out)
    
def preprocess_cross_study_data(params):
#    ic50_file = params['data_input']
#    binary_file = params['binary_input']
    cross_study_dir =  params['cross_study']
    data_types =  params['data_type']
    drug_synonyms = params['drug_synonyms']
    fpkm_file = params['fpkm_file']
#    data_cleaned_out = params['data_cleaned_out']
    data_bin_cleaned_out = params['data_bin_cleaned_out']
#    data_tuples_out = params['data_tuples_out']
    syn = pd.read_csv(drug_synonyms, header=None)
    cells = pd.read_csv(fpkm_file, index_col=0).columns
    data_type_list = data_types.split(',')
    for dt in data_type_list:
        ic50_file = cross_study_dir + "/" + dt + "_test.csv"
        binary_file = cross_study_dir + "/" + dt + "binary_test.csv"
        cleaned_outfile = cross_study_dir + "/" + dt + "_cleaned_test.csv"
        bin_cleaned_outfile = cross_study_dir + "/" + dt + "_bincleaned_test.csv"
        tuples_out = cross_study_dir + "/" + dt + "_tuples_test.csv"
        lnic50 = pd.read_csv(ic50_file, index_col=1, header=None)                                                 
        cells = lnic50.index[1:]
        df = lnic50.T
        df = filter_labels(df, syn, cells, drug_col='sample_names')
        df = df.sort_index()[cells]
        df.to_csv(cleaned_outfile)
        print("Data matrix size:", df.shape)
        drugs = list(df.index)
        # BINARIZED DATA                                                                                      
        print("\n\nProcessing binarized data...")
        bin_data = pd.read_csv(binary_file, index_col=0, header=None)
        bin_data.loc['compounds'] = bin_data.loc['compounds']#.str.lower()
        bin_data = bin_data.T
        print("BIN: cells match?", len(cells.intersection(bin_data.columns))==len(cells))
        bin_data = filter_labels(bin_data, syn, cells, drug_col='compounds')
        reps = {
            'R': 1,
            'R ': 1,
            'S': 0,
            'S ': 0
        }
        bin_data = bin_data.replace(reps)
        bin_data = bin_data.sort_index()[cells]
        bin_data.to_csv(bin_cleaned_outfile)
        print("Generated cleaned bin file {0}".format(bin_cleaned_outfile))
        print("Binarized matrix size:", bin_data.shape)

        # TUPLE DATA                                                                                            
        print("Processing tuple data...")
        tuples = pd.DataFrame(columns=['drug', 'improve_id', 'cell_line', 'response', 'resistant'])
        idx = np.transpose((1-np.isnan(np.asarray(df.values, dtype=float))).nonzero())

        print("num tuples:",  len(idx))
        i = 0
        for drug, cl in tqdm(idx):
            x = {'drug': drugs[drug],
                 'improve_id': lnic50.loc[cells[cl]][0],
                 'cell_line': cells[cl],
                 'response': df.loc[drugs[drug], cells[cl]],
                 'resistant': bin_data.loc[drugs[drug], cells[cl]]}
            tuples.loc[i] = x
            i += 1
        OUTFILE = tuples_out
        tuples.to_csv(OUTFILE)
        print("Generated tuple labels {0}".format(OUTFILE))

    
def generate_drug_descriptors(df, params):
    descriptor_out = params['descriptor_out']
    OUTFILE = descriptor_out
    smiles = df #pd.read_csv(smiles_file, index_col=0)
    smiles = smiles.dropna()

    allDes = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(allDes)

    new_df = pd.DataFrame(index=smiles.index, columns=allDes)
    for drug in smiles.index:
        x = smiles.loc[drug]['canSMILES']
        mol = Chem.MolFromSmiles(x)
        desc = calc.CalcDescriptors(mol)
        new_df.loc[drug] = desc

    new_df.to_csv(OUTFILE)
    print("Drug descriptor file generated at {0}".format(OUTFILE))


def generate_morganprint(smiles_df, params):
    OUTFILE = params['train_ml_data_dir'] + '/' + params['morgan_out']
    smiles = smiles_df #pd.read_csv(smiles_file, index_col=0)
    smiles = smiles.dropna()
    new_df = pd.DataFrame(index=smiles.index, columns=range(512))
    for drug in smiles.index:
        x = smiles.loc[drug]['canSMILES']
        mol = Chem.MolFromSmiles(x)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp,arr)
        new_df.loc[drug] = arr 
    
    new_df.to_csv(OUTFILE)
    print("morgan fingerprint file generated at {0}".format(OUTFILE))

    
def generate_drug_smiles(data_bin_cleaned_out, smiles_out):
    drugs = pd.read_csv(data_bin_cleaned_out, index_col=0).index

    drugs_smiles = pd.DataFrame()
    no_data = []
    for d in drugs:
        x = pcp.get_compounds(d, 'name')

        if len(x) >= 1:
            cs = x[0].canonical_smiles
            drugs_smiles[d] = [cs]
        else:
            no_data.append(d)

    for d in no_data:
        drugs_smiles[d] = [np.nan]

    drugs_smiles = drugs_smiles.T
    drugs_smiles.columns = ['smiles']
    OUTFILE = smiles_out
    drugs_smiles.to_csv(OUTFILE)
    print("drug smiles file created at {0}".format(OUTFILE))
    
def get_splits(idx, n_folds):
    """
    idx: list of indices
    n_folds: number of splits for n-fold
    """
    random.shuffle(idx)
    
    folds = []
    offset = (len(idx)+n_folds)//n_folds
    for i in range(n_folds):
        folds.append(idx[i*offset:(i+1)*offset])

    return folds

def leave_cells_out(labels, common, cells):
    while True:
        folds = get_splits(common, 5)
        success = True
        for fold in folds:
            k = (labels[fold].T.notna()).sum()
            if (k == 0).any():
                # There is a fold such that the drug is never seen
                print("There are drugs that will not be seen in one of the folds for this configuration...")
                print("Using a different configuration...")
                success = False
                break

        if success:
            break

    df = pd.DataFrame(index=cells, columns=['fold'])
    for i in range(5):
        #df.at[folds[i], 'fold'] = i
        df.loc[folds[i], 'fold'] = i
    orig = len(df)
    df = df.dropna()

    return df

def leave_pairs_out(tuples, drugs):
    folds = [[],[],[],[],[]]
    k = np.arange(5)

    # split the data evenly per drug instead of random sampling
    # so that all folds have almost equel number of samples per drug
    for drug in drugs:
        x = tuples.loc[tuples['drug'] == drug]
        sen = x.loc[x['resistant'] == 0].index
        res = x.loc[x['resistant'] == 1].index
        
        folds_sen = get_splits(list(sen), 5)
        folds_res = get_splits(list(res), 5)
        random.shuffle(k)
        for i, k_i in enumerate(k):
            folds[i] += folds_sen[k_i]
            folds[i] += folds_res[k_i]

    for i in range(5):
        print("fold %d: %d"%(i, len(folds[i])))

    fold_ass = pd.Series(np.zeros(len(tuples), dtype=int), index=tuples.index)
    for i, fold in enumerate(folds):
        fold_ass[fold] = i

    tuples['pair_fold'] = fold_ass

    for i in range(5):
        x = tuples.loc[tuples['pair_fold']==i]
        y = tuples.loc[tuples['pair_fold']!=i]
        print("(fold %d) missing cell lines in graph:"%i, set(x['cell_line'])-set(y['cell_line']))

    return tuples


def generate_splits_anl(params):
    expression_out = params['fpkm_file']
    split_type = ['train', 'val', 'test']
    morgan_out = params['morgan_data_out']
    cells = pd.read_csv(expression_out, index_col=0).columns
    drugs = pd.read_csv(morgan_out, index_col=0)
    orig_cells_gex = len(cells)
    count = 0
    dataframes_list = []
    tuples_label_fold_out = params['tuples_label_fold_out']
    for dt in split_type:
        count +=  1
        data_bin_cleaned_out = params['train_ml_data_dir'] + "/BiG_DRP_data_bined." + dt + ".csv"
        tuples_labels_out = params['train_ml_data_dir'] + "/BiG_DRP_data_tuples." + dt + ".csv"
        labels = pd.read_csv(data_bin_cleaned_out, index_col=0)
#        print(labels)
        orig_cells_lab = labels.shape[1]
        common = list(set(cells).intersection(set(labels.columns)))
        labels = labels[common] # (drugs x cells)
#        print(labels)
        drug_list = labels.index.tolist()
        drug_out =  params['drug_out'] 
        with open(drug_out, "w+") as fout:
            for i in drug_list:
                fout.write(i +'\n')
            
        print('original cells (GEX): %d'%(orig_cells_gex))
        print('original cells (labels): %d'%(orig_cells_lab))
        print('current cells: %d'%len(common) )
        orig_drugs = len(drugs)
        drugs = drugs.dropna()
        labels = labels.loc[drug_list]

        print('original drugs: %d'%(orig_drugs))
        print('current drugs: %d (dropped %d)'%(len(drugs), orig_drugs-len(drugs)))

        print("Doing leave cell lines out splits...")
 #       lco = leave_cells_out(labels, common, cells)

        ### leave pairs

        labels = labels[common].T

        print('current cells: %d'%len(labels.index) )

        tuples = pd.read_csv(tuples_labels_out, index_col=0)
        tuples['resistant'] = tuples['resistant'].astype(int)
        tuples['sensitive'] = (tuples['resistant'] + 1)%2
        # remove tuples that don't have drug or cell line data
        
#        tuples = tuples.loc[tuples['drug'].isin(drugs)].loc[tuples['cell_line'].isin(labels.index)] 
        print("number of tuples before filter:", tuples.shape[0])
        print("removed cell lines with < 3 drugs tested...")
        labels = labels.loc[labels.notna().sum(axis=1) > 2]
#        lpo_tuples = tuples.loc[tuples['cell_line'].isin(labels.index)].copy()
#        print("number of tuples after filter:", lpo_tuples.shape[0])
#        lpo = leave_pairs_out(lpo_tuples, drugs)
        df = tuples.copy()
        df['cl_fold'] = count
#        df.at[lpo.index, 'pair_fold'] = lpo['pair_fold']
 #       df['cl_fold'] = np.zeros(len(df), dtype=int)

#        for i in range(lco['fold'].max()+1):
#            cells_in_fold = lco.loc[lco['fold']==i].index
#            df.loc[df['cell_line'].isin(cells_in_fold), 'cl_fold'] = i

        df.index = range(len(df))
 #       df['pair_fold'] = df['pair_fold'].replace(np.nan, -1).astype(int)
        dataframes_list.append(df)

    final_dataframe = pd.concat(dataframes_list, ignore_index=True)
    print('this is tuples label fold out')
    print(tuples_label_fold_out)
    final_dataframe.to_csv(tuples_label_fold_out)
    

def write_out_constants(params):
    constant_file = "utils/constants.py"
    with open(constant_file, 'w+') as fout:
        _LABEL_FILE = params['tuples_label_fold_out']#.split("Data")[1] 
        _GENE_EXPRESSION_FILE = params['fpkm_file']#.split("Data")[1]
        _LABEL_MATRIX_FILE = params['data_cleaned_out']#.split("Data")[1]
        _DRUG_DESCRIPTOR_FILE = params['descriptor_out']#.split("Data")[1]
        _MORGAN_FP_FILE = params['morgan_data_out']#.split("Data")[1]
        fout.write("_LABEL_FILE = '{0}'".format(_LABEL_FILE) + '\n')
        fout.write("_GENE_EXPRESSION_FILE = '{0}'".format(_GENE_EXPRESSION_FILE) + '\n')
        fout.write("_LABEL_MATRIX_FILE = '{0}'".format(_LABEL_MATRIX_FILE) + '\n')
        fout.write("_DRUG_DESCRIPTOR_FILE = '{0}'".format(_DRUG_DESCRIPTOR_FILE) + '\n')
        fout.write("_MORGAN_FP_FILE = '{0}'".format(_MORGAN_FP_FILE) + '\n')        
        

def download_anl_data(params: Dict, inputdtd: frm.DataPathDict):
    csa_data_folder = os.path.join(os.environ['IMPROVE_DATA_DIR'], 'raw_data')
    print("data downloaded dir is {0}".format(csa_data_folder))
    if not os.path.exists(csa_data_folder):
        print('creating folder: %s'%csa_data_folder)
        os.makedirs(csa_data_folder)
#        mkdir(splits_dir)
        mkdir(x_data_dir)
        mkdir(y_data_dir)
#    fname = [inputdtd["x_data_path"] / fname for fname in params["x_data_files"]]
#    for f in fname:
#        if f.exists() == False:
#            raise Exception(f"ERROR ! File '{fname}' file not found.\n")
#    print(fname)
#    df_drug = drp.load_drug_data(fname)
#    print("\nLoading omics data...")
    oo = drp.OmicsLoader(params)
    ge = oo.dfs['cancer_gene_expression.tsv']
    process_expression(ge, params) 
    dd = drp.DrugsLoader(params)
    smi = dd.dfs['drug_SMILES.tsv']
    #params['drug_synonyms']
    smd = dd.dfs['drug_ecfp4_nbits512.tsv']
    creating_drug_and_smiles_input(smi,params)
    generate_drug_descriptors(smi, params)
    generate_morganprint(smi, params)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    scaler = None
    for stage, split_file in stages.items():
        rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
        df_response = rr.dfs["response.tsv"]
        ydf, df_canc = drp.get_common_samples(df1=df_response, df2=ge,
                                              ref_col=params["canc_col_name"])
        create_big_drp_data(ydf, stage, params)
        #print(ydf[[params["canc_col_name"], params["drug_col_name"]]].nunique())



#    url_dir = improve_data_url + "/y_data/"
#    response_file  = 'response.tsv'
#    candle.file_utils.get_file(fname=response_file, origin=url_dir + response_file,
#                                   datadir=y_data_dir,
#                                   cache_subdir=None)


def run(params):
    params['data_type'] = str(params['data_type'])
#    params = frm.build_paths(params)
    params = preprocess_param_inputs(params)
    json_out = params['output_dir']+'/params.json'
    try:
        with open (json_out, 'w') as fp:
            json.dump(params, fp, indent=4, cls=MyEncoder)
    except AttributeError:
        pass
    inputd, outputd = check_data_available(params)
    download_anl_data(params, inputd)
#    download_author_data(params, data_dir)
    preprocess_data(params)
    generate_splits_anl(params)
    write_out_constants(params)



def main(args):
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="BiG_DRP_model.txt",
        # default_model="lgbm_params_ws.txt",
        # default_model="lgbm_params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_preprocess_params,
        required=None,
    )
    print(params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


if __name__ == "__main__":
    main(sys.argv[1:])
