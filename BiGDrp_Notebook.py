#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pubchempy as pcp 
#import dask.dataframe as dd
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
from pubchempy import PubChemHTTPError
import os
import time
from scipy.stats import pearsonr, spearmanr
from collections import deque
from sklearn.metrics import r2_score
torch.cuda.empty_cache()
import gc
import candle
import os
import json
import shutil
from json import JSONEncoder
from utils.utils import mkdir
#from Big_DRP_train import main
os.environ['NUMEXPR_MAX_THREADS']='6'
from pathlib import Path
import numexpr as ne
import improve_utils
file_path = os.path.dirname(os.path.realpath('__file__'))
#del variables
gc.collect()


# ## INPUTS

# In[4]:


## INPUTS

model_name = 'BiG-DRP'
data_url="http://chia.team/IMPROVE_data/BiG_DRP_data.tar.gz"
improve_data_url="https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data"
metric='auc'
data_type='CC'
auc_threshold='0.5'
learning_rate = 0.0001
batch_size = 200
epochs = 200
cuda_name = 'cuda:0'
network_percentile = 1
drug_feature = 'desc'
split = "lco"
mode = "collate"
seed = 0
outroot = "."
response = 'auc'
normalize_response = "True"
improve_analysis = 'no'
original_data='BiG_DRP_data.tar.gz'
tcga_file = 'tcga_one_sample.csv'
drug_synonyms = "drug_synonyms.txt"
data_cleaned_out = "BiG_DRP_data_cleaned.csv"
data_tuples_out = "BiG_DRP_data_tuples.csv"
tuples_label_fold_out = "BiG_DRP_tuples_fold.csv"
expression_out = "BiG_DRP_fpkm.csv"
data_bin_cleaned_out = "BiG_DRP_data_bined.csv"
morgan_out = 'BiG_DRP_morgan.csv'
descriptor_out = 'BiG_DRP_descriptors.csv'
data_file = 'ln_ic50.csv'
smiles_file = 'BiG_DRP_smiles.csv'
binary_file = "binary_response.csv"
drugset = "drug_list.txt"
labels = 'BiG_DRP_tuple_labels_folds.csv'

dataroot="/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/"
drug_feature_dir = "drug_feature/"
drug_response_dir = "drug_response/"
expression_dir = "sanger_tcga/"
weight_folder = ""
folder = "results"
results_dir = 'results'
output_dir = 'results'
CANDLE_DATA_DIR="Data/"
lnic50_file = '/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/ln_ic50.train.csv'
binary_file = '/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/binary_response.train.csv'
drug_synonyms = '/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/drug_synonyms.txt'
fpkm_file = '/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/drp-data/grl-preprocessed/sanger_tcga/BiG_DRP_fpkm.csv'
data_cleaned_out = '/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/BiG_DRP_data_bined.train.csv'
data_bin_cleaned_out = "/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/BiG_DRP_data_cleaned.train.csv"
tuples_label_fold_out = "/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/drp-data/grl-preprocessed/drug_response/BiG_DRP_tuple_labels_folds.csv"
smiles_out = "/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/BiG_DRP_smiles.csv"
morgan_out = "/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/drp-data/grl-preprocessed/drug_features/BiG_DRP_morgan.csv"
expression_out = "/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/drp-data/grl-preprocessed/sanger_tcga/BiG_DRP_fpkm.csv"
tuples_label_out = '/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/BiG_DRP_data_tuples.train.csv'
drug_out = "/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/Data/BiG_DRP_data/drug_list.txt"


# In[5]:


required=None
additional_definitions=None

# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/'

#parent path
fdir = Path('__file__').resolve().parent
#source = "csa_data/raw_data/splits/"
auc_threshold=0.5

# initialize class
class BiG_drp_candle(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions

            
def initialize_parameters():
    preprocessor_bmk = BiG_drp_candle(file_path,
        'BiG_DRP_model.txt',
        'pytorch',
        prog='BiG_drp_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

def mv_file(download_file, req_file):
    if os.path.isfile(req_file):
        pass
    else:
        shutil.move(download_file, req_file)


# In[6]:


def preprocess(params, data_dir):
    preprocessed_dir = data_dir + "/preprocessed"
    drug_feature_dir = data_dir + "/drp-data/grl-preprocessed/drug_features/"
    drug_response_dir = data_dir + "/drp-data/grl-preprocessed/drug_response/"
    sanger_tcga_dir = data_dir + "/drp-data/grl-preprocessed/sanger_tcga/"
    cross_study = data_dir + "/cross_study"
    mkdir(drug_feature_dir)
    mkdir(drug_response_dir)
    mkdir(sanger_tcga_dir)
    mkdir(preprocessed_dir)
    mkdir(cross_study)
    args = candle.ArgumentStruct(**params)
    drug_synonym_file = data_dir + "/" + params['drug_synonyms']
    gene_expression_file = sanger_tcga_dir + "/" + params['expression_out']
    ln50_file = data_dir + "/" + params['data_file']
    model_label_file = data_dir + "/" + params['binary_file']
    tcga_file =  data_dir +'/BiG_DRP_data/' + 'supplementary/' + params['tcga_file']
    data_bin_cleaned_out = drug_feature_dir + params['data_bin_cleaned_out']
    data_cleaned_out = drug_response_dir + params['data_cleaned_out']
    data_tuples_out = drug_response_dir + params['data_tuples_out']
    tuples_label_fold_out = drug_response_dir + params['labels']
    smiles_file = data_dir + params['smiles_file']
    params['cross_study'] = cross_study
    params['data_bin_cleaned_out'] = data_bin_cleaned_out
    params['data_input'] = data_dir + "/" + params['data_file']
    params['binary_input'] = data_dir + "/" + params['binary_file']
    params['drug_out'] = data_dir + '/' + params['drugset']
    params['fpkm_file'] = gene_expression_file
    params['descriptor_out'] = drug_feature_dir + "/" + params['descriptor_out'] 
    params['morgan_data_out'] = drug_feature_dir + "/" + params['morgan_out']
    params['model_label_file'] = model_label_file
    params['smiles_file'] =  smiles_file
    params['model_label_file'] = model_label_file
    params['tuples_label_out'] = drug_response_dir + "/" + params['data_tuples_out']
    params['tuples_label_fold_out'] = drug_response_dir + "/" + params['labels']
    params['tcga_file'] = tcga_file
    params['dataroot'] = data_dir
    params['folder'] = params['outroot']
    params['outroot'] = params['outroot']
    params['network_perc'] = params['network_percentile']
    params['drug_feat'] = params['drug_feature']
    params['drug_synonyms'] = drug_synonym_file
    params['data_bin_cleaned_out'] = data_bin_cleaned_out
    params['data_cleaned_out'] = data_cleaned_out
    params['data_tuples_out'] = data_tuples_out
    params['tuples_label_fold_out'] = tuples_label_fold_out
    return(params)


# In[7]:


def download_anl_data(model_name, improve_data_url, data_type):
    csa_data_folder = os.path.join(os.environ['CANDLE_DATA_DIR'] + model_name, 'csa_data', 'raw_data')
    splits_dir = os.path.join(csa_data_folder, 'splits') 
    x_data_dir = os.path.join(csa_data_folder, 'x_data')
    y_data_dir = os.path.join(csa_data_folder, 'y_data')
    #improve_data_url = params['improve_data_url']
    #data_type = params['data_type']
    data_type_list = data_type.split(",")
    print("data downloaded dir is {0}".format(csa_data_folder))
    if not os.path.exists(csa_data_folder):
        print('creating folder: %s'%csa_data_folder)
        os.makedirs(csa_data_folder)
        mkdir(splits_dir)
        mkdir(x_data_dir)
        mkdir(y_data_dir)
    

    for files in ['_all.txt', '_split_0_test.txt',
                  '_split_0_train.txt', '_split_0_val.txt']:
        url_dir = improve_data_url + "/splits/"
        for dt in data_type_list:
            improve_file = dt + files
            data_file = url_dir + improve_file
            print("downloading file: %s"%data_file)
            candle.file_utils.get_file(improve_file, url_dir + improve_file,
                                       datadir=splits_dir,
                                       cache_subdir=None)

    for improve_file in ['cancer_gene_expression.tsv', 'drug_SMILES.tsv','drug_ecfp4_nbits512.tsv' ]:
        url_dir = improve_data_url + "/x_data/"
        data_file = url_dir + improve_file
        print("downloading file: %s"%data_file)        
        candle.file_utils.get_file(fname=improve_file, origin=url_dir + improve_file,
                                   datadir=x_data_dir,
                                   cache_subdir=None)

    url_dir = improve_data_url + "/y_data/"
    response_file  = 'response.tsv'
    candle.file_utils.get_file(fname=response_file, origin=url_dir + response_file,
                                   datadir=y_data_dir,
                                   cache_subdir=None)
    


# In[8]:


def create_big_drp_data(df, data_dir, split_type, metric):
    #metric = params['metric']
    rs_df = df.pivot(index='improve_chem_id', columns='improve_sample_id', values=metric)
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
    binary_input=data_dir + binary_file
    rs_binary_df.to_csv(binary_input, index=None)
    rs_tdf = rs_tdf.reset_index()
    rs_tdf = rs_tdf.rename({'compounds': "sample_names"},axis=1)
    ic50_file = 'data_' + split_type + "_file"    
    data_input = data_dir + ic50_file
    rs_tdf.to_csv(data_input, index_label="improve_id")


# ## PREPROCESSING

# In[9]:


def filter_labels(df, syn, cells, drug_col):
    ## Check for rescreeens
    to_check = []
    for i in df[drug_col].unique():
        if 'rescreen' in str(i):
            to_check.append(i)
            to_check.append(i.split(' ')[0])
            print(i)
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


# In[10]:


def drug_smiles(bin_out, data_dir, dt):
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
    OUTFILE = data_dir + "BiG_DRP_smiles." + dt +".csv"
    drugs_smiles.to_csv(OUTFILE)
    return OUTFILE
    


# In[11]:


def generate_morganprint(smiles_file, data_dir, dt):
    OUTFILE = data_dir + "BiG_DRP_morgan." + dt + ".csv"
    smiles = pd.read_csv(smiles_file, index_col=0)
    smiles = smiles.dropna()
    new_df = pd.DataFrame(index=smiles.index, columns=range(512))
    for drug in smiles.index:
        print(drug)
        x = smiles.loc[drug]['smiles']
        mol = Chem.MolFromSmiles(x)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp,arr)
        new_df.loc[drug] = arr 
    
    new_df.to_csv(OUTFILE)
    print("morgan fingerprint file generated at {0}".format(OUTFILE))
    return OUTFILE


# In[12]:


def generate_drug_descriptors(smiles_file, descriptor_out):
    OUTFILE = descriptor_out

    smiles = pd.read_csv(smiles_file, index_col=0)
    smiles = smiles.dropna()

    allDes = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(allDes)

    new_df = pd.DataFrame(index=smiles.index, columns=allDes)
    for drug in smiles.index:
        x = smiles.loc[drug]['smiles']
        mol = Chem.MolFromSmiles(x)
        desc = calc.CalcDescriptors(mol)
        new_df.loc[drug] = desc

    new_df.to_csv(OUTFILE)
    print("Drug descriptor file generated at {0}".format(OUTFILE))
    


# In[13]:


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


# In[14]:


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
        folds = get_splits(common, 1)
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
    data = [[]]
    k = np.arange(1)

    # split the data evenly per drug instead of random sampling
    # so that all folds have almost equel number of samples per drug
    for drug in drugs:
        x = tuples.loc[tuples['drug'] == drug]
        sen = x.loc[x['resistant'] == 0].index
        res = x.loc[x['resistant'] == 1].index
        
        #folds_sen = get_splits(list(sen), 1)
        #folds_res = get_splits(list(res), 1)
        #random.shuffle(k)
        #for i, k_i in enumerate(k):
        #    folds[i] += folds_sen[k_i]
        #    folds[i] += folds_res[k_i]

    #for i in range(1):
    #    print("fold %d: %d"%(i, len(folds[i])))

    #fold_ass = pd.Series(np.zeros(len(tuples), dtype=int), index=tuples.index)
    #for i, fold in enumerate(folds):
    #    fold_ass[fold] = i

    #tuples['pair_fold'] = fold_ass

    #for i in range(1):
    #    x = tuples.loc[tuples['pair_fold']==i]
    #    y = tuples.loc[tuples['pair_fold']!=i]
    #    print("(fold %d) missing cell lines in graph:"%i, set(x['cell_line'])-set(y['cell_line']))

    return tuples

def generate_splits(expression_out,data_bin_cleaned_out,
                    morgan_out, tuples_label_out,drug_out ):
    #expression_out = params['fpkm_file']
    #data_bin_cleaned_out = params['data_bin_cleaned_out']
    #morgan_out = params['morgan_data_out']
    #tuples_label_out = params['tuples_label_out']
    #tuples_label_fold_out = params['tuples_label_fold_out']
    cells = pd.read_csv(expression_out
                        , index_col=0).columns

    labels = pd.read_csv(data_bin_cleaned_out, index_col=0)
    drugs = pd.read_csv(morgan_out, index_col=0)
    orig_cells_gex = len(cells)
    orig_cells_lab = labels.shape[1]
    common = list(set(cells).intersection(set(labels.columns)))
    labels = labels[common] # (drugs x cells)

    drug_list = labels.index.tolist()
    #drug_out =  params['drug_out'] 
    with open(drug_out, "w+") as fout:
        for i in drug_list:
            fout.write(i +'\n')
            
    #print(drug_list)
    print('original cells (GEX): %d'%(orig_cells_gex))
    print('original cells (labels): %d'%(orig_cells_lab))
    print('current cells: %d'%len(common) )
    orig_drugs = len(drugs)
    drugs = drugs.dropna().index
    #print(drugs)
    #print(labels.loc[drug_list])
    labels = labels.loc[drug_list]

    print('original drugs: %d'%(orig_drugs))
    print('current drugs: %d (dropped %d)'%(len(drugs), orig_drugs-len(drugs)))

    print("Doing leave cell lines out splits...")
    #lco = leave_cells_out(labels, common, cells)

    ### leave pairs

    labels = labels[common].T

    print('current cells: %d'%len(labels.index) )

    tuples = pd.read_csv(tuples_label_out, index_col=0)
    tuples['resistant'] = tuples['resistant'].astype(int)
    tuples['sensitive'] = (tuples['resistant'] + 1)%2
    # remove tuples that don't have drug or cell line data
    tuples = tuples.loc[tuples['drug'].isin(drugs)].loc[tuples['cell_line'].isin(labels.index)] 

    print("number of tuples before filter:", tuples.shape[0])
    print("removed cell lines with < 3 drugs tested...")
    labels = labels.loc[labels.notna().sum(axis=1) > 2]
    lpo_tuples = tuples.loc[tuples['cell_line'].isin(labels.index)].copy()
    print("number of tuples after filter:", lpo_tuples.shape[0])
    print(lpo_tuples)
    #lpo = leave_pairs_out(lpo_tuples, drugs)
    #df = tuples.copy()
    #df.at[lpo.index, 'pair_fold'] = lpo['pair_fold']
    #df['cl_fold'] = np.zeros(len(df), dtype=int)

    #for i in range(lco['fold'].max()+1):
    #    cells_in_fold = lco.loc[lco['fold']==i].index
    #    df.loc[df['cell_line'].isin(cells_in_fold), 'cl_fold'] = i

    #df.index = range(len(df))
    #df['pair_fold'] = df['pair_fold'].replace(np.nan, -1).astype(int)
    #df.to_csv(tuples_label_fold_out)


# In[15]:


def get_splits_anl(idx, n_folds):
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

def leave_cells_out_anl(labels, common, cells):
    while True:
        folds = get_splits_anl(common, 5)
        success = True
        #print(folds)
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
    print(df.fold.value_counts())
#    return df

#def leave_pairs_out(tuples, drugs):
#    data = [[]]
#    k = np.arange(1)

    # split the data evenly per drug instead of random sampling
    # so that all folds have almost equel number of samples per drug
#    for drug in drugs:
#        x = tuples.loc[tuples['drug'] == drug]
#        sen = x.loc[x['resistant'] == 0].index
#        res = x.loc[x['resistant'] == 1].index
#        
#        folds_sen = get_splits(list(sen), 1)
#        folds_res = get_splits(list(res), 1)
#        random.shuffle(k)
#        for i, k_i in enumerate(k):
#            folds[i] += folds_sen[k_i]
#            folds[i] += folds_res[k_i]

#    for i in range(1):
#        print("fold %d: %d"%(i, len(folds[i])))

#    fold_ass = pd.Series(np.zeros(len(tuples), dtype=int), index=tuples.index)
#    for i, fold in enumerate(folds):
#        fold_ass[fold] = i

#    tuples['pair_fold'] = fold_ass

#    for i in range(1):
#        x = tuples.loc[tuples['pair_fold']==i]
#        y = tuples.loc[tuples['pair_fold']!=i]
#        print("(fold %d) missing cell lines in graph:"%i, set(x['cell_line'])-set(y['cell_line']))

#    return tuples

def generate_test_train_splits(expression_out, data_bin_cleaned_out, morgan_out, tuples_label_out, 
                                tuples_label_fold_out,drug_out):
    cells = pd.read_csv(expression_out
                        , index_col=0).columns
    labels = pd.read_csv(data_bin_cleaned_out, index_col=0)
    drugs = pd.read_csv(morgan_out, index_col=0)
    print(drugs)
    orig_cells_gex = len(cells)
    orig_cells_lab = labels.shape[1]
    common = list(set(cells).intersection(set(labels.columns)))
    print(common)
    print(labels)
    labels = labels[common] # (drugs x cells)
    #print(labels)
    
    drug_list = labels.index.tolist()
    #print(drug_list)
    #drug_out =  params['drug_out'] 
    with open(drug_out, "w+") as fout:
        for i in drug_list:
            fout.write(i +'\n')
            
    print('original cells (GEX): %d'%(orig_cells_gex))
    print('original cells (labels): %d'%(orig_cells_lab))
    
    orig_drugs = len(drugs)
    drugs = drugs.dropna().index
    dropped_indices = drugs[drugs.isna()]


    #drugs = drugs.dropna().reset_index(drop=True)
    #print(drugs)
    drugs_to_keep = labels.index.intersection(drugs)
    labels = labels.loc[drugs_to_keep]
    #labels = labels.loc[dropped_indices]
    #print(labels)
    print('original drugs: %d'%(orig_drugs))
    print('current drugs: %d (dropped %d)'%(len(drugs), orig_drugs-len(drugs)))

    print("Doing leave cell lines out splits...")
    
    #print(cells)
    #lco = leave_cells_out_anl(labels, common, cells)

    ### leave pairs

    labels = labels[common].T
    #print(labels)
    print('current cells: %d'%len(labels.index) )

    tuples = pd.read_csv(tuples_label_out, index_col=0)
    tuples['resistant'] = tuples['resistant'].astype(int)
    tuples['sensitive'] = (tuples['resistant'] + 1)%2
    # remove tuples that don't have drug or cell line data
    tuples = tuples.loc[tuples['drug'].isin(drugs)].loc[tuples['cell_line'].isin(labels.index)] 

    print("number of tuples before filter:", tuples.shape[0])
    print("removed cell lines with < 3 drugs tested...")
    labels = labels.loc[labels.notna().sum(axis=1) > 2]
    lpo_tuples = tuples.loc[tuples['cell_line'].isin(labels.index)].copy()
    print("number of tuples after filter:", lpo_tuples.shape[0])
    #print(lpo_tuples)
    lpo_tuples.to_csv(tuples_label_fold_out, index=None)
    
    #lpo = leave_pairs_out(lpo_tuples, drugs)
    #df = tuples.copy()
    #df.at[lpo.index, 'pair_fold'] = lpo['pair_fold']
    #df['cl_fold'] = np.zeros(len(df), dtype=int)

    #for i in range(lco['fold'].max()+1):
    #    cells_in_fold = lco.loc[lco['fold']==i].index
    #    df.loc[df['cell_line'].isin(cells_in_fold), 'cl_fold'] = i

    #df.index = range(len(df))
    #df['pair_fold'] = df['pair_fold'].replace(np.nan, -1).astype(int)
    #df.to_csv(tuples_label_fold_out)


# In[16]:


## Preprocess GDSC data

def preprocess_data(ic50_file, fpkm_file, drug_synonyms,data_dir, expression_out, drug_out):
    split_type = ['train', 'val', 'test']
    syn = pd.read_csv(drug_synonyms, header=None)
    fpkm = pd.read_csv(fpkm_file, index_col=0).columns

    for dt in split_type:
        binary_data = 'binary_response.'+ dt + '.csv'
        binary_file = data_dir + binary_data
        ic50_data = 'ln_ic50.' +  dt + '.csv'
        ic50_file = data_dir + ic50_data
        data_cleaned_data = "BiG_DRP_data_cleaned." + dt + ".csv" 
        data_cleaned_out = data_dir + data_cleaned_data
        data_bin_cleaned_data = "BiG_DRP_data_bined." + dt + ".csv"
        data_bin_cleaned_out = data_dir + data_bin_cleaned_data
        data_tuples_data = "BiG_DRP_data_tuples." + dt + ".csv"
        data_tuples_out = data_dir + data_tuples_data
        lnic50 = pd.read_csv(ic50_file, index_col=1, header=None)                                                 
        cells = lnic50.index[1:]
        df = lnic50.T
        df = filter_labels(df, syn, cells, drug_col='sample_names')
        df = df.sort_index()[cells]
        df.to_csv(data_cleaned_out)
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
        
        #generate smiles
        print(data_bin_cleaned_out)
        smiles_file = drug_smiles(data_bin_cleaned_out, data_dir, dt)
        morgan_file = generate_morganprint(smiles_file, data_dir, dt)
        print(morgan_file)
        # TUPLE DATA                                                                                                                            
        print("Processing tuple data...")
        tuples = pd.DataFrame(columns=['drug', 'improve_id', 'cell_line', 'response', 'resistant'])
        idx = np.transpose((1-np.isnan(np.asarray(df.values, dtype=float))).nonzero())
        
        print("num tuples:",  len(idx))
        i = 0
        for drug, cl in tqdm(idx):
           #print(drug)
            x = {'drug': drugs[drug],
                 'improve_id': lnic50.loc[cells[cl]][0],
                 'cell_line': cells[cl],
                 'response': df.loc[drugs[drug], cells[cl]],
                 'resistant': bin_data.loc[drugs[drug], cells[cl]]}
            tuples.loc[i] = x
        i += 1
        OUTFILE = data_tuples_out
        tuples.to_csv(OUTFILE)
        generate_splits(expression_out,data_bin_cleaned_out,
                        morgan_file, data_tuples_out,drug_out )
        print("Generated tuple labels {0}".format(OUTFILE))
    print("Generated tuple labels {0}".format(OUTFILE))


# In[17]:


def run_preprocessing(lnic50_file,  fpkm_file, drug_synonyms, expression_out, drug_out):
#    params = initialize_parameters()
    data_dir = os.environ['CANDLE_DATA_DIR'] + model_name + "/Data/BiG_DRP_data/"
#    params =  preprocess(params, data_dir)
#    print(params)#ic50_file, fpkm_file, drug_synonyms,data_dir
    preprocess_data(lnic50_file, fpkm_file,drug_synonyms,  data_dir, expression_out, drug_out)
    #generate_morganprint(smiles_out, morgan_out, data_dir)
    #generate_drug_smiles(data_bin_cleaned_out, smiles_out, data_dir)
    #print(tuples_label_out)
    #generate_splits(expression_out, data_bin_cleaned_out, morgan_out, tuples_label_out, 
    #                tuples_label_fold_out, drug_out)
    #generate_test_train_splits(expression_out, data_bin_cleaned_out, morgan_out, tuples_label_out, 
    #                tuples_label_fold_out, drug_out)
    #print(tuples_label_fold_out)
#run_preprocessing()   
#run_preprocessing(lnic50_file, fpkm_file, drug_synonyms, expression_out, drug_out)


# ## TRAINING
# 

# In[18]:


def reset_seed(seed=None):
    if seed is not None:
        global _SEED
        _SEED = seed

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    random.seed(_SEED)

def create_fold_mask(tuples, label_matrix):
    x = pd.DataFrame(index=label_matrix.index, columns=label_matrix.columns)
    for index, row in tuples.iterrows():
        x.loc[row['cell_line'], row['drug']] = row['cl_fold']
    return x
    
def save_flags(FLAGS):
    with open(FLAGS.outroot + "/results/" + FLAGS.folder + '/flags.cfg','w') as f:
        for arg in vars(FLAGS):
            f.write('--%s=%s\n'%(arg, getattr(FLAGS, arg)))


def mkdir(directory):
    directories = directory.split("/")   

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s'%folder)
            os.mkdir(folder)


def reindex_tuples(tuples, drugs, cells, start_index=0):
    """
    Transforms strings in the drug and cell line columns to numerical indices
    
    tuples: dataframe with columns: cell_line_name, drug_col, drug_row
    drugs: list of drugs
    cells: list of cell line names
    """
    tuples = tuples.copy()
    for i, drug in enumerate(drugs):
        tuples = tuples.replace(drug, i)
    for i, cell in enumerate(cells):
        tuples = tuples.replace(cell, i+start_index)

    return tuples

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# In[19]:


class TupleMatrixDataset(Dataset):
    def __init__(self, tuples, cell_features, label_matrix, bin_label_matrix=None, weighted=False):
        order = ['drug', 'cell_line']
        self.tuples = tuples[order].values
        self.cell_features = cell_features
        self.label_matrix = label_matrix
        self.num_drugs = label_matrix.shape[1]
        self.bin_label_matrix = bin_label_matrix

        if weighted:
            self.weight = 1./label_matrix.std(axis=0)
        else:
            self.weight = torch.ones(self.num_drugs)

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

        if self.bin_label_matrix is not None and self.bin_label_matrix.dim() == 2:
            self.bin_label_matrix = self.bin_label_matrix.unsqueeze(2)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.to_list()

        tuples = self.tuples[idx]
        drug = tuples[0]
        cell = tuples[1]

        if self.bin_label_matrix is None:
            sample = (self.cell_features[cell],
                drug,
                self.label_matrix[cell, drug],
                self.weight[drug])
        else:
            sample = (self.cell_features[cell],
                drug,
                self.label_matrix[cell, drug],
                self.bin_label_matrix[cell, drug],
                self.weight[drug])

        return sample

    def get_all_labels(self):
        return self.labels.numpy()

class TupleMapDataset(Dataset):
    def __init__(self, tuples, drug_features, cell_features, label_matrix, bin_label_matrix=None):

        order = ['drug', 'cell_line']
        self.tuples = tuples[order].values #index = range(len(triplets))    
        self.cell_features = torch.Tensor(cell_features)
        self.drug_features = torch.Tensor(drug_features.values)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = label_matrix.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

        self.bin_label_matrix = bin_label_matrix
        if bin_label_matrix is not None:
            self.bin_label_matrix = torch.Tensor(bin_label_matrix)
            if self.bin_label_matrix.dim() == 2:
                self.bin_label_matrix = self.bin_label_matrix.unsqueeze(2)


    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.to_list()

        tuples = self.tuples[idx]
        drug = tuples[0]
        cell = tuples[1]

        if self.bin_label_matrix is None:
            sample = (self.cell_features[cell],
                    self.drug_features[drug],
                    self.label_matrix[cell, drug])
        else:
            sample = (self.cell_features[cell],
                    self.drug_features[drug],
                    self.label_matrix[cell, drug],
                    self.bin_label_matrix[cell, drug])

        return sample

def create_network(tuples, percentile=1):    

    sen_edges = pd.DataFrame()
    res_edges = pd.DataFrame()
    for drug in tuples['drug'].unique():
        drug_edges = tuples.loc[tuples['drug']==drug]        
        thresh = np.percentile(drug_edges['response'], percentile)
        sen_edges = pd.concat([sen_edges, drug_edges.loc[drug_edges['response']<thresh]])
        thresh = np.percentile(drug_edges['response'], (100-percentile))
        res_edges = pd.concat([res_edges, drug_edges.loc[drug_edges['response']>thresh]])

    print("generated a network with %d sensitive edges and %d resistant edges "%(len(sen_edges), len(res_edges)))

    graph_data = {
             ('cell_line', 'is_sensitive', 'drug'): (sen_edges['cell_line'].values, sen_edges['drug'].values),
             ('drug', 'is_effective', 'cell_line'): (sen_edges['drug'].values, sen_edges['cell_line'].values),
             ('cell_line', 'is_resistant', 'drug'): (res_edges['cell_line'].values, res_edges['drug'].values),
             ('drug', 'is_ineffective', 'cell_line'): (res_edges['drug'].values, res_edges['cell_line'].values)}
    network = dgl.heterograph(graph_data)
    print(network)
    cl = list(set(sen_edges['cell_line']).union(set(res_edges['cell_line'])))
    print('unique_CLs', len(cl))

    if len(cl) != tuples['cell_line'].nunique():
        network.add_nodes(tuples['cell_line'].max()-max(cl), ntype='cell_line')
    return network


# In[20]:


def standardize(train_x, test_x):
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.transform(test_x)
    return train_x, test_x


def initialize(drug_feat, normalize_response, binary=False, multitask=False):

    reset_seed(seed)
    mkdir(outroot + "/results/" + folder)

    LABEL_FILE = "Data/BiG_DRP_data/drp-data/grl-preprocessed/drug_response/BiG_DRP_tuple_labels_folds.csv" 
    GENE_EXPRESSION_FILE = "Data//BiG_DRP_data//drp-data/grl-preprocessed/sanger_tcga//BiG_DRP_fpkm.csv"
    LABEL_MATRIX_FILE = "Data//BiG_DRP_data//drp-data/grl-preprocessed/drug_response/BiG_DRP_data_cleaned.csv"
    DRUG_DESCRIPTOR_FILE = 'Data//BiG_DRP_data//drp-data/grl-preprocessed/drug_features//BiG_DRP_descriptors.csv'
    MORGAN_FP_FILE = 'Data//BiG_DRP_data//drp-data/grl-preprocessed/drug_features//BiG_DRP_morgan.csv'

    if drug_feat == 'desc' or drug_feat == 'mixed':
        DRUG_FEATURE_FILE = dataroot + DRUG_DESCRIPTOR_FILE
        drug_feats = pd.read_csv(DRUG_FEATURE_FILE, index_col=0)

        df = StandardScaler().fit_transform(drug_feats.values) # normalize
        drug_feats = pd.DataFrame(df, index=drug_feats.index, columns=drug_feats.columns)

        if drug_feat == 'mixed':
            DRUG_MFP_FEATURE_FILE = dataroot + MORGAN_FP_FILE
            drug_mfp = pd.read_csv(DRUG_MFP_FEATURE_FILE, index_col=0)
            drug_feats[drug_mfp.columns] = drug_mfp

        valid_cols = drug_feats.columns[~drug_feats.isna().any()] # remove columns with missing data
        drug_feats = drug_feats[valid_cols]
        
    else:
        DRUG_FEATURE_FILE = dataroot + MORGAN_FP_FILE
        drug_feats = pd.read_csv(DRUG_FEATURE_FILE, index_col=0)

    #print(drug_feats)
    cell_lines = pd.read_csv(GENE_EXPRESSION_FILE, index_col=0).T # need to normalize
    labels = pd.read_csv(LABEL_FILE)
    labels['cell_line'] = labels['cell_line'].astype(str)
    labels['response'] = labels['response'] 
    columns = ['drug','improve_id','cell_line', 'response', 'resistant', 'sensitive','cl_fold']
    labels = labels[columns]
    labels = labels.loc[labels['drug'].isin(drug_feats.index)] # use only cell lines with data
    labels = labels.loc[labels['cell_line'].isin(cell_lines.index)] # use only drugs with data
    cell_lines = cell_lines.loc[cell_lines.index.isin(labels['cell_line'].unique())] # use only cell lines with labels
    drug_feats = drug_feats.loc[drug_feats.index.isin(labels['drug'].unique())]      # use only drugs in labels
    drug_list = drug_feats.index.tolist()
    label_matrix = pd.read_csv(LABEL_MATRIX_FILE, index_col=0).T 
    label_matrix = label_matrix.loc[cell_lines.index] # align the matrix
    label_matrix = label_matrix[drug_list]
    label_matrix = label_matrix.drop_duplicates()
    label_matrix = label_matrix.loc[:, ~label_matrix.columns.duplicated()]
    if normalize_response:
        ss = StandardScaler() # normalize IC50
        temp = ss.fit_transform(label_matrix.values)
        label_matrix = pd.DataFrame(temp, index=label_matrix.index, columns=label_matrix.columns)
    else:
        label_matrix = label_matrix.astype(float)
    return drug_feats, cell_lines, labels, label_matrix, standardize 


# In[21]:


def fold_validation(hyperparams, seed, network, train_data, val_data, cell_lines, 
                    drug_feats, tuning, epoch, maxout=False):
    reset_seed(seed)
    print('The batch size is {0}'.format(hyperparams['batch_size']))
    train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=hyperparams['batch_size'], shuffle=False)
    #print('the val_data shape is {0}'.format(val_data.shape))
    n_genes = cell_lines.shape[1]
    n_drug_feats = drug_feats.shape[1]

    trainer = Trainer(n_genes, cell_lines, drug_feats, network, hyperparams)
    val_error, metric_names = trainer.fit(
            num_epoch=epoch, 
            train_loader=train_loader, 
            val_loader=val_loader,
            tuning=tuning,
            maxout=maxout)

    return val_error, trainer, metric_names

def create_dataset(tuples, train_x, val_x, 
    train_y, val_y, train_mask, val_mask, drug_feats, percentile):
    network = create_network(tuples, percentile)

    train_data = TupleMatrixDataset( 
        tuples,
        torch.FloatTensor(train_x),
        torch.FloatTensor(train_y))

    val_data = TensorDataset(
        torch.FloatTensor(val_x),
        torch.FloatTensor(val_y),
        torch.FloatTensor(val_mask))

    cell_lines = torch.FloatTensor(train_x)
    drug_feats = torch.FloatTensor(drug_feats.values)

    return network, train_data, val_data, cell_lines, drug_feats



# In[22]:


def nested_cross_validation(drug_feats, cell_lines, labels,
                            label_matrix, normalizer, learning_rate, epoch, batch_size):
    reset_seed(seed)
    hyperparams = {
        'learning_rate': learning_rate,
        'num_epoch': epoch,
        'batch_size': batch_size,
        'common_dim': 512,
        'expr_enc': 1024,
        'conv1': 512,
        'conv2': 512,
        'mid': 512,
        'drop': 1}

    label_mask = create_fold_mask(labels, label_matrix)
    label_matrix = label_matrix.replace(np.nan, 0)

    final_metrics = None
    drug_list = list(drug_feats.index)

    for i in range(5):
        print('==%d=='%i)
        test_fold = i 
        val_fold = (i+1)%5
        train_folds = [x for x in range(5) if (x != test_fold) and (x != val_fold)]

        hp = hyperparams.copy()

        # === find number of epochs ===

        train_tuples = labels.loc[labels['fold'].isin(train_folds)]
        train_samples = list(train_tuples['cell_line'].unique())
        train_x = cell_lines.loc[train_samples].values
        train_y = label_matrix.loc[train_samples].values
        train_mask = (label_mask.loc[train_samples].isin(train_folds))*1

        val_tuples = labels.loc[labels['fold'] == val_fold]
        val_samples = list(val_tuples['cell_line'].unique())
        val_x = cell_lines.loc[val_samples].values
        val_y = label_matrix.loc[val_samples].values
        val_mask = ((label_mask.loc[val_samples]==val_fold)*1).values

        train_tuples = train_tuples[['drug', 'cell_line', 'response']]
        train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds

        train_x, val_x = normalizer(train_x, val_x)

        network, train_data, val_data, \
        cl_tensor, df_tensor = create_dataset(
            train_tuples, 
            train_x, val_x, 
            train_y, val_y, 
            train_mask, val_mask, drug_feats, network_percentile)

        val_error,_,_ = fold_validation(hp, seed, network, train_data, 
            val_data, cl_tensor, df_tensor, tuning=False, 
            epoch=hp['num_epoch'], maxout=False)

        average_over = 3
        mov_av = moving_average(val_error[:,0], average_over)
        smooth_val_loss = np.pad(mov_av, average_over//2, mode='edge')
        epoch = np.argmin(smooth_val_loss)
        hp['num_epoch'] = int(max(epoch, 2)) 

        # === actual test fold ===

        train_folds = train_folds + [val_fold]
        train_tuples = labels.loc[labels['fold'].isin(train_folds)]
        train_samples = list(train_tuples['cell_line'].unique())
        train_x = cell_lines.loc[train_samples].values
        train_y = label_matrix.loc[train_samples].values
        train_mask = (label_mask.loc[train_samples].isin(train_folds))*1

        test_tuples = labels.loc[labels['fold'] == test_fold]
        test_samples = list(test_tuples['cell_line'].unique())
        test_x = cell_lines.loc[test_samples].values
        test_y = label_matrix.loc[test_samples].values
        test_mask = (label_mask.loc[test_samples]==test_fold)*1

        train_tuples = train_tuples[['drug', 'cell_line', 'response']]
        train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds

        train_x, test_x = normalizer(train_x, test_x)
        network, train_data, test_data, \
        cl_tensor, df_tensor = create_dataset(
            train_tuples, 
            train_x, test_x, 
            train_y, test_y, 
            train_mask, test_mask.values, drug_feats, network_percentile)

        test_error, trainer, metric_names = fold_validation(hp, seed, network, train_data, 
                                                            test_data, cl_tensor, df_tensor, tuning=False, 
                                                            epoch=hp['num_epoch'], maxout=True) # set maxout so that the trainer uses all epochs

        if i == 0:
            final_metrics = np.zeros((5, test_error.shape[1]))

        final_metrics[i] = test_error[-1]
        test_metrics = pd.DataFrame(test_error, columns=metric_names)
        test_metrics.to_csv(outroot + "/results/" + folder + '/fold_%d.csv'%i, index=False)

        drug_enc = trainer.get_drug_encoding().cpu().detach().numpy()
        pd.DataFrame(drug_enc, index=drug_list).to_csv(FLAGS.outroot + "/results/" + FLAGS.folder + '/encoding_fold_%d.csv')

        trainer.save_model(outroot + "/results/" + folder, i, hp)

        # save predictions
        test_data = TensorDataset(torch.FloatTensor(test_x))
        test_data = DataLoader(test_data, batch_size=hyperparams['batch_size'], shuffle=False)

        prediction_matrix = trainer.predict_matrix(test_data, drug_encoding=torch.Tensor(drug_enc))
        prediction_matrix = pd.DataFrame(prediction_matrix, index=test_samples, columns=drug_list)

        # remove predictions for non-test data
        test_mask = test_mask.replace(0, np.nan)
        prediction_matrix = prediction_matrix*test_mask

        prediction_matrix.to_csv(outroot + "/results/val_prediction_fold_%d.csv")
    
    return final_metrics


def create_dataset_anl(tuples, train_x, val_x, 
    train_y, val_y, drug_feats, percentile):
    network = create_network(tuples, percentile)

    train_data = TupleMatrixDataset( 
        tuples,
        torch.FloatTensor(train_x),
        torch.FloatTensor(train_y))

    val_data = TensorDataset(
        torch.FloatTensor(val_x),
        torch.FloatTensor(val_y))
        #torch.FloatTensor(val_mask))

    cell_lines = torch.FloatTensor(train_x)
    drug_feats = torch.FloatTensor(drug_feats.values)

    return network, train_data, val_data, cell_lines, drug_feats


def anl_test_data(drug_feats, cell_lines, labels,
                            label_matrix, normalizer, learning_rate, epoch, batch_size):
    reset_seed(seed)
    hyperparams = {
        'learning_rate': learning_rate,
        'num_epoch': epoch,
        'batch_size': batch_size,
        'common_dim': 512,
        'expr_enc': 1024,
        'conv1': 512,
        'conv2': 512,
        'mid': 512,
        'drop': 1}

    label_matrix = label_matrix.replace(np.nan, 0)
    hp = hyperparams.copy()
    final_metrics = None
    drug_list = list(drug_feats.index)
    train_tuples = labels.loc[labels['cl_fold'] == 1]
    train_tuples = labels
    train_samples = list(train_tuples['cell_line'].unique())
    train_x = cell_lines.loc[train_samples].values
    train_y = label_matrix.loc[train_samples].values
    val_tuples = labels.loc[labels['cl_fold'] == 2]
    val_samples = list(val_tuples['cell_line'].unique())
    val_x = cell_lines.loc[val_samples].values
    val_y = label_matrix.loc[val_samples].values
    train_tuples = train_tuples[['drug', 'cell_line', 'response']]
    train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds
    train_x, val_x = normalizer(train_x, val_x)
    network, train_data, val_data, cl_tensor, df_tensor = create_dataset_anl(train_tuples, 
                                                                             train_x, val_x, 
                                                                             train_y, val_y,                                                                    
                                                                             drug_feats, 
                                                                             network_percentile)

    val_error,_,_ = fold_validation(hp, seed, network, 
                                    train_data, val_data, 
                                    cl_tensor, df_tensor,
                                    tuning=False, 
                                    epoch=hp['num_epoch'], maxout=False)

    average_over = 3
    mov_av = moving_average(val_error[:,0], average_over)
    smooth_val_loss = np.pad(mov_av, average_over//2, mode='edge')
    epoch = np.argmin(smooth_val_loss)
    hp['num_epoch'] = int(max(epoch, 2)) 
    train_tuples = labels.loc[labels['cl_fold'] == 1]
    train_samples = list(train_tuples['cell_line'].unique())
    train_x = cell_lines.loc[train_samples].values
    train_y = label_matrix.loc[train_samples].values
    #train_mask = (label_mask.loc[train_samples].isin(train_folds))*1
    test_tuples = labels.loc[labels['cl_fold'] == 3]
    test_samples = list(test_tuples['cell_line'].unique())
    test_x = cell_lines.loc[test_samples].values
    test_y = label_matrix.loc[test_samples].values
    #test_mask = (label_mask.loc[test_samples]==test_fold)*1
    train_tuples = train_tuples[['drug', 'cell_line', 'response']]
    train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds

    train_x, test_x = normalizer(train_x, test_x)
    network, train_data, test_data, cl_tensor, df_tensor = create_dataset_anl(train_tuples, 
                                                                          train_x, test_x, 
                                                                          train_y, test_y, 
                                                                          drug_feats, network_percentile)

    test_error, trainer, metric_names = fold_validation(hp, seed, network, 
                                                        train_data, 
                                                        test_data, cl_tensor, df_tensor, tuning=False, 
                                                        epoch=hp['num_epoch'], maxout=True) # set maxout so that the trainer uses all epochs

    test_metrics = pd.DataFrame(test_error, columns=metric_names)
    test_metrics.to_csv(outroot + "/results/" + folder + '/fold_%d.csv', index=False)

    drug_enc = trainer.get_drug_encoding().cpu().detach().numpy()
    pd.DataFrame(drug_enc, index=drug_list).to_csv(outroot + "/results/" + folder + '/encoding_drug.csv')

    trainer.save_model(outroot + "/results/" + folder, hp)

        # save predictions
    test_data = TensorDataset(torch.FloatTensor(test_x))
    test_data = DataLoader(test_data, batch_size=hyperparams['batch_size'], shuffle=False)

    prediction_matrix = trainer.predict_matrix(test_data, drug_encoding=torch.Tensor(drug_enc))
    prediction_matrix = pd.DataFrame(prediction_matrix, index=test_samples, columns=drug_list)

    prediction_matrix.to_csv(outroot + "/results/val_prediction_fold_%d.csv")
    
    return final_metrics


# In[23]:


class Trainer:
    def __init__(self, n_genes, cell_feats, drug_feats, network, hyp, test=False, load_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cell_feats = cell_feats
        self.drug_feats = drug_feats
        self.network = network

        if load_model_path is not None:
            self.model = ModelHead(n_genes, hyp)
            self.model.load_state_dict(torch.load(load_model_path), strict=False)
            self.model = self.model.to(self.device)

        if not test:
            self.model = BiGDRP(n_genes, self.cell_feats.shape[1], drug_feats.shape[1], network.etypes, hyp).to(self.device)
            
            self.mse_loss = torch.nn.MSELoss()
            params = self.model.parameters()
            self.optimizer = torch.optim.Adam(params, lr=hyp['learning_rate'])

            graph_sampler = MultiLayerFullNeighborSampler(2)
            self.network.ndata['features'] = {'drug': self.drug_feats, 'cell_line': self.cell_feats}
            _,_, blocks = graph_sampler.sample_blocks(self.network, {'drug': range(len(drug_feats))})
            self.blocks = [b.to(self.device) for b in blocks]
           
            #make sure they are aligned correctly
            self.cell_feats = self.blocks[0].ndata['features']['cell_line'].to(self.device)
            self.drug_feats = self.blocks[0].ndata['features']['drug'].to(self.device)


    def train_step(self, train_loader, device):
        # trains on tuples
        self.model.train()
        for (x, d1, y, w) in train_loader:
            x, d1, y = x.to(device), d1.to(device), y.to(device)
            w = w.to(device)

            self.optimizer.zero_grad()
            pred = self.model(self.blocks, self.drug_feats, self.cell_feats, x, d1)
            loss = self.mse_loss(pred, y)

            loss.backward()
            self.optimizer.step()

        r2 = r2_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        return loss.item(), r2

    def validation_step_cellwise(self, val_loader, device):
        """
        Creates a matrix of predictions with shape (n_cell_lines, n_drugs) and calculates the metrics
        """

        self.model.eval()

        val_loss = 0
        preds = []
        ys = []
        with torch.no_grad():
            for (x, y, mask) in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = self.model.predict_all(self.blocks, self.drug_feats, self.cell_feats, x)
                val_loss += (((pred - y)*mask)**2).sum()

                mask = mask.cpu().detach().numpy().nonzero()
                ys.append(y.cpu().detach().numpy()[mask])
                preds.append(pred.cpu().detach().numpy()[mask])

        preds = np.concatenate(preds, axis=0)
        ys = np.concatenate(ys, axis=0)
        r2 = r2_score(ys, preds)
        pearson = pearsonr(ys, preds)[0]
        spearman = spearmanr(ys, preds)[0]

        return val_loss.item()/len(ys), r2, pearson, spearman
    
    #def validation_step_cellwise_anl(self, val_loader, device):
    #    """
    #    Creates a matrix of predictions with shape (n_cell_lines, n_drugs) and calculates the metrics
    #    """

    #    self.model.eval()

    #    val_loss = 0
    #    preds = []
    #    ys = []
    #    with torch.no_grad():
    #        for (x, y) in val_loader:
    #            x, y = x.to(device), y.to(device)
    #            #print(x.shape)
    #            #print(y.shape)
    #            pred = self.model.predict_all(self.blocks, self.drug_feats, self.cell_feats, x)
    #            new_pred = pred.T
    #            new_y = y.T
                #print(new_pred.shape)
                #print(new_y.shape)
    #            val_score = (pred -y)**2
    #            print(val_score)
    #            val_loss += ((new_pred - new_y)**2).sum()

                #mask = mask.cpu().detach().numpy().nonzero()
    #            ys.append(y.cpu().detach().numpy())
    #            preds.append(pred.cpu().detach().numpy())

     #   preds = np.concatenate(preds, axis=0)
     #   ys = np.concatenate(ys, axis=0)
     #   r2 = r2_score(ys, preds)
     #   pearson = pearsonr(ys, preds)[0]
     #   spearman = spearmanr(ys, preds)[0]

      #  return val_loss.item()/len(ys), r2, pearson, spearman

    def validation_step_cellwise_anl(self, val_loader, device):
        """
        Creates a matrix of predictions with shape (n_cell_lines, n_drugs) and calculates the metrics
        """

        self.model.eval()

        val_loss = 0
        preds = []
        ys = []
        with torch.no_grad():
            for (x, y) in val_loader:
                x, y = x.to(device), y.to(device)
                x_shape = x.shape
                y_shape = y.shape
                #print("the shape of y is {0}".format(y.shape))
                pred = self.model.predict_all(self.blocks, self.drug_feats, self.cell_feats, x)
                #print("the shape of pred is {0}".format(pred.shape))
                #pred_mean = torch.mean(output_tensor, dim=2)
                val_score = (pred - y)**2
                val_loss += val_score.mean()  # Use the mean squared error as the loss

                ys.append(y.cpu().detach().numpy())
                preds.append(pred.cpu().detach().numpy())

        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=0)
        min_rows = min(ys.shape[0], preds.shape[0])
        ys = ys[:min_rows]
        preds = preds[:min_rows]
        print("ys shape is {0}".format(ys.shape))
        print(preds.shape)
        r2 = r2_score(ys, preds)
        pearson = pearsonr(ys.flatten(), preds.flatten())[0]
        spearman = spearmanr(ys.flatten(), preds.flatten())[0]

        return val_loss.item() / len(ys), r2,pearson, spearman
        #preds = np.concatenate(preds, axis=0)
        #ys = np.concatenate(ys, axis=0)
        #r2 = r2_score(ys, preds)
        #pearson = pearsonr(ys, preds)[0]
        #spearman = spearmanr(ys, preds)[0]

        #return val_loss.item()/len(ys), r2, pearson, spearman


    def get_drug_encoding(self):
        """
        returns the tensor of drug encodings (by GraphConv])
        """
        self.model.eval()
        with torch.no_grad():
            drug_encoding = self.model.get_drug_encoding(self.blocks, self.drug_feats, self.cell_feats)
        return drug_encoding

    def predict_matrix(self, data_loader, drug_encoding=None):
        """
        returns a prediction matrix of (N, n_drugs)
        """

        self.model.eval()

        preds = []
        if drug_encoding is None:
            drug_encoding = self.get_drug_encoding() # get the encoding first so that we don't have top run the conv every time
        else:
            drug_encoding = drug_encoding.to(self.device)

        with torch.no_grad():
            for (x,) in data_loader:
                x = x.to(self.device)
                pred = self.model.predict_response_matrix(x, drug_encoding)
                preds.append(pred)

        preds = torch.cat(preds, axis=0).cpu().detach().numpy()
        return preds


    def fit(self, num_epoch, train_loader, val_loader, tuning=False, maxout=False):
        start_time = time.time()

        ret_matrix = np.zeros((num_epoch, 6))
        loss_deque = deque([], maxlen=5)

        best_loss = np.inf
        best_loss_avg5 = np.inf
        best_loss_epoch = 0
        best_avg5_loss_epoch = 0

        count = 0

        for epoch in range(num_epoch):
            train_metrics = self.train_step(train_loader, self.device)
            val_metrics = self.validation_step_cellwise_anl(val_loader, self.device)

            #ret_matrix[epoch, :4] = val_metrics
            ret_matrix[epoch, :4] = val_metrics[:4]
            ret_matrix[epoch, 4:] = train_metrics

            if best_loss > val_metrics[0]:
                best_loss = val_metrics[0]
                best_loss_epoch = epoch+1

            loss_deque.append(val_metrics[0])
            loss_avg5 = sum(loss_deque)/len(loss_deque)
            
            if best_loss_avg5 > loss_avg5:
                best_loss_avg5 = loss_avg5
                best_avg5_loss_epoch = epoch+1
                count = 0
            else:
                count += 1

            if count == 10 and not maxout:
                ret_matrix = ret_matrix[:epoch+1]
                break

            elapsed_time = time.time() - start_time
            start_time = time.time()
            print("%d\tval-mse:%.4f\tbatch-mse:%.4f\tval-r2:%.4f\tbatch-r2:%.4f\tval-spearman:%.4f\t%ds"%(
                epoch+1, val_metrics[0], train_metrics[0],val_metrics[1], train_metrics[1], val_metrics[3], int(elapsed_time)))

#        if not tuning:
        metric_names = ['test MSE', 'test R^2', 'test pearsonr', 'test spearmanr', 'train MSE', 'train R^2']
        #metric_names = ['test MSE', 'test R^2', 'train MSE', 'train R^2']
        return ret_matrix, metric_names

    def save_model(self, directory, hyp):
        os.makedirs(directory, exist_ok=True)
        model_weights_path = os.path.join(directory, f'model_weights_fold.pt')
        torch.save(self.model.state_dict(), model_weights_path)

        # Save hyperparameters as JSON
        hyp_path = os.path.join(directory, f'model_config_fold.txt')
        with open(hyp_path, "w") as f:
            json.dump(hyp, f)
            
        #torch.save(self.model.state_dict(), directory+'/model_weights_fold_%d')

        #x = json.dumps(hyp)
        #f = open(directory+"/model_config_fold_%d.txt","w")
        #f.write(x)
        #f.close()


# In[24]:


## MODEL
class DRPPlus(nn.Module):
    def __init__(self, n_genes, hyp):
        super(DRPPlus, self).__init__()

        self.expr_l1 = nn.Linear(n_genes, hyp['expr_enc'])
        self.mid = nn.Linear(hyp['expr_enc'] + hyp['conv2'], hyp['mid'])
        self.out = nn.Linear(hyp['mid'], 1)
        
        if hyp['drop'] == 0:
            drop=[0,0]
        else:
            drop=[0.2,0.5]

        self.in_drop = nn.Dropout(drop[0])
        self.mid_drop = nn.Dropout(drop[1])
        self.alpha = 0.5

    def forward(self, cell_features, drug_enc):
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
 
        return out

    def predict_response_matrix(self, cell_features, drug_enc):
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        expr_enc = expr_enc.unsqueeze(1) # (batch, 1, expr_enc_size)
        drug_enc = drug_enc.unsqueeze(0) # (1, n_drugs, drug_enc_size)
        
        expr_enc = expr_enc.repeat(1,drug_enc.shape[1],1) # (batch, n_drugs, expr_enc_size)
        drug_enc = drug_enc.repeat(expr_enc.shape[0],1,1) # (batch, n_drugs, drug_enc_size)
        
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
        out = out.view(-1, drug_enc.shape[1]) # (batch, n_drugs)
        return out
        


class BiGDRP(nn.Module):
    def __init__(self, n_genes, n_cl_feats, n_drug_feats, rel_names, hyp):
        super(BiGDRP, self).__init__()

        self.conv1 = HeteroGraphConv(
            {rel: dgl.nn.GraphConv(in_feats=hyp['common_dim'], out_feats=hyp['common_dim']) for rel in rel_names})
        self.conv2 = HeteroGraphConv(
            {rel: dgl.nn.GraphConv(in_feats=hyp['common_dim'], out_feats=hyp['common_dim']) for rel in rel_names})

        self.drug_l1 = nn.Linear(n_drug_feats, hyp['common_dim'])
        self.cell_l1 = nn.Linear(n_cl_feats, hyp['common_dim'])
        self.expr_l1 = nn.Linear(n_genes, hyp['expr_enc'])
        self.mid = nn.Linear(hyp['expr_enc'] + hyp['common_dim'], hyp['mid'])
        self.out = nn.Linear(hyp['mid'], 1)

        if hyp['drop'] == 0:
            drop=[0,0]
        else:
            drop=[0.2,0.5]

        self.in_drop = nn.Dropout(drop[0])
        self.mid_drop = nn.Dropout(drop[1])
        self.alpha = 0.5
        

    def forward(self, blocks, drug_features, cell_features_in_network, cell_features, drug_index):
        cell_enc = F.leaky_relu(self.cell_l1(cell_features_in_network))
        drug_enc = F.leaky_relu(self.drug_l1(drug_features))
        node_features = {'drug': drug_enc, 'cell_line': cell_enc}

        h1 = self.conv1(blocks[0], node_features)
        h1 = {k: F.leaky_relu(v + self.alpha*node_features[k]) for k, v in h1.items()}

        h2 = self.conv2(blocks[1], h1)
        h2['drug'] = F.leaky_relu(h2['drug'] + self.alpha*h1['drug'])
        # h2 = {k: F.leaky_relu(v + self.alpha*h1[k]) for k, v in h2.items()}
        
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))        
        drug_enc = h2['drug'][drug_index]

        x = torch.cat([expr_enc,drug_enc],-1) # (batch, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) 
        x = self.mid_drop(x)
        out = self.out(x)
        return out

    def predict_all(self, blocks, drug_features, cell_features_in_network, cell_features):

        #print("shape of drug features {0}".format(drug_features.shape))
        #print("shape of cell features {0}".format(cell_features.shape))
        cell_enc = F.leaky_relu(self.cell_l1(cell_features_in_network))
        drug_enc = F.leaky_relu(self.drug_l1(drug_features))
        node_features = {'drug': drug_enc, 'cell_line': cell_enc}
        
        h1 = self.conv1(blocks[0], node_features)
        h1 = {k: F.leaky_relu(v + self.alpha*node_features[k]) for k, v in h1.items()}
        
        h2 = self.conv2(blocks[1], h1)
        h2['drug'] = F.leaky_relu(h2['drug'] + self.alpha*h1['drug'])
        # h2 = {k: F.leaky_relu(v + self.alpha*h1[k]) for k, v in h2.items()}
        
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        expr_enc = expr_enc.unsqueeze(1) # (batch, 1, expr_enc_size)
        drug_enc = h2['drug'].unsqueeze(0) # (1, n_drugs, drug_enc_size)
        
        expr_enc = expr_enc.repeat(1,drug_enc.shape[1],1) # (batch, n_drugs, expr_enc_size)
        drug_enc = drug_enc.repeat(expr_enc.shape[0],1,1) # (batch, n_drugs, drug_enc_size)
        #print("the shape of expr_enc is {0}".format(expr_enc.shape))
        #print("the shape of drug_enc is {0}".format(drug_enc.shape))
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        #print("the shape of x is {0}".format(x.shape))
        out = self.out(x) # (batch, n_drugs, 1)
        #print("the shape of out is {0}".format(out.shape))
        out = out.view(-1, drug_enc.shape[1])
        #print("the shape of out is {0}".format(out.shape))
        return out

    def get_drug_encoding(self, blocks, drug_features, cell_features_in_network):

        cell_enc = F.leaky_relu(self.cell_l1(cell_features_in_network))
        drug_enc = F.leaky_relu(self.drug_l1(drug_features))
        node_features = {'drug': drug_enc, 'cell_line': cell_enc}
        
        h1 = self.conv1(blocks[0], node_features)
        h1 = {k: F.leaky_relu(v + self.alpha*node_features[k]) for k, v in h1.items()}
        
        h2 = self.conv2(blocks[1], h1)
        h2['drug'] = F.leaky_relu(h2['drug'] + self.alpha*h1['drug'])
        # h2 = {k: F.leaky_relu(v + self.alpha*h1[k]) for k, v in h2.items()}
        
        drug_enc = h2['drug']
        return drug_enc

    def predict_response_matrix(self, cell_features, drug_enc):

        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        expr_enc = expr_enc.unsqueeze(1) # (batch, 1, expr_enc_size)
        drug_enc = drug_enc.unsqueeze(0) # (1, n_drugs, drug_enc_size)
        
        expr_enc = expr_enc.repeat(1,drug_enc.shape[1],1) # (batch, n_drugs, expr_enc_size)
        drug_enc = drug_enc.repeat(expr_enc.shape[0],1,1) # (batch, n_drugs, drug_enc_size)
        
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
        out = out.view(-1, drug_enc.shape[1]) # (batch, n_drugs)
        return out


# ## Run training

# In[25]:


drug_feats, cell_lines, labels, label_matrix, normalizer = initialize(drug_feature, normalize_response)
#print(cell_lines, drug_feats)
#label_matrix.columns.tolist()
#print(drug_feats)

#test_metrics = anl_test_data(drug_feats, cell_lines, labels,label_matrix, normalizer, 
#                            learning_rate, epochs, batch_size)


# In[26]:


#CCLE_cross_study

#normalizer


# In[27]:


def initialize_crossstudy(cs,drug_feat, normalize_response,LABEL_FILE, GENE_EXPRESSION_FILE, LABEL_MATRIX_FILE,
                          DRUG_DESCRIPTOR_FILE,MORGAN_FP_FILE,
                          binary=False, multitask=False):

    reset_seed(seed)
    mkdir(outroot + "/results/" + folder + cs )

    if drug_feat == 'desc' or drug_feat == 'mixed':
        DRUG_FEATURE_FILE = dataroot + DRUG_DESCRIPTOR_FILE
        drug_feats = pd.read_csv(DRUG_FEATURE_FILE, index_col=0)

        df = StandardScaler().fit_transform(drug_feats.values) # normalize
        drug_feats = pd.DataFrame(df, index=drug_feats.index, columns=drug_feats.columns)

        if drug_feat == 'mixed':
            DRUG_MFP_FEATURE_FILE = dataroot + MORGAN_FP_FILE
            drug_mfp = pd.read_csv(DRUG_MFP_FEATURE_FILE, index_col=0)
            drug_feats[drug_mfp.columns] = drug_mfp

        valid_cols = drug_feats.columns[~drug_feats.isna().any()] # remove columns with missing data
        drug_feats = drug_feats[valid_cols]
        
    else:
        DRUG_FEATURE_FILE = dataroot + MORGAN_FP_FILE
        drug_feats = pd.read_csv(DRUG_FEATURE_FILE, index_col=0)

    #print(drug_feats)
    cell_lines = pd.read_csv(GENE_EXPRESSION_FILE, index_col=0).T # need to normalize
    labels = pd.read_csv(LABEL_FILE)
    labels['cell_line'] = labels['cell_line'].astype(str)
    labels['response'] = labels['response'] 
    labels['sensitive'] = (labels['resistant'] + 1)%2
    columns = ['drug','improve_id','cell_line', 'response', 'resistant', 'sensitive']
    labels = labels[columns]
    labels = labels.loc[labels['drug'].isin(drug_feats.index)] # use only cell lines with data
    labels = labels.loc[labels['cell_line'].isin(cell_lines.index)] # use only drugs with data
    cell_lines = cell_lines.loc[cell_lines.index.isin(labels['cell_line'].unique())] # use only cell lines with labels
    drug_feats = drug_feats.loc[drug_feats.index.isin(labels['drug'].unique())]      # use only drugs in labels
    drug_list = drug_feats.index.tolist()
    label_matrix = pd.read_csv(LABEL_MATRIX_FILE, index_col=0).T 
    label_matrix = label_matrix.loc[cell_lines.index] # align the matrix
    label_matrix = label_matrix[drug_list]
    label_matrix = label_matrix.drop_duplicates()
    label_matrix = label_matrix.loc[:, ~label_matrix.columns.duplicated()]
    if normalize_response:
        ss = StandardScaler() # normalize IC50
        temp = ss.fit_transform(label_matrix.values)
        label_matrix = pd.DataFrame(temp, index=label_matrix.index, columns=label_matrix.columns)
    else:
        label_matrix = label_matrix.astype(float)
    return drug_feats, cell_lines, labels, label_matrix, standardize 


# In[63]:


GENE_EXPRESSION_FILE = "Data//BiG_DRP_data//drp-data/grl-preprocessed/sanger_tcga//BiG_DRP_fpkm.csv"
DRUG_DESCRIPTOR_FILE = 'Data//BiG_DRP_data//drp-data/grl-preprocessed/drug_features//BiG_DRP_descriptors.csv'
MORGAN_FP_FILE = 'Data//BiG_DRP_data//drp-data/grl-preprocessed/drug_features//BiG_DRP_morgan.csv'
LABEL_FILE = "Data/BiG_DRP_data/drp-data/grl-preprocessed/drug_response/BiG_DRP_tuple_labels_folds.csv" 
LABEL_MATRIX_FILE = "Data//BiG_DRP_data//drp-data/grl-preprocessed/drug_response/BiG_DRP_data_cleaned.csv"

CCLE_LABEL_file = "Data/BiG_DRP_data/cross_study/CCLE_tuples_test.csv"
CCLE_LABEL_MATRIX_FILE = "Data/BiG_DRP_data/cross_study/CCLE_cleaned_test.csv"

GDSCv1_LABEL_FILE = "Data/BiG_DRP_data/cross_study/GDSCv1_tuples_test.csv"
GDSCv1_LABEL_MATRIX_FILE = "Data/BiG_DRP_data/cross_study/GDSCv1_cleaned_test.csv"

GCSI_LABEL_FILE = "Data/BiG_DRP_data/cross_study/gCSI_tuples_test.csv"
GCSI_LABEL_MATRIX_FILE = "Data/BiG_DRP_data/cross_study/gCSI_cleaned_test.csv"

GDSCv2_LABEL_FILE = "Data/BiG_DRP_data/cross_study/GDSCv2_tuples_test.csv"
GDSCv2_LABEL_MATRIX_FILE = "Data/BiG_DRP_data/cross_study/GDSCv2_cleaned_test.csv"
model_weights_path = "/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/results/results/model_weights_fold.pt" 
hyperparams = {
        'learning_rate': learning_rate,
        'num_epoch': epochs,
        'batch_size': batch_size,
        'common_dim': 512,
        'expr_enc': 1024,
        'conv1': 512,
        'conv2': 512,
        'mid': 512,
        'drop': 1}

drug_feat ="desc"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[62]:


ccle_drug_feats, ccle_cell_lines, ccle_labels, ccle_label_matrix, standardize= initialize_crossstudy("CCLE",drug_feat,
                                                                                                     normalize_response, 
                                                                                                     CCLE_LABEL_file, 
                                                                                                     GENE_EXPRESSION_FILE, 
                                                                                                     CCLE_LABEL_MATRIX_FILE,
                                                                                                     DRUG_DESCRIPTOR_FILE,
                                                                                                     MORGAN_FP_FILE)


ccle_tuples = pd.read_csv(CCLE_LABEL_file)
ccle_label_matrix = ccle_label_matrix.replace(np.nan, 0)
ccle_final_metrics = None
ccle_drug_list = list(ccle_drug_feats.index)
ccle_samples = list(ccle_labels['cell_line'].unique())
ccle_tuples = ccle_tuples[['drug', 'cell_line', 'response']]
ccle_tuples = reindex_tuples(ccle_tuples, ccle_drug_list, ccle_samples)
ccle_x = ccle_cell_lines.loc[ccle_samples].values
ccle_y = ccle_label_matrix.loc[ccle_samples].values

ss = StandardScaler()
ccle_x = StandardScaler().fit_transform(ccle_x)

ccle_y = StandardScaler().fit_transform(ccle_y)
graph_sampler = MultiLayerFullNeighborSampler(2)
ccle_network = create_network(ccle_tuples, network_percentile)

ccle_drug_feats_tensor = torch.FloatTensor(ccle_drug_feats.values).to(device)  # Assuming 'device' is defined
ccle_cell_lines_tensor = torch.FloatTensor(ccle_cell_lines.values).to(device)
ccle_network = ccle_network.to(device)
ccle_network.ndata['features'] = {'drug': ccle_drug_feats_tensor, 'cell_line': ccle_cell_lines_tensor}
_,_, ccle_blocks = graph_sampler.sample_blocks(ccle_network, {'drug': range(len(ccle_drug_feats)), 
                                                              'cell_line': range(len(ccle_cell_lines))})                                            
ccle_blocks = [b.to(device) for b in ccle_blocks]

hyp = hyperparams.copy()
n_genes = ccle_cell_lines.shape[1]
n_drug_feats = ccle_drug_feats.shape[1]
n_cell_feats = cell_lines.shape[1]
drug_feats_tensor = torch.FloatTensor(drug_feats.values)
cell_line_feats_tensor = torch.FloatTensor(ccle_x)
ccle_data = TensorDataset(torch.FloatTensor(ccle_x),torch.FloatTensor(ccle_y))
ccle_loader = DataLoader(ccle_data, batch_size=hyp['batch_size'], shuffle=False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BiGDRP(n_genes, n_cell_feats, n_drug_feats, ccle_network.etypes, hyp)

 # Update with the correct path


model.load_state_dict(torch.load(model_weights_path, map_location=device))  # Use map_location to ensure models are loaded to the correct device
model = model.to(device)
model.eval()

drug_enco = model.get_drug_encoding(ccle_blocks, ccle_drug_feats_tensor, ccle_cell_lines_tensor)
ccle_results = []
ccle_ys = []
with torch.no_grad():
    for (x, y) in ccle_loader:
        x = x.to(device)
        y = y.to(device)
        drug_encoding = drug_enco.to(device)
        pred = model.predict_response_matrix(x, drug_encoding)
        ccle_ys.append(y.cpu().detach().numpy())
        ccle_results.append(pred.cpu().detach().numpy())
        r2 = r2_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
ccle_ys = np.concatenate(ccle_ys)
ccle_results = np.concatenate(ccle_results)
min_rows = min(ccle_ys.shape[0], ccle_results.shape[0])
ccle_results = ccle_results[:min_rows]
ccle_ys = ccle_ys[:min_rows]

ccle_r2 = r2_score(ccle_ys, ccle_results)
print(ccle_r2)


# In[67]:


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
    study_drug_feats_tensor = torch.FloatTensor(study_drug_feats.values).to(device)  # Assuming 'device' is defined
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
    model.load_state_dict(torch.load(model_weights_path, map_location=device))  # Use map_location to ensure models are loaded to the correct device
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


# In[69]:


run_infer("GDSCv1", GDSCv1_LABEL_FILE, GDSCv1_LABEL_MATRIX_FILE)
run_infer("GDSCv2", GDSCv2_LABEL_FILE, GDSCv2_LABEL_MATRIX_FILE)
run_infer("GCSI", GCSI_LABEL_FILE, GCSI_LABEL_MATRIX_FILE)


# In[ ]:




