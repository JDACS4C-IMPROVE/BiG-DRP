import candle
import os
import json
import shutil
from json import JSONEncoder
from utils.utils import mkdir
from Big_DRP_train import main
os.environ['NUMEXPR_MAX_THREADS']='6'
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
import time
from scipy.stats import pearsonr, spearmanr
from collections import deque
from sklearn.metrics import r2_score
import improve_utils
file_path = os.path.dirname(os.path.realpath(__file__))


# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/'

#parent path
fdir = Path('__file__').resolve().parent

# additional definitions
#additional_definitions = [
#    {
#        "name": "batch_size",
#        "type": int,
#        "help": "...",
#    },
#    {
#        "name": "learning_rate",
#        "type": int,
#        "help": "learning rate for the model",
#    },
#    {   
#        "name": "epoch",
#        "type": int,
#        "help": "number of epochs to train on",
#    },
#    {
#        "name": "network_percentile",
#        "type": int,
#        "help": "network percentile for metrics",
#    },
#    {   
#        "name": "cuda",
#        "type": int, 
#        "help": "CUDA ID",
#    },
#]

# required definitions
#required = None



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

def preprocess(params):
    data_dir = os.environ['CANDLE_DATA_DIR'] + "/BiG-DRP/Data/"
#    preprocessed_dir = os.environ['CANDLE_DATA_DIR'] + "/BiG-DRP/Data/preprocessed"
#    drug_feature_dir = data_dir + "/drp-data/grl-preprocessed/drug_features/"
#    drug_response_dir = data_dir + "/drp-data/grl-preprocessed/drug_response/"
#    sanger_tcga_dir = data_dir + "/drp-data/grl-preprocessed/sanger_tcga/"    
    mkdir(drug_feature_dir)
    mkdir(drug_response_dir)
    mkdir(sanger_tcga_dir)
    mkdir(preprocessed_dir)

    model_param_key = []
    for key in params.keys():
        if key not in keys_parsing:
                model_param_key.append(key)
    model_params = {key: params[key] for key in model_param_key}
    params['model_params'] = model_params
    args = candle.ArgumentStruct(**params)
    drug_synonym_file = data_dir + "/" + params['drug_synonyms']
    gene_expression_file = data_dir + "/" + params['rnaseq_fpkm']
    gene_indentifiers_file = data_dir + "/" + params['gene_identifiers']
    enst_file =  data_dir + "/" + params['enst_list']
    ln50_file = data_dir + "/" + params['lnic50_file']
    model_label_file = data_dir + "/" + params['binary_file']
    tcga_file =  data_dir + "//supplementary/" + params['tcga_file']
    smiles_file =  data_dir + "/supplementary/" + params['smiles_file']
    params['fpkm_file'] = data_dir + "/preprocessed/sanger_fpkm.csv"
    params['model_label_file'] = model_label_file
    params['enst_file'] = enst_file
    params['smiles_file'] =  smiles_file
    params['model_label_file'] = model_label_file
    params['smiles_file'] = smiles_file
    params['tcga_file'] = tcga_file
    params['dataroot'] = data_dir
    params['folder'] = params['outroot']
    params['outroot'] = params['outroot']
    params['network_perc'] = params['network_percentile']
    params['drug_feat'] = params['drug_feature']
    params['drug_synonym'] = drug_synonym_file
    return(params)


class MyEncoder(JSONEncoder):
    def default(self, obj):
        return obj.__dict__ 

#    drp_params = dict((k, params[k]) for k in ('dataroot', 'drug_feat',
#                                               'folder', 'mode', 'network_perc',
#                                               'normalize_response', 'outroot', 'seed',
#                                               'split', 'weight_folder'))
#    scores = main(drp_params, params['learning_rate'], params['epochs'], params['batch_size'])
#    with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
#        json.dump(scores, f, ensure_ascii=False, indent=4)
#    print('IMPROVE_RESULT RMSE:\t' + str(scores['rmse']))


#def processes_geneexpression(gene_identifiers, rnaseq_fpkm, enst_list, tcga_file):
#    filter_enst_df = pd.read_csv(enst_list)
#    filter_enst_list = filter_enst_df['ensembl_gene_id'].tolist()
#    fpkm_chunk = pd.read_csv(rnaseq_fpkm, chunksize=500000, index_col=0)
#    fpkm_df = pd.concat(fpkm_chunk)
#    fpkm = fpkm_df.copy()
#    gene_ids = pd.read_csv(gene_identifiers,sep='\t', index_col=0)
#    gene_ids = gene_ids.dropna(subset=['ensembl_gene_id'])
#    fpkm = fpkm.loc[fpkm.index.isin(gene_ids.index)] # remove those with no conversion
#    fpkm['ensembl_gene_id'] = gene_ids.loc[fpkm.index]['ensembl_gene_id'] 
#    fpkm = fpkm.drop_duplicates(subset=['ensembl_gene_id']) # remove duplicates
#    fpkm.index=fpkm['ensembl_gene_id'] # use ensembl as ID
#    fpkm =fpkm.drop('symbol', axis=1)
#    fpkm = fpkm.loc[fpkm.index.isin(filter_enst_list)]
#    fpkm = fpkm.drop('ensembl_gene_id', axis=1)
#    fpkm = fpkm.dropna()
#    fpkm = fpkm.T
##    fpkm_num_thresh = np.floor(0.1*len(fpkm))
##    to_keep = (fpkm.astype(float) > 1).sum() > fpkm_num_thresh
##    f_log = (fpkm.loc[:,to_keep].astype(float) +1).apply(np.log2)
##    f_log = f_log.loc[:, f_log.std() > 0.1]
##    tcga = pd.read_csv(tcga_file, index_col=0)
##    common = tcga.index.intersection(f_log.columns)
##    preprocess_out = "Data/preprocessed/sanger_fpkm.csv"
##    f_log[common].T.to_csv(preprocess_out)


def process_expression(tcga_file):
    expression_df = improve_utils.load_gene_expression_data(gene_system_identifier="Ensembl")
    expression_tdf = expression_df.T
    expression_tdf = expression_tdf.dropna()
    expression_threshold = np.floor(0.1*len(expression_tdf))
    to_keep = (expression_tdf.astype(float) > 1).sum() > expression_threshold)
    f_log = (expression_tdf.loc[:,to_keep].astype(float) +1).apply(np.log2)
    f_log = f_log.loc[:, f_log.std() > 0.1]
    tcga = pd.read_csv(tcga_file, index_col=0)
    common = tcga.index.intersection(f_log.columns)
    preprocess_out = "Data/preprocessed/sanger_fpkm.anl.csv"
    f_log[common].T.to_csv(preprocess_out)
    
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

    x = df.loc[df[drug_col] == 'bleomycin (50 um)']
    to_remove.append(x.index[0])
    df = df.loc[~df.index.isin(to_remove)]

    ## Rename synonyms
    
    df = df.copy()
    print(df)
    for i in syn.index:
        x = syn.loc[i, 0]
        y = syn.loc[i, 1]
        df[drug_col] = df[drug_col].replace(x, y)

        ## Find duplicates
    dups = df.loc[df.duplicated(drug_col, keep=False)]
    not_dups = df.loc[~df.index.isin(dups.index)].index

        ## Keep those with more labels
    to_keep = []
    for d in dups[drug_col].unique():
        print(d)
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
    df = df.loc[df.index!='cosmic_id']

    return df

## Preprocess GDSC data

def preprocess_gdsc(ic50_file,binary_file,drug_synonyms, fpkm_file):
    # LN_IC50 DATA
    print("Processing lnIC50 data...")
    syn = pd.read_csv(drug_synonyms, header=None)
    cells = pd.read_csv(fpkm_file, index_col=0).columns
    lnic50 = pd.read_csv(ic50_file, index_col=1, header=None)
    lnic50 = lnic50.replace(681640, '681640') # int to string just for this specific drug
    cells = lnic50.index[1:]
    lnic50.loc[cells, range(2, 267)] = lnic50.loc[cells, range(2, 267)].astype(float)
    lnic50.loc['sample_names'] = lnic50.loc['sample_names'].astype(str).str.lower()
    df = lnic50.T
    df = filter_labels(df, syn, cells, drug_col='sample_names')
    df = df.sort_index()[cells]
    df.to_csv('Data/preprocessed/gdsc_ln_ic50_cleaned.csv')
    print("lnIC50 matrix size:", df.shape)
    drugs = list(df.index)

    # BINARIZED DATA
    print("\n\nProcessing binarized data...")
    bin_data = pd.read_csv(binary_file, index_col=0, header=None)
    bin_data = bin_data.replace(681640, '681640') # int to string just for this specific drug
    bin_data.loc['compounds'] = bin_data.loc['compounds'].str.lower()
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
    OUTFILE = "Data/preprocessed/gdsc_bin_cleaned.csv"
    bin_data.to_csv(OUTFILE)
    print("Generated cleaned gdsc bin file {0}".format(OUTFILE))
    print("Binarized matrix size:", bin_data.shape)


    # TUPLE DATA
    print("Processing tuple data...")
    tuples = pd.DataFrame(columns=['drug', 'cosmic_id', 'cell_line', 'ln_ic50', 'resistant'])
    idx = np.transpose((1-np.isnan(np.asarray(df.values, dtype=float))).nonzero())

    print("num tuples:",  len(idx))
    i = 0
    for drug, cl in tqdm(idx):
        x = {'drug': drugs[drug],
            'cosmic_id': lnic50.loc[cells[cl]][0],
            'cell_line': cells[cl],
            'ln_ic50': df.loc[drugs[drug], cells[cl]],
            'resistant': bin_data.loc[drugs[drug], cells[cl]]}
        tuples.loc[i] = x
        i += 1
    OUTFILE = "Data/preprocessed/gdsc_tuple_labels.csv"
    tuples.to_csv(OUTFILE)
    print("Generated gdsc tuple labels {0}".format(OUTFILE))

def generate_drug_smiles():
    drugs = pd.read_csv('Data/preprocessed/gdsc_bin_cleaned.csv', index_col=0).index

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
    OUTFILE = "Data/preprocessed/gdsc_drugs_smiles_250.csv"
    drugs_smiles.to_csv(OUTFILE)

def generate_morganprint(smiles_file):
    OUTFILE = 'Data/preprocessed/gdsc_250_morgan.csv'
    smiles = pd.read_csv(smiles_file, index_col=0)
    smiles = smiles.dropna()
    new_df = pd.DataFrame(index=smiles.index, columns=range(512))
    for drug in smiles.index:
        x = smiles.loc[drug]['smiles']
        mol = Chem.MolFromSmiles(x)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp,arr)
        new_df.loc[drug] = arr 
    
    new_df.to_csv(OUTFILE)
    print("morgan fingerprint file generated at {0}".format(OUTFILE))


def generate_drug_descriptors(smiles_file):
    OUTFILE = 'Data/preprocessed/gdsc_250_drug_descriptors.csv'

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

def generate_splits():
    cells = pd.read_csv('Data/preprocessed/sanger_fpkm.csv', index_col=0).columns
    labels = pd.read_csv('Data/preprocessed/gdsc_bin_cleaned.csv', index_col=0)
    drugs = pd.read_csv('Data/preprocessed/gdsc_250_morgan.csv', index_col=0)

    orig_cells_gex = len(cells)
    orig_cells_lab = labels.shape[1]
    common = list(set(cells).intersection(set(labels.columns)))
    labels = labels[common] # (drugs x cells)


    print('original cells (GEX): %d'%(orig_cells_gex))
    print('original cells (labels): %d'%(orig_cells_lab))
    print('current cells: %d'%len(common) )

    orig_drugs = len(drugs)
    drugs = drugs.dropna().index
    labels = labels.loc[drugs]

    print('original drugs: %d'%(orig_drugs))
    print('current drugs: %d (dropped %d)'%(len(drugs), orig_drugs-len(drugs)))

    print("Doing leave cell lines out splits...")

    lco = leave_cells_out(labels, common, cells)

    ### leave pairs

    labels = labels[common].T

    # print('current cells: %d'%len(labels.index) )

    tuples = pd.read_csv('Data/preprocessed/gdsc_tuple_labels.csv', index_col=0)
    tuples['resistant'] = tuples['resistant'].astype(int)
    tuples['sensitive'] = (tuples['resistant'] + 1)%2
    # remove tuples that don't have drug or cell line data
    tuples = tuples.loc[tuples['drug'].isin(drugs)].loc[tuples['cell_line'].isin(labels.index)] 

    print("number of tuples before filter:", tuples.shape[0])
    print("removed cell lines with < 3 drugs tested...")
    labels = labels.loc[labels.notna().sum(axis=1) > 2]
    lpo_tuples = tuples.loc[tuples['cell_line'].isin(labels.index)].copy()
    print("number of tuples after filter:", lpo_tuples.shape[0])

    lpo = leave_pairs_out(lpo_tuples, drugs)

    df = tuples.copy()
    df.at[lpo.index, 'pair_fold'] = lpo['pair_fold']
    df['cl_fold'] = np.zeros(len(df), dtype=int)

    for i in range(lco['fold'].max()+1):
        cells_in_fold = lco.loc[lco['fold']==i].index
        df.loc[df['cell_line'].isin(cells_in_fold), 'cl_fold'] = i

    df.index = range(len(df))
    df['pair_fold'] = df['pair_fold'].replace(np.nan, -1).astype(int)
    df.to_csv('Data/preprocessed/gdsc_tuple_labels_folds.csv')


def run(params):
    params['data_type'] = str(params['data_type'])
    json_out = params['output_dir']+'/params.json'
    try:
        with open (json_out, 'w') as fp:
            json.dump(params, fp, indent=4, cls=MyEncoder)
    except AttributeError:
        pass
    print(params)

    processes_geneexpression(params['gene_identifiers'],
                             params['rnaseq_fpkm'],
                             params['enst_list'], params['tcga_file'])
#    print(params['fpkm_file'])
    preprocess_gdsc(params['lnic50_file'], params['binary_file'],
                    params['drug_synonyms'], params['fpkm_file'])
    generate_morganprint(params['smiles_file'])
    generate_drug_descriptors(params['smiles_file'])
    generate_splits()

    
def candle_main():
    params = initialize_parameters()
    params =  preprocess(params)
    run(params)


if __name__ == "__main__":
    candle_main()
