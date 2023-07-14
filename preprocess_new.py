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
from sklearn.metrics import r2_score
import improve_utils
file_path = os.path.dirname(os.path.realpath(__file__))

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

def preprocess(params, data_dir):
    preprocessed_dir = data_dir + "/preprocessed"
    drug_feature_dir = data_dir + "/drp-data/grl-preprocessed/drug_features/"
    drug_response_dir = data_dir + "/drp-data/grl-preprocessed/drug_response/"
    sanger_tcga_dir = data_dir + "/drp-data/grl-preprocessed/sanger_tcga/"    
    mkdir(drug_feature_dir)
    mkdir(drug_response_dir)
    mkdir(sanger_tcga_dir)
    mkdir(preprocessed_dir)

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

def download_anl_data(params):
    csa_data_folder = os.path.join(os.environ['CANDLE_DATA_DIR'] + params['model_name'], 'csa_data', 'raw_data')
    splits_dir = os.path.join(csa_data_folder, 'splits') 
    x_data_dir = os.path.join(csa_data_folder, 'x_data')
    y_data_dir = os.path.join(csa_data_folder, 'y_data')
    improve_data_url = params['improve_data_url']
    data_type = params['data_type']
    
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
        improve_file = data_type + files
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

    
def download_author_data(params, data_dir):
    print("downloading file: %s"%params['data_url'])
    data_download_filepath = candle.get_file(params['original_data'], params['data_url'],
                                             datadir = data_dir,
                                             cache_subdir = None)
    return(params)


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

    
def create_data_inputs(params):
    data_input = params['data_input']
    binary_input = params['binary_input']
    data_type = params['data_type']
    metric = params['metric']
    print(metric)
    auc_threshold = params['auc_threshold']
    rs = improve_utils.load_single_drug_response_data(source=data_type, split=0,
                                                      split_type=["train", "val", "test"],
                                                      y_col_name=metric)
    #generate binary file
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
    rs_binary_df.to_csv(binary_input, index=None)
    rs_tdf = rs_tdf.reset_index()
    rs_tdf = rs_tdf.rename({'compounds': "sample_names"},axis=1)
    rs_tdf.to_csv(data_input, index_label="improve_id")
   
def process_expression(tcga_file, expression_out):
    ge = improve_utils.load_gene_expression_data(gene_system_identifier="Ensembl")
    expression_tdf = ge.dropna()
    expression_threshold = np.floor(0.1*len(expression_tdf))
    to_keep = (expression_tdf.astype(float) > 1).sum() > expression_threshold
    f_log = (expression_tdf.loc[:,to_keep].astype(float) +1).apply(np.log2)
    f_log = f_log.loc[:, f_log.std() > 0.1]
    tcga = pd.read_csv(tcga_file, index_col=0)
    common = tcga.index.intersection(f_log.columns)
    fpkm_df = f_log[common].T
    fpkm_df.index.names = ['ensembl_gene_id']
    fpkm_df= fpkm_df.reset_index()
    fpkm_df.to_csv(expression_out, index=None)
    

def creating_drug_and_smiles_input(smiles_out, drug_synonyms_out):
    se = improve_utils.load_smiles_data()
    se = se.set_index("improve_chem_id")
    se.to_csv(smiles_out, index_label=None)
    #creating the synonmys file
    improve_chem_list = se.index.tolist()
    syn_list = []
    for x in improve_chem_list:
        str_x = "imporve_drug_for_" + str(x)
        syn_list.append(str_x)
    se['syn'] = syn_list
    drug_syn_df = se.drop(['smiles'], axis=1)
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

def preprocess_gdsc(params):
    ic50_file = params['data_input']
    binary_file = params['binary_input']
    drug_synonyms = params['drug_synonyms']
    fpkm_file = params['fpkm_file']
    data_cleaned_out = params['data_cleaned_out']
    data_bin_cleaned_out = params['data_bin_cleaned_out']
    data_tuples_out = params['data_tuples_out']
    syn = pd.read_csv(drug_synonyms, header=None)
    cells = pd.read_csv(fpkm_file, index_col=0).columns
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
    print("Generated tuple labels {0}".format(OUTFILE))

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


def generate_morganprint(smiles_file, morgan_out):
    OUTFILE = morgan_out
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


def generate_splits(params):
    expression_out = params['fpkm_file']
    data_bin_cleaned_out = params['data_bin_cleaned_out']
    morgan_out = params['morgan_data_out']
    tuples_label_out = params['tuples_label_out']
    tuples_label_fold_out = params['tuples_label_fold_out']
    cells = pd.read_csv(expression_out
                        , index_col=0).columns
    labels = pd.read_csv(data_bin_cleaned_out, index_col=0)
    drugs = pd.read_csv(morgan_out, index_col=0)
    orig_cells_gex = len(cells)
    orig_cells_lab = labels.shape[1]
    common = list(set(cells).intersection(set(labels.columns)))
    labels = labels[common] # (drugs x cells)

    drug_list = labels.index.tolist()
    drug_out =  params['drug_out'] 
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
    lco = leave_cells_out(labels, common, cells)

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

    lpo = leave_pairs_out(lpo_tuples, drugs)
    df = tuples.copy()
    df.at[lpo.index, 'pair_fold'] = lpo['pair_fold']
    df['cl_fold'] = np.zeros(len(df), dtype=int)

    for i in range(lco['fold'].max()+1):
        cells_in_fold = lco.loc[lco['fold']==i].index
        df.loc[df['cell_line'].isin(cells_in_fold), 'cl_fold'] = i

    df.index = range(len(df))
    df['pair_fold'] = df['pair_fold'].replace(np.nan, -1).astype(int)
    df.to_csv(tuples_label_fold_out)
    

def write_out_constants(params):
    constant_file = "utils/constants.py"
    with open(constant_file, 'w+') as fout:
        _LABEL_FILE = params['tuples_label_fold_out'].split("Data")[1] 
        _GENE_EXPRESSION_FILE = params['fpkm_file'].split("Data")[1]
        _LABEL_MATRIX_FILE = params['data_cleaned_out'].split("Data")[1]
        _DRUG_DESCRIPTOR_FILE = params['descriptor_out'].split("Data")[1]
        _MORGAN_FP_FILE = params['morgan_data_out'].split("Data")[1]
        fout.write("_LABEL_FILE = '{0}'".format(_LABEL_FILE) + '\n')
        fout.write("_GENE_EXPRESSION_FILE = '{0}'".format(_GENE_EXPRESSION_FILE) + '\n')
        fout.write("_LABEL_MATRIX_FILE = '{0}'".format(_LABEL_MATRIX_FILE) + '\n')
        fout.write("_DRUG_DESCRIPTOR_FILE = '{0}'".format(_DRUG_DESCRIPTOR_FILE) + '\n')
        fout.write("_MORGAN_FP_FILE = '{0}'".format(_MORGAN_FP_FILE) + '\n')        
        
def run(params):
    params['data_type'] = str(params['data_type'])
    json_out = params['output_dir']+'/params.json'
    try:
        with open (json_out, 'w') as fp:
            json.dump(params, fp, indent=4, cls=MyEncoder)
    except AttributeError:
        pass
    print(params)

    
def candle_main(anl):
    params = initialize_parameters()
    data_dir = os.environ['CANDLE_DATA_DIR'] + params['model_name'] + "/Data/BiG_DRP_data/"
    params =  preprocess(params, data_dir)
    run(params)
    if params['improve_analysis'] == 'yes' or anl:
        download_anl_data(params)
        create_data_inputs(params)
        download_author_data(params, data_dir)
        creating_drug_and_smiles_input(params['smiles_file'], params['drug_synonyms'])
        process_expression(params['tcga_file'], params['fpkm_file'])
        preprocess_gdsc(params)
        generate_drug_descriptors(params['smiles_file'], params['descriptor_out'])
        generate_morganprint(params['smiles_file'], params['morgan_data_out'])
        generate_splits(params)
        write_out_constants(params)
    else:
        download_author_data(params, data_dir)
        write_out_constants(params)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-a', dest='anl',  default=False)
    args = parser.parse_args()
    candle_main(args.anl)

