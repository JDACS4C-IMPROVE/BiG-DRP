from utils.utils import mkdir, reset_seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import utils.constants as c

def standardize(train_x, test_x):
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.transform(test_x)
    return train_x, test_x


def initialize(FLAGS, LABEL_FILE, GENE_EXPRESSION_FILE, LABEL_MATRIX_FILE, DRUG_DESCRIPTOR_FILE, MORGAN_FP_FILE,
               binary=False, multitask=False):

    reset_seed(FLAGS.seed)
    mkdir(FLAGS.outroot + "/results/" + FLAGS.folder)

#    LABEL_FILE = FLAGS.dataroot  +  
#    print(LABEL_FILE)
#    GENE_EXPRESSION_FILE = FLAGS.dataroot +  c._GENE_EXPRESSION_FILE
#    print(GENE_EXPRESSION_FILE)
#    LABEL_MATRIX_FILE = FLAGS.dataroot + c._LABEL_MATRIX_FILE
#    print(LABEL_MATRIX_FILE)
    
    if FLAGS.drug_feat == 'desc' or FLAGS.drug_feat == 'mixed':
#        DRUG_FEATURE_FILE = FLAGS.dataroot + c._DRUG_DESCRIPTOR_FILE
        drug_feats = pd.read_csv(DRUG_DESCRIPTOR_FILE, index_col=0)
#        print(drug_feats)
#        drug_feats = pd.read_csv(DRUG_DESCRIPTOR_FILE)
        df = StandardScaler().fit_transform(drug_feats.values) # normalize
        drug_feats = pd.DataFrame(df, index=drug_feats.index, columns=drug_feats.columns)

        if FLAGS.drug_feat == 'mixed':
#            DRUG_MFP_FEATURE_FILE = FLAGS.dataroot + c._MORGAN_FP_FILE
            drug_mfp = pd.read_csv(MORGAN_FP_FILE, index_col=0)
            drug_feats[drug_mfp.columns] = drug_mfp

        valid_cols = drug_feats.columns[~drug_feats.isna().any()] # remove columns with missing data
        drug_feats = drug_feats[valid_cols]
        
    else:
#        DRUG_FEATURE_FILE = FLAGS.dataroot + c._MORGAN_FP_FILE
        drug_feats = pd.read_csv(DRUG_DESCRIPTOR_FILE, index_col=0)

    cell_lines = pd.read_csv(GENE_EXPRESSION_FILE, index_col=0).T # need to normalize
    labels = pd.read_csv(LABEL_FILE, index_col=0)
    labels['cell_line'] = labels['cell_line'].astype(str)
    labels['response'] = labels['auc']
    print(labels.columns)
    #Index(['drug', 'cosmic_id', 'cell_line', 'ln_ic50', 'auc', 'resistant',
    #   'sensitive', 'pair_fold', 'cl_fold', 'response'],
    #  dtype='object')
    columns = ['drug','cosmic_id','cell_line', 'response', 'resistant', 'sensitive','cl_fold']
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
    if FLAGS.normalize_response:
        ss = StandardScaler() # normalize IC50
        temp = ss.fit_transform(label_matrix.values)
        label_matrix = pd.DataFrame(temp, index=label_matrix.index, columns=label_matrix.columns)
    else:
        label_matrix = label_matrix.astype(float)
    return drug_feats, cell_lines, labels, label_matrix, standardize
#    cell_lines = pd.read_csv(GENE_EXPRESSION_FILE, index_col=0).T # need to normalize
#    labels = pd.read_csv(LABEL_FILE)
#    labels['cell_line'] = labels['cell_line'].astype(str)
#    labels['response'] = labels['ln_ic50'] 

#    labels = labels.loc[labels['drug'].isin(drug_feats.index)] # use only cell lines with data
#    labels = labels.loc[labels['cell_line'].isin(cell_lines.index)] # use only drugs with data
#    cell_lines = cell_lines.loc[cell_lines.index.isin(labels['cell_line'].unique())] # use only cell lines with labels
#    drug_feats = drug_feats.loc[drug_feats.index.isin(labels['drug'].unique())]      # use only drugs in labels
    
#    label_matrix = pd.read_csv(LABEL_MATRIX_FILE, index_col=0).T 
#    label_matrix = label_matrix.loc[cell_lines.index][drug_feats.index] # align the matrix

#    if FLAGS.normalize_response:
#        ss = StandardScaler() # normalize IC50
#        temp = ss.fit_transform(label_matrix.values)
#        label_matrix = pd.DataFrame(temp, index=label_matrix.index, columns=label_matrix.columns)
#    else:
#        label_matrix = label_matrix.astype(float)


#    if FLAGS.split == 'lpo': # leave pairs out
#        labels['fold'] = labels['pair_fold']
#    else: # default: leave cell lines out
#        labels['fold'] = labels['cl_fold']

#    print('tuples per fold:')
#    print(labels.groupby('fold').size())

#    return drug_feats, cell_lines, labels, label_matrix, standardize

def initialize_crossstudy(cs,drug_feat, normalize_response,
                          LABEL_FILE, GENE_EXPRESSION_FILE, LABEL_MATRIX_FILE,
                          DRUG_DESCRIPTOR_FILE, MORGAN_FP_FILE,
                          binary=False, multitask=False):

    reset_seed(seed)
    mkdir(outroot + "/results/" + folder + cs )

    if drug_feat == 'desc' or drug_feat == 'mixed':
#        DRUG_FEATURE_FILE = dataroot + DRUG_DESCRIPTOR_FILE
        drug_feats = pd.read_csv(DRUG_DESCRIPTOR_FILE, index_col=0)

        df = StandardScaler().fit_transform(drug_feats.values) # normalize
        drug_feats = pd.DataFrame(df, index=drug_feats.index, columns=drug_feats.columns)

        if drug_feat == 'mixed':
#            DRUG_MFP_FEATURE_FILE = dataroot + MORGAN_FP_FILE
            drug_mfp = pd.read_csv(MORGAN_FP_FILE, index_col=0)
            drug_feats[drug_mfp.columns] = drug_mfp

        valid_cols = drug_feats.columns[~drug_feats.isna().any()] # remove columns with missing data
        drug_feats = drug_feats[valid_cols]
        
    else:
#        DRUG_FEATURE_FILE = dataroot + MORGAN_FP_FILE
        drug_feats = pd.read_csv(DRUG_DESCRIPTOR_FILE, index_col=0)

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
