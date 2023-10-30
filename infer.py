import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from argparse import Namespace
from utils.data_initializer import initialize
from utils.data_initializer import initialize_crossstudy
import json
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
#import train as bmk
import candle
import torch
from torch.utils.data import DataLoader, TensorDataset
#from bigdrp.trainer import Trainer
#from utils.tuple_dataset import TupleMatrixDataset

# Just because the tensorflow warnings are a bit verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_path = os.path.dirname(os.path.realpath(__file__))
required=None
additional_definitions=None

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


def launch(args):
    folder = args.folder
    data_dir = os.environ['CANDLE_DATA_DIR'] + "/Data/"
    drug_response_dir = data_dir + "BiG_DRP_data/drp-data/grl-preprocessed/drug_response/"
    drugset = data_dir + "/" + args.drugset
    # load labels
    labels = drug_response_dir + "/" + args.labels
    y_tup = pd.read_csv(labels, index_col=0)
    if args.split == 'lpo':
        y_tup['fold'] = y_tup['pair_fold']
    else:
        y_tup['fold'] = y_tup['cl_fold']
        
    y_tup = y_tup.loc[y_tup['fold']>=0]
    y = y_tup.pivot(index='cell_line', columns='drug', values='response')
    y_bin = y_tup.pivot(index='cell_line', columns='drug', values='resistant')
    
    samples = list(y_tup['cell_line'].unique())

    # load drugs
    drugs = open(drugset).read().split('\n')
    if drugs[-1] == '': drugs=drugs[:-1]

    # filter out unnecessary samples/drugs
    y_tup = y_tup.loc[y_tup['drug'].isin(drugs)]
    y = y.loc[samples, drugs]
    y_bin = y_bin.loc[samples, drugs] # binary response
    
    y0 = y.replace(np.nan, 0)
    null_mask = y0.values.nonzero()
    y_norm = (y - y.mean())/y.std()   # normalized response

    print("calculating for %d drugs and %d cell lines..."%(len(drugs), len(samples)))
        
    # create mask for folds
    # NOTE: This code assumes that all (drug, CCL) pairs can only exist in 1 fold
    fold_mask = y_tup.pivot(index='cell_line', columns='drug', values='fold')
    fold_mask = fold_mask.loc[samples, drugs]
    
    # load predictions
    _, df = load_predictions(folder, split=args.split, fold_mask=fold_mask)
    df = df.loc[samples, drugs]
    preds_norm = df                      # actual prediction for normalized response
    preds_unnorm = df*y.std() + y.mean() # if we revert back to unnormalized response
    
    # Calculate overall metrics
    print('calculating overall metrics...')
    
    mets = ["spearman (fold.%d)"%i for i in range(5)]
    overall = pd.DataFrame(index=mets, columns=['normalized %s'%args.response, 'raw %s'%args.response])
    scores = overall
    s = np.zeros((5, 2))
    for i in range(5):
        m  = ((fold_mask == i)*1).values.nonzero()
        s[i, 0] = spearmanr(y_norm.values[m], preds_norm.values[m])[0]
        s[i, 1] = spearmanr(y.values[m], preds_unnorm.values[m])[0]
    overall.loc[mets] = s
    overall.loc['spearman (fold.mean)'] = s.mean(axis=0)
    overall.loc['spearman (fold.stdev)'] = s.std(axis=0)
    outfile = '%s/%s_performance_%d_drugs.xlsx'%(args.results_dir, args.split, len(drugs))
    exwrite = pd.ExcelWriter(outfile)#, engine='xlsxwriter')
    overall.to_excel(exwrite, sheet_name='Overall')
        
    if args.mode == 'collate':
        per_drug_metric = get_per_drug_metric(preds_norm, y_norm, y_bin)
    elif args.mode == 'per_fold':
        per_drug_metric = get_per_drug_fold_metric(preds_norm, y_norm, fold_mask, y_bin)
    per_drug_metric = per_drug_metric.sort_values('SCC', ascending=False)

    drug_summary = pd.DataFrame(index=per_drug_metric.columns, columns=['mean', 'stdev'])
    drug_summary['mean'] = per_drug_metric.mean()
    drug_summary['stdev'] = per_drug_metric.std()
    print(drug_summary)

    per_drug_metric.to_excel(exwrite, sheet_name='Drug')
    drug_summary.to_excel(exwrite, sheet_name='Summary Drug')
    exwrite.save()
    print("Results written to: %s"%outfile)
    return scores
        
    
def run(gParameters):
    print("In Run Function:\n")
    args = candle.ArgumentStruct(**gParameters)
    # Call launch() with specific model arch and args with all HPs
    scores = launch(args)
    print('printing scores ...')
    print(scores)
    # Supervisor HPO
    with open(args.output_dir + "/scores_infer.json", "w", encoding="utf-8") as f:
        json.dump(scores.to_json(), f, ensure_ascii=False, indent=4)
    return scores

#def run_cross_study(gParameters, cell_lines, data_input_tuple, data_label):
#    load_model(gParameters, cell_lines, data_input_tuple, data_label)
#    scores = launch(args, test_data)
    

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

def candle_main(ANL=True):
    params = initialize_parameters()
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    data_dir = os.environ['CANDLE_DATA_DIR'] + "/Data/"
    cross_study_dir = data_dir + "/" + params['cross_study_dir']
    data_type_list = params['data_type'].split(',')
    gene_expression = data_dir + "/drp-data/grl-preprocessed/sanger_tcga/BiG_DRP_fpkm.csv"
#    label_matrix = data_dir + "/drp-data/grl-preprocessed//drug_response/BiG_DRP_data_cleaned.csv"
    if ANL:
        for dt in data_type_list:
            print(dt)
            test_data_tup = cross_study_dir + "/" + dt + "_tuples_test.csv"
            test_data_cleaned = cross_study_dir + "/" + dt + "_cleaned_test.csv"
            assert(os.path.isfile(test_data_tup))
            cross_study_scores = run_cross_study(params, dt, gene_expression, test_data_cleaned, test_data_tup)
    else:
#        run_infer(study, study_label, study_matrix)
        print("Done inference.")

    
if __name__ == "__main__":
    candle_main()
