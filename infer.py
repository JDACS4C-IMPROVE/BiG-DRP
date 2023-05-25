import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
import train as bmk
import candle

# Just because the tensorflow warnings are a bit verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_path = os.path.dirname(os.path.realpath(__file__))


# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/'



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

    data_dir = os.environ['CANDLE_DATA_DIR'] + "/BiG-DRP/Improve/Data/"
    drug_response_dir = data_dir + "/drp-data/grl-preprocessed/drug_response/"
    drugset = data_dir + "/" + args.drug_list
    # load labels
    labels = drug_response_dir + "/" + args.tuples_label_fold_out
    y_tup = pd.read_csv(labels, index_col=0)
    if args.split == 'lpo':
        y_tup['fold'] = y_tup['pair_fold']
    else:
        y_tup['fold'] = y_tup['cl_fold']

    y_tup = y_tup.loc[y_tup['fold']>=0]
    y = y_tup.pivot(index='cell_line', columns='drug', values='ln_ic50')
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

    s = np.zeros((5, 2))
    for i in range(5):
        m  = ((fold_mask == i)*1).values.nonzero()
        s[i, 0] = spearmanr(y_norm.values[m], preds_norm.values[m])[0]
        s[i, 1] = spearmanr(y.values[m], preds_unnorm.values[m])[0]
    overall.loc[mets] = s
    overall.loc['spearman (fold.mean)'] = s.mean(axis=0)
    overall.loc['spearman (fold.stdev)'] = s.std(axis=0)
    print(overall)

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

def run(gParameters):
    print("In Run Function:\n")
    args = candle.ArgumentStruct(**gParameters)
    # Call launch() with specific model arch and args with all HPs
    print(args)
    scores = launch(args)

    # Supervisor HPO
    with open(args.output_dir + "/scores_infer.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores

def initialize_parameters():
    preprocessor_bmk = bmk.BiG_drp_candle(file_path,
        'BiG_DRP_model.txt',
        'pytorch',
        prog='BiG_drp_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

    
def main():
    gParameters = initialize_parameters()
    print(gParameters)
    scores = run(gParameters)
    print("Done inference.")

if __name__ == "__main__":
    main()
