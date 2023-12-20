from pathlib import Path
from utils.tuple_dataset import TupleMatrixDataset
from utils.utils import mkdir, reindex_tuples, moving_average, reset_seed, create_fold_mask
from utils.network_gen import create_network
from utils.data_initializer import initialize
from argparse import Namespace
import torch
from pathlib import Path
from pprint import pformat
from typing import Dict, Union
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from bigdrp.trainer import Trainer
from improve import framework as frm
from improve.metrics import compute_metrics
from candle import CandleCkptPyTorch
import numpy as np
import pandas as pd
import os
import candle
torch.cuda.empty_cache()
# Just because the tensorflow warnings are a bit verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_path = os.path.dirname(os.path.realpath(__file__))

filepath = Path(__file__).resolve().parent

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_train_params = []

# [GraphDRP] Model-specific params (Model: GraphDRP)
model_train_params = [
    {
        "name": "log_interval",
        "action": "store",
        "type": int,
        "help": "Interval for saving o/p"
    },
    {
        "name": "cuda_name",  # TODO. frm. How should we control this?
        "action": "store",
        "type": str,
        "help": "Cuda device (e.g.: cuda:0, cuda:1."
    },
    {
        "name": "learning_rate",
        "type": float,
        "default": 0.0001,
        "help": "Learning rate for the optimizer."
    },
    {
        "name": "batch_size",
        "type": int,
        "help": "...",
    },
    {
        "name": "epochs",
        "type": int,
        "help": "number of epochs to train on",
    },
    {
        "name": "network_percentile",
        "type": int,
        "help": "network percentile for metrics",
    },
]

req_preprocess_args = [ll["name"] for ll in model_train_params]

def config_checkpointing(params: Dict, model, optimizer):
    """Configure CANDLE checkpointing. Reads last saved state if checkpoints exist.

    :params str ckpt_directory: String with path to directory for storing the CANDLE checkpointing for the model being trained.

    :return: Number of training iterations already run (this may be > 0 if reading from checkpointing).
    :rtype: int
    """
    # params["ckpt_directory"] = ckpt_directory
    initial_epoch = 0
    # TODO. This creates directory self.params["ckpt_directory"]
    # import pdb; pdb.set_trace()
    ckpt = CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})
    J = ckpt.restart(model)
    if J is not None:
        initial_epoch = J["epoch"]
        print("restarting from ckpt: initial_epoch: %i" % initial_epoch)
    return ckpt, initial_epoch

            
def initialize_parameters():
    params = frm.initialize_parameters(
        filepath,
        default_model="BiG_DRP_model.txt",
        additional_definitions=model_train_params,
        required=req_preprocess_args,
    )
    return params


def fold_validation(hyperparams, seed, network, train_data, val_data,
                    cell_lines, drug_feats, tuning, epoch, maxout=False):
    reset_seed(seed)
    print('The batch size is {0}'.format(hyperparams['batch_size']))
    train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=hyperparams['batch_size'], shuffle=False)

    n_genes = cell_lines.shape[1]
    n_drug_feats = drug_feats.shape[1]
    print(n_drug_feats)
    trainer = Trainer(n_genes, cell_lines, drug_feats, network, hyperparams)
    print("number of epochs is {0}".format(epoch))
    val_error, metric_names = trainer.fit(
        num_epoch=epoch, 
        train_loader=train_loader, 
        val_loader=val_loader,
        tuning=tuning,
        maxout=maxout) #validation_step_cellwise_anl

    return val_error, trainer, metric_names

def create_dataset(tuples, train_x, val_x,
                   train_y, val_y, train_mask,
                   val_mask, drug_feats, percentile):

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


def nested_cross_validation(FLAGS, drug_feats, cell_lines, labels,
                            label_matrix, normalizer, learning_rate, epoch, batch_size):
    reset_seed(FLAGS.seed)
    print("number of epochs is {0}".format(epoch))
    print("learning rate is {0}".format(learning_rate))
    print("batch_size is {0}".format(batch_size))    
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
            train_mask, val_mask, drug_feats, FLAGS.network_percentile)

        val_error,_,_ = fold_validation(hp, FLAGS.seed, network, train_data, 
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
            train_mask, test_mask.values, drug_feats, FLAGS.network_percentile)

        test_error, trainer, metric_names = fold_validation(hp, FLAGS.seed, network, train_data, 
                                                            test_data, cl_tensor, df_tensor, tuning=False, 
                                                            epoch=hp['num_epoch'], maxout=True) # set maxout so that the trainer uses all epochs

        if i == 0:
            final_metrics = np.zeros((5, test_error.shape[1]))

        final_metrics[i] = test_error[-1]
        test_metrics = pd.DataFrame(test_error, columns=metric_names)
#        test_metrics.to_csv(
        test_metrics.to_csv(FLAGS.model_outdir + "/results/" + 'fold_%d.csv'%i, index=False)

        drug_enc = trainer.get_drug_encoding().cpu().detach().numpy()
        pd.DataFrame(drug_enc, index=drug_list).to_csv(FLAGS.model_outdir + "/results/" + 'encoding_fold_%d.csv'%i)

        trainer.save_anl_model(FLAGS.model_outdir + "/results/" + "model", i, hp)
#        test_x.to_csv("testData.csv", index=False)
        # save predictions
        test_data = TensorDataset(torch.FloatTensor(test_x))
        test_data = DataLoader(test_data, batch_size=hyperparams['batch_size'], shuffle=False)
        prediction_matrix = trainer.predict_matrix(test_data, drug_encoding=torch.Tensor(drug_enc))
        prediction_matrix = pd.DataFrame(prediction_matrix, index=test_samples, columns=drug_list)

        # remove predictions for non-test data
        test_mask = test_mask.replace(0, np.nan)
        prediction_matrix = prediction_matrix*test_mask

        prediction_matrix.to_csv(FLAGS.model_outdir + "/results/" + '/val_prediction_fold_%d.csv'%i)
    
    return final_metrics


def anl_test_data(FLAGS, drug_feats, cell_lines, labels,
                  label_matrix, normalizer, learning_rate, epoch, batch_size):
    reset_seed(FLAGS.seed)
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
                                                                             FLAGS.network_percentile)

    val_error,_,_ = fold_validation(hp, FLAGS.seed, network, 
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

    test_tuples = labels.loc[labels['cl_fold'] == 3]
    test_samples = list(test_tuples['cell_line'].unique())
    test_x = cell_lines.loc[test_samples].values
    test_y = label_matrix.loc[test_samples].values

    train_tuples = train_tuples[['drug', 'cell_line', 'response']]
    train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds

    train_x, test_x = normalizer(train_x, test_x)
    network, train_data, test_data, cl_tensor, df_tensor = create_dataset_anl(train_tuples, 
                                                                              train_x, test_x, 
                                                                              train_y, test_y, 
                                                                              drug_feats, FLAGS.network_percentile)

    test_error, trainer, metric_names = fold_validation(hp, FLAGS.seed, network, 
                                                        train_data, 
                                                        test_data, cl_tensor, df_tensor, tuning=False, 
                                                        epoch=hp['num_epoch'], maxout=True) 

    test_metrics = pd.DataFrame(test_error, columns=metric_names)
#    test_metrics.to_csv( + "/fold_%d.csv", index=False)
    test_metrics.to_csv(FLAGS.model_outdir + "/results/fold_%d.csv", index=False)

    drug_enc = trainer.get_drug_encoding().cpu().detach().numpy()
    pd.DataFrame(drug_enc, index=drug_list).to_csv(FLAGS.model_outdir + '/results/encoding_fold_%d.csv')
    outdir= FLAGS.model_outdir + "/results/"
    trainer.save_anl_model(outdir, hp)
    print("model built at {0}".format(outdir))
    test_data = TensorDataset(torch.FloatTensor(test_x))
    test_data = DataLoader(test_data, batch_size=hyperparams['batch_size'], shuffle=False)

    prediction_matrix = trainer.predict_matrix(test_data, drug_encoding=torch.Tensor(drug_enc))
    prediction = prediction_matrix
    true_label = test_y
    prediction_flat = prediction.flatten()
    label_flat = true_label.flatten()
    prediction_mean = np.mean(prediction)
    label_mean = np.mean(true_label)
    TSS = np.sum((true_label - label_mean) ** 2)
    RSS = np.sum((prediction - prediction_mean) ** 2)    
    rmse = np.sqrt(((prediction - true_label)**2).mean())
    scc, _ = spearmanr(label_flat, prediction_flat) 
#    scc, _ = spearmanr(true_label, prediction)
    pcc, _ = pearsonr(label_flat,prediction_flat)
    r2 = r2_score(label_flat,prediction_flat) 
    prediction_matrix = pd.DataFrame(prediction_matrix, index=test_samples, columns=drug_list)
    prediction_matrix.to_csv(FLAGS.model_outdir + "/results/val_prediction_fold_%d.csv")
#    final_metrics = pd.DataFrame([test_error[-1]], columns=metric_names)
#    final_metrics = pd.DataFrame(test_error[-1])
#    final_metrics = final_metrics.T
#    final_metrics.colums = metric_names
    final_metrics = [rmse, r2, pcc, scc]
#    print(final_metrics)
#    final_metrics = pd.DataFrame(final_metrics, columns=metric_names)
    final_metrics = pd.DataFrame(final_metrics)
    final_metrics = final_metrics.T    
    final_metrics.columns = ["RMSE", "R2", "PCC", "SCC"]
    return final_metrics


def main(drp_params, params):
    ns = Namespace(**drp_params)
    FLAGS = ns
    tuples_label_fold_out = params['train_ml_data_dir'] + "/" +  params['tuples_label_fold_out']
    expression_out = params['train_ml_data_dir'] + "/" +  params['expression_out']
    data_cleaned_out = params['train_ml_data_dir'] + "/" +  params['data_cleaned_out']
    descriptor_out = params['train_ml_data_dir'] + "/" +  params['descriptor_out']
    morgan_out = params['train_ml_data_dir'] + "/" +  params['morgan_out']    
    drug_feats, cell_lines, labels, label_matrix, normalizer = initialize(FLAGS,
                                                                          tuples_label_fold_out,
                                                                          expression_out,
                                                                          data_cleaned_out,
                                                                          descriptor_out,
                                                                          morgan_out)
    test_metrics = anl_test_data(FLAGS, drug_feats, cell_lines, labels,label_matrix, normalizer, 
                                 params['learning_rate'], params['epochs'], params['batch_size'])
#    test_metrics = nested_cross_validation(FLAGS, drug_feats, cell_lines, labels,
#                                           label_matrix, normalizer, learning_rate, epoch, batch_size)

 #   test_metrics = test_metrics.mean(axis=0)
    print(test_metrics)
    print("Overall Performance")
    print("RMSE: %f"%test_metrics[0])
#    print("RMSE: %f"%np.sqrt(test_metrics[0]))
    print("R2: %f"%test_metrics[1])
    print("Pearson: %f"%test_metrics[2])
    print("Spearman: %f"%test_metrics[3])
    print("Note: This is not the per-drug performance that is reported in the paper")
    print("To obtain per-drug performance, use metrics/calculate_metrics.py")
#    test_out = 'test_results.csv'
#    test_metrics.to_csv(test_out, index=False)



def candle_main():
    params = initialize_parameters()
    frm.create_outdir(outdir=params["model_outdir"])
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
#    params =  preprocess(params)
#    print(params)
    drp_params = dict((k, params[k]) for k in ('descriptor_out', 'expression_out','morgan_out','data_cleaned_out',
                                               'labels', 'tuples_label_fold_out',
                                               'dataroot', 'drug_feat',
                                               'model_outdir', 'mode', 'network_percentile',
                                               'normalize_response', 'seed',
                                               'split', 'weight_folder', 'epochs',
                                               'batch_size', 'learning_rate'))
    scores = main(drp_params, params)


if __name__ == "__main__":
    candle_main()
