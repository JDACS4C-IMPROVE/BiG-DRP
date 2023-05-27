# BiG-DRP: Bipartite Graph-based Drug Response Predictor


# IMPROVE PROJECT INSTRUCTIONS

The improve project requires standarized interfaces for data preprocessing, training and inference

# DATA PREPROCESSING

To create the data run the preprocess.sh code to download the data. To use a custom dataset, set the 'improve_analysis" flag to 'yes' in the DrugCell_params.txt file

# Model Training

1. train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR 

CANDLE_DATA_DIR=<PATH OF REPO/Data/>

Note: The train.sh script will download the original authors data if the Data directory is empty

      * set CUDA_VISIBLE_DEVICES to a GPU device ID to make this devices visible to the application.
      * CANDLE_DATA_DIR, path to base CANDLE directory for model input and outputs.
      * CANDLE_CONFIG , path to CANDLE config file must be inside CANDLE_DATA_DIR.

## Example

   * git clone ....
   * cd DrugCell
   * mkdir Data
   * check permissions if all scripts are executable
   * ./preprocess.sh 2 ./Data
   * ./train.sh 2 ./Data
   * ./infer.sh 2 ./Data


## Setting up environment

### Install Conda version version 22.11.1

* step 1: conda create -n python_BigDRP python=3.8 anaconda
* step 2: conda activate python_BigDRP
* step 3: conda install pip
* step 4: pip install -r requirements.txt
* step 5: pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
* step 6: pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
* step 5: pip install git+https://github.com/ECP-CANDLE/candle_lib@0d32c6bb97ace0370074194943dbeaf9019e6503


## REQUIREMENTS

* pandas==1.1.2
* six==1.15.0
* scipy==1.6.2
* torch==1.6.0
* tqdm==4.60.0
* nose==1.3.7
* numpy==1.18.5
* scikit_learn==0.24.2
* json-encoder==0.4.4

