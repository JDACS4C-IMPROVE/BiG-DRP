#


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


# BiG-DRP Data Preprocessing

Here are the steps to process the GDSC data used by BiG-DRP

## Step 1: Download and format the data

1. Download the Drug Response data here: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
	- TableS4A.xlsx
	- TableS5C.xlsx

2. Copy the TableS4A and remove all formatting and merged columns. Save into a csv file "ln_ic50.csv". 
	This could be done easily by copying A6:JG996 into a fresh spreadsheed then saving as csv.
	The following should be the headers:
	
|  cosmic_id| sample_names  | drug1 | ... | drugN  |
|---|---|---|---|---|
|  |   |   |   |   |
|  |   |   |   |   |

3. Copy the TableS5C and remove all formatting and merged columns. Save into a csv file "binary_response.csv".  This could be done easily by copying B6:JG1008 into a fresh spreadsheed then saving as csv. The following should be the headers (and first row):

|  compounds| drug1  | drug2 | ... | drugN  |
|---|---|---|---|---|
| threshold |  |  |  |  |
| CCL1 |   |   |   |   |
| CCL2 |   |   |   |   |

5. Download the RNASeq gene expression (in FPKM) and genefrom Cell Model Passports: https://cellmodelpassports.sanger.ac.uk/downloads


## Step 2: Process the gene expression
Open the gdsc_sanger_preprocessing.ipynb, replace the filenames as needed, then run everything


## Step 3: Process the labels
Set the appropriate filenames in `label_preprocessing.py`, then run on terminal:

```
python label_preprocessing.py
```


## Step 4: Get the drug features

1. If you do not have the SMILES, run the following:

```
python graph_scripts/get_drug_smiles.py
```

2. (1) might not be enough. Manually, add some of the SMILES. The SMILES we used are already in the supplementary folder.

3. To get the morgan fingerprints:

```
python get_drug_morgan_fingerprint.py
```

4. To get the drug descriptors:

```
python get_drug_descriptors.py
```


## Step 5: Split the data

run:

```
python data_splits.py
```
