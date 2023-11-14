# BiG-DRP: Bipartite Graph-based Drug Response Predictor


# IMPROVE PROJECT INSTRUCTIONS

The improve project requires standarized interfaces for data preprocessing, training and inference


# Requirements

**conda>=23.5**

# IMPROVE PROJECT INSTRUCTIONS

The improve project [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE)requires standarized interfaces for data preprocessing, training and inference, follow the code for BiG-DRP in [_BiG-DRP_](git@github.com:JDACS4C-IMPROVE/BiG-DRP.git)

#Installation

The IMPROVE project is currently using the develop branch


## Using Conda

**Create environment**

```
conda env create -f Big-DRP_conda.yml
```

**Activate the environment**

```
conda activate python_BigDRP
```

**Download BiG-DRP**

```
git clone -b develop git@github.com:JDACS4C-IMPROVE/BiG-DRP.git
cd BiG-DRP
```

**Install Torch for CUDA. dgl and CANDLE package**

```
python3 -m pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
python3 -m pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
python3 -m pip install git+https://github.com/ECP-CANDLE/candle_lib@develop

```

**Example usuage without container (running BiG-DRP)***

**Preprocess (optional)**

```
bash preprocess.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```

**Training**
```
bash train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR

```

**Testing**
```
bash infer.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```


## Using pip [RECOMMENDED]

```
pip install --upgrade pip
python3 -m pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 torchmetrics==0.11.1 --extra-index-url https://download.pytorch.org/whl/cu102
python3 -m pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
python3 -m pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
python3 -m pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
cd BiG-DRP
python3 -m pip install -r requirements.txt
chmod a+x *.sh
chmod a+x *.py
sh train.sh 1 data
```

# Example usage with container

Model definition file 'BiG_DRP.def' is located in [_here_](https://github.com/JDACS4C-IMPROVE/Singularity/tree/develop/definitions)

```
git clone -b develop https://github.com/JDACS4C-IMPROVE/Singularity.git
cd Singularity
```

Build Singularity

```
singularity build --fakeroot BiG_DRP.def definitions/BiG_DRP.def
```

Execute with container

```
singularity exec --nv BiG_DRP.sif train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```



# DATA PREPROCESSING

To create the data run the preprocess.sh code to download the data. To use a custom dataset, set the 'improve_analysis" flag to 'yes' in the BiG_DRP_model.txt file

# Model Training

1. train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR 

CANDLE_DATA_DIR=<PATH OF REPO/Data/>

Note: The train.sh script will download the original authors data if the Data directory is empty

      * set CUDA_VISIBLE_DEVICES to a GPU device ID to make this devices visible to the application.
      * CANDLE_DATA_DIR, path to base CANDLE directory for model input and outputs.
      * CANDLE_CONFIG , path to CANDLE config file must be inside CANDLE_DATA_DIR.

## Example

   * git clone ....
   * cd BiG-DRP
   * check permissions if all scripts are executable
   * ./preprocess.sh 2 $CANDLE_DATA_DIR
   * ./train.sh 2 $CANDLE_DATA_DIR
   * ./infer.sh 2 $CANDLE_DATA_DIR


## REQUIREMENTS

   * pandas==1.1.2
   * six==1.15.0
   * scipy==1.6.2
   * tqdm==4.60.0
   * nose==1.3.7
   * numpy==1.18.5
   * scikit_learn==0.24.2
   * json-encoder==0.4.4
   * kiwisolver==1.4.5