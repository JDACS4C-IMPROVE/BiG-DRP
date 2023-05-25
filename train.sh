#!/bin/bash
  
# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###
CANDLE_MODEL=/homes/ac.rgnanaolivu/improve_data_dir/BiG-DRP/train.py

if [ $# -lt 2 ]; then
    echo "Illegal number of parameters"
    echo "CUDA_VISIBLE_DEVICES and CANDLE_DATA_DIR are required"
    exit
fi

if [ $# -eq 2 ] ; then
    CUDA_VISIBLE_DEVICES=$1 ; shift
    CANDLE_DATA_DIR=$1 ; shift
    CMD="python ${CANDLE_MODEL}"
    echo "CMD = $CMD"

elif [ $# -ge 3 ] ; then
    CUDA_VISIBLE_DEVICES=$1 ; shift
    CANDLE_DATA_DIR=$1 ; shift
    
    # if original $3 is a file, set candle_config and passthrough $@
    if [ -f $CANDLE_DATA_DIR/$1 ] ; then
	echo "$CANDLE_DATA_DIR/$1 is a file"
        CANDLE_CONFIG=$1 ; shift
        CMD="python ${CANDLE_MODEL} --config_file $CANDLE_CONFIG $@"
        echo "CMD = $CMD $@"

     # else passthrough $@
    else
	echo "$1 is not a file"
        CMD="python ${CANDLE_MODEL} $@"
        echo "CMD = $CMD"
	
    fi
fi

# Display runtime arguments
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"

# Set up environmental variables and execute model
echo "activating environment"
. /homes/ac.rgnanaolivu/miniconda3/etc/profile.d/conda.sh
conda activate python_BigDRP
export TF_CPP_MIN_LOG_LEVEL=3

# Set up environmental variables and execute model
echo "running command ${CMD}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
