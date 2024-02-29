#!/bin/bash
  
#########################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE #PATH# WITH THE MODEL EXECUTABLE.
#########################################################################


# arg 1 CANDLE_DATA_DIR
# arg 2 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###

CANDLE_MODEL_SCRIPT==BiGDRP_preprocess_improve.py

# Set env if CANDLE_MODEL is not in same directory as this script
IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}

# Combine path and name and check if executable exists
CANDLE_MODEL=${IMPROVE_MODEL_DIR}/${CANDLE_MODEL}
if [ ! -f ${CANDLE_MODEL} ] ; then
        echo No such file ${CANDLE_MODEL}
        exit 404
fi


if [ $# -lt 2 ] ; then
        echo "Illegal number of parameters"
        echo "CANDLE_DATA_DIR PARAMS are required"
        exit -1
fi

if [ $# -eq 2 ] ; then
        CANDLE_DATA_DIR=$1 ; shift
        
        # if $2 is a file, then set candle_config
        if [ -f $CANDLE_DATA_DIR/$1 ] ; then
                CONFIG_FILE=$1 ; shift
                CMD="python ${CANDLE_MODEL} --config_file ${CONFIG_FILE}"
        else
                CMD="python ${CANDLE_MODEL} $@"
                echo "CMD = $CMD"
        fi

elif [ $# -ge 3 ] ; then

        CANDLE_DATA_DIR=$1 ; shift

        # if $2 is a file, then set candle_config
        if [ -f $CANDLE_DATA_DIR/$1 ] ; then
                echo "$1 is a file"
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
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
=======
# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###

CANDLE_MODEL=preprocess_new.py

if [ $# -lt 2 ] ; then
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
echo "running command ${CMD}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD

