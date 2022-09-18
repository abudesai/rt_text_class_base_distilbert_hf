#!/bin/sh

image=$1

mkdir -p ml_vol/inputs/data_config
mkdir -p ml_vol/inputs/data/training/regressionBaseMainInput

mkdir -p ml_vol/model/model_config
mkdir -p ml_vol/model/artifacts

mkdir -p ml_vol/hpt/results

mkdir -p ml_vol/outputs

chmod 777 ml_vol/model

cp examples/abalone_schema.json ml_vol/inputs/data_config
cp examples/abalone_train.csv ml_vol/inputs/data/training/regressionBaseMainInput
cp examples/hyperparameters.json ml_vol/model/model_config

# docker run --gpus all -v $(pwd)/ml_vol:/opt/ml_vol --rm ${image} hpo -n 5
docker run -v $(pwd)/ml_vol:/opt/ml_vol --rm ${image} tune -n 5
