#!/bin/sh

image=$1

mkdir -p ml_vol/model/artifacts

# if [ -z "$(ls -A mldir/model/)" ]; then
#     tar xvzf examples/model.tgz -C ml_vol/model/artifacts
# fi

cp -a ./examples/artifacts/. ./ml_vol/model/artifacts/

docker run -v $(pwd)/ml_vol:/opt/ml_vol -p 8080:8080 --rm ${image} serve