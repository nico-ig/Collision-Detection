#!/bin/bash

RUN_BASE_DIR=$1

if [ -z ${RUN_BASE_DIR} ]; then
    echo "Usage: $0 <run_base_dir>"
    exit 1
fi

for n_clusters in  3 2 5; do
    RUN_DIR=${RUN_BASE_DIR}/n_clusters_${n_clusters}
    mkdir -p ${RUN_DIR}
    echo "Treinando modelos com ${n_clusters} clusters"
    python -u scripts/train.py -i dataset/train.csv -o ${RUN_DIR}/models -n ${n_clusters} | tee ${RUN_DIR}/train.log

    echo "Validando modelos"
    python -u scripts/validate.py -i dataset/val.csv -o ${RUN_DIR}/validate -m ${RUN_DIR}/models | tee ${RUN_DIR}/validate.log

    echo "Testando modelos"
    python -u scripts/validate.py -i dataset/test.csv -o ${RUN_DIR}/test -m ${RUN_DIR}/models | tee ${RUN_DIR}/test.log
done

echo "Concluído"
