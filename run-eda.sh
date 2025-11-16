#!/bin/bash

(python -u ./scripts/eda.py -f dataset/train.csv > eda-train.log 2>&1) &

(python -u ./scripts/eda.py -f dataset/val.csv > eda-val.log 2>&1) &

(python -u ./scripts/eda.py -f dataset/test.csv > eda-test.log 2>&1) &

(python -u ./scripts/eda.py -f dataset/full.csv > eda-full.log 2>&1) &

wait

echo "Processos finalizados"
