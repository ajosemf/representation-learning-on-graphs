#!/bin/bash

python -W ignore train.py --dataset_name pubmed --learning_rate 0.01 --batch_size 32 --epochs 50 --early_stopping 5
