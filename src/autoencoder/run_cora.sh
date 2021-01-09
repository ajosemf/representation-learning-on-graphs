#!/bin/bash

python -W ignore train.py --dataset_name cora --learning_rate 0.001 0.005 0.01 0.05 --batch_size 32 64 128 --epochs 200 --early_stopping 40
