#!/bin/bash

mprof run --include-children --output "results/cora/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name cora --learning_rate 0.01 --batch_size 64 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/cora/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name cora --learning_rate 0.01 --batch_size 64 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/cora/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name cora --learning_rate 0.01 --batch_size 64 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/pubmed/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name pubmed --learning_rate 0.01 --batch_size 128 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/pubmed/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name pubmed --learning_rate 0.01 --batch_size 128 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/pubmed/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name pubmed --learning_rate 0.01 --batch_size 128 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/reddit/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name reddit --learning_rate 0.001 --batch_size 128 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/reddit/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name reddit --learning_rate 0.001 --batch_size 128 --epochs 200 --early_stopping 40

mprof run --include-children --output "results/reddit/mprofile_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset_name reddit --learning_rate 0.001 --batch_size 128 --epochs 200 --early_stopping 40

