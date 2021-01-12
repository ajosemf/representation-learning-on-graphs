# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

mprof run --include-children --output "results/cora/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset cora --data_prefix ../datasets --nomultilabel --num_layers 3 --bsize 3 --hidden1 128 --dropout 0.2 --weight_decay 0 --early_stopping 40 --epochs 200 --save_name coramodel --diag_lambda 0.0001 --learning_rate 0.001 --num_clusters 15 --num_clusters_val 2 --num_clusters_test 1 --model_idx 0

mprof run --include-children --output "results/cora/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset cora --data_prefix ../datasets --nomultilabel --num_layers 3 --bsize 3 --hidden1 128 --dropout 0.2 --weight_decay 0 --early_stopping 40 --epochs 200 --save_name coramodel --diag_lambda 0.0001 --learning_rate 0.001 --num_clusters 15 --num_clusters_val 2 --num_clusters_test 1 --model_idx 0

mprof run --include-children --output "results/cora/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset cora --data_prefix ../datasets --nomultilabel --num_layers 3 --bsize 3 --hidden1 128 --dropout 0.2 --weight_decay 0 --early_stopping 40 --epochs 200 --save_name coramodel --diag_lambda 0.0001 --learning_rate 0.001 --num_clusters 15 --num_clusters_val 2 --num_clusters_test 1 --model_idx 0

mprof run --include-children --output "results/pubmed/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset pubmed --data_prefix ../datasets --nomultilabel --num_layers 4 --bsize 3 --hidden1 128 --dropout 0.2 --weight_decay 0 --early_stopping 40 --epochs 200 --save_name pubmedmodel --diag_lambda 0.0001 --learning_rate 0.05 --num_clusters 80 --num_clusters_val 2 --num_clusters_test 1 --model_idx 0

mprof run --include-children --output "results/pubmed/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset pubmed --data_prefix ../datasets --nomultilabel --num_layers 4 --bsize 3 --hidden1 128 --dropout 0.2 --weight_decay 0 --early_stopping 40 --epochs 200 --save_name pubmedmodel --diag_lambda 0.0001 --learning_rate 0.05 --num_clusters 80 --num_clusters_val 2 --num_clusters_test 1 --model_idx 0

mprof run --include-children --output "results/pubmed/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset pubmed --data_prefix ../datasets --nomultilabel --num_layers 4 --bsize 3 --hidden1 128 --dropout 0.2 --weight_decay 0 --early_stopping 40 --epochs 200 --save_name pubmedmodel --diag_lambda 0.0001 --learning_rate 0.05 --num_clusters 80 --num_clusters_val 2 --num_clusters_test 1 --model_idx 0

mprof run --include-children --output "results/reddit/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset reddit --data_prefix ../datasets --nomultilabel --num_layers 4 --num_clusters 1500 --bsize 20 --hidden1 128 --dropout 0.2 --weight_decay 0  --early_stopping 40 --num_clusters_val 20 --num_clusters_test 1 --epochs 200 --save_name redditmodel --learning_rate 0.01 --diag_lambda 0.0001 --model_idx 0

mprof run --include-children --output "results/reddit/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset reddit --data_prefix ../datasets --nomultilabel --num_layers 4 --num_clusters 1500 --bsize 20 --hidden1 128 --dropout 0.2 --weight_decay 0  --early_stopping 40 --num_clusters_val 20 --num_clusters_test 1 --epochs 200 --save_name redditmodel --learning_rate 0.01 --diag_lambda 0.0001 --model_idx 0

mprof run --include-children --output "results/reddit/mprofile_0_$(date +%Y%m%d%H%M%S).dat" python -W ignore train.py --dataset reddit --data_prefix ../datasets --nomultilabel --num_layers 4 --num_clusters 1500 --bsize 20 --hidden1 128 --dropout 0.2 --weight_decay 0  --early_stopping 40 --num_clusters_val 20 --num_clusters_test 1 --epochs 200 --save_name redditmodel --learning_rate 0.01 --diag_lambda 0.0001 --model_idx 0

