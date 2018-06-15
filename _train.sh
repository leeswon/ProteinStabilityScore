#!/usr/bin/env bash

### Single-task
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type STL --save_mat_name result_separate_noreg_min.mat --test_type -1 >> tmp.out
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type STL --save_mat_name result_separate_reg_min.mat --test_type 1 >> tmp.out


### Single NN for MTL
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type SNN --save_mat_name result_single_noreg_min.mat --test_type -1 >> tmp.out
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type SNN --save_mat_name result_single_reg_min.mat --test_type 1 >> tmp.out


### Hard-Param Shared NN for MTL
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type HPS --save_mat_name result_hps_noreg_min.mat --test_type -1 >> tmp.out
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type HPS --save_mat_name result_hps_reg_min.mat --test_type 1 >> tmp.out


### Tensor Factored NN for MTL
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type TF --save_mat_name result_tf_noreg_min.mat --test_type -1 >> tmp.out
nohup python3 -u main.py --use_gpu --gpu_device 0 --model_type TF --save_mat_name result_tf_reg_min.mat --test_type 1 >> tmp.out
