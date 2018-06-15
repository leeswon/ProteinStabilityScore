# Cautions
1. Put data files (.csv) in Data directory
2. Use TensorFlow and Python3, but it is guessed to work on Python2 as well.

# Run
Example scripts to train a model are written in shell script _train.sh.
## Input Arguments
    - use_gpu, gpu_device : GPU based training
    - model_type : determines the type of a neural net model. STL (separate models for libraries), SNN (single neural net for all libraries), HPS and TF (multi-task model)
    - test_type : determines scale of regularization loss
    - save_mat_name : the name of MATLAB file for training result

# Files
- main.py : contains input argument of program and a neural net's hyper-parameter
- train.py : contains code of training neural net models as well as evaluating the models
- train_wrapper.py : enables to run several independent training procedures to average each model's performance (make result robust to random split of training/validation data)
- read_data.py : process data files for training
- utils_*.py : supplementary functions to construct neural net models


