from os import listdir, getcwd, mkdir, remove
from os.path import isfile
from math import ceil, sqrt

import scipy.io as spio
import numpy as np
from tensorflow import reset_default_graph

from read_data import read_data, read_data_15k
from train import train
from train_rand_forest import train_rand_forest

############################################
#### functions to load training results ####
############################################
def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

#### checks if entries in dictionary are mat-objects
def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], np.ndarray):
            if isinstance(dict[key][0], spio.matlab.mio5_params.mat_struct):
                #### cell of structure case
                tmp = np.zeros((len(dict[key]),), dtype=np.object)
                for cnt in range(len(dict[key])):
                    tmp[cnt] = _todict(dict[key][cnt])
                dict[key] = tmp
            else:
                #### just array case
                dict[key] = list(dict[key])
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

#### recursive function constructing nested dictionaries from matobjects
def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem, unicode):
            dict[strg] = str(elem)
        elif isinstance(elem, np.ndarray):
            if isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
                #### cell of structure case
                tmp = np.zeros((len(elem),), dtype=np.object)
                for cnt in range(len(elem)):
                    tmp[cnt] = _todict(elem[cnt])
                dict[strg] = tmp
            else:
                #### just array case
                dict[strg] = list(elem)
        elif strg == 'hidden_layer':
            dict[strg] = [elem]
        else:
            dict[strg] = elem
    return dict

#### compare two model_info_summary
#### cannot check whether two models are equal because floating number in Python and MATLABare different
def check_model_equivalency(model_info1, model_info2):
    check_keys1 = [(model_info1_elem in model_info2) for model_info1_elem in model_info1]
    check_keys2 = [(model_info2_elem in model_info1) for model_info2_elem in model_info2]
    if all(check_keys1) and all(check_keys2):
        return all([(model_info1[key] == model_info2[key]) for key in model_info1])
    else:
        return False

def print_model_info(model_dict):
    key_list = model_dict.keys()
    key_list.sort()
    for key in key_list:
        print(key + " : ", model_dict[key])

############################################
#### functions to save training results ####
############################################
def mean_of_list(list_input):
    return float(sum(list_input))/len(list_input)

def stddev_of_list(list_input):
    list_mean = mean_of_list(list_input)
    sq_err = [(x-list_mean)**2 for x in list_input]

    if len(list_input)<2:
        return 0.0
    else:
        return sqrt(sum(sq_err)/float(len(list_input)-1))


def model_info_summary(model_architecture, model_hyperpara, train_hyperpara, num_para_in_model=-1):
    tmp_dict = {}
    tmp_dict['architecture'] = model_architecture
    tmp_dict['learning_rate'] = train_hyperpara['lr']
    tmp_dict['improvement_threshold'] = train_hyperpara['improvement_threshold']
    tmp_dict['early_stopping_para'] = [train_hyperpara['patience'], train_hyperpara['patience_multiplier']]

    for model_hyperpara_elem in model_hyperpara:
        tmp_dict[model_hyperpara_elem] = model_hyperpara[model_hyperpara_elem]

    if num_para_in_model > -1:
        tmp_dict['number_of_trainable_parameters'] = num_para_in_model
    return tmp_dict


def reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, data_group_list, num_para_in_model=-1, doLifelong=False):
    result_of_curr_run = {}
    #### 'model_specific_info' element
    result_of_curr_run['model_specific_info'] = model_info_summary(model_architecture, model_hyperpara, train_hyperpara, num_para_in_model)

    num_run_per_model, best_valid_error_list, best_test_error_list, train_time_list = len(result_from_train_run), [], [], []
    result_of_curr_run['result_of_each_run'] = np.zeros((num_run_per_model,), dtype=np.object)
    if doLifelong:
        best_test_error_each_task_list, test_error_at_last_epoch_list = [], []
        best_test_error_each_task_avg_list, test_error_at_last_epoch_avg_list = [], []

    for cnt in range(num_run_per_model):
        result_of_curr_run['result_of_each_run'][cnt] = result_from_train_run[cnt]
        best_valid_error_list.append(result_from_train_run[cnt]['best_validation_error'])
        best_test_error_list.append(result_from_train_run[cnt]['test_error_at_best_epoch'])
        train_time_list.append(result_from_train_run[cnt]['training_time'])
        if doLifelong:
            chk_epoch = [x-1 for x in result_from_train_run[cnt]['task_changed_epoch'][1:]]+[result_from_train_run[cnt]['num_epoch']]
            best_test_error_each_task_list.append([result_from_train_run[cnt]['history_best_test_error'][x] for x in chk_epoch])
            best_test_error_each_task_avg_list.append(mean_of_list(best_test_error_each_task_list[cnt]))

            test_error_at_last_epoch_list.append(result_from_train_run[cnt]['history_test_error'][-1][0:-1])
            test_error_at_last_epoch_avg_list.append(mean_of_list(test_error_at_last_epoch_list[cnt]))

    result_of_curr_run['best_valid_error'] = best_valid_error_list
    result_of_curr_run['best_valid_error_mean'] = mean_of_list(best_valid_error_list)
    result_of_curr_run['best_valid_error_stddev'] = stddev_of_list(best_valid_error_list)
    result_of_curr_run['best_test_error'] = best_test_error_list
    result_of_curr_run['best_test_error_mean'] = mean_of_list(best_test_error_list)
    result_of_curr_run['best_test_error_stddev'] = stddev_of_list(best_test_error_list)
    result_of_curr_run['training_time'] = train_time_list
    result_of_curr_run['training_time_mean'] = mean_of_list(train_time_list)
    result_of_curr_run['training_time_stddev'] = stddev_of_list(train_time_list)
    result_of_curr_run['train_valid_data_group'] = data_group_list
    if doLifelong:
        result_of_curr_run['best_test_error_each_task'] = best_test_error_each_task_list
        result_of_curr_run['best_test_error_each_task_mean'] = mean_of_list(best_test_error_each_task_avg_list)
        result_of_curr_run['best_test_error_each_task_std'] = stddev_of_list(best_test_error_each_task_avg_list)
        result_of_curr_run['test_error_at_last_epoch'] = test_error_at_last_epoch_list
        result_of_curr_run['test_error_at_last_epoch_mean'] = mean_of_list(test_error_at_last_epoch_avg_list)
        result_of_curr_run['test_error_at_last_epoch_std'] = stddev_of_list(test_error_at_last_epoch_avg_list)

    return result_of_curr_run


#### functions for Random Forest model
def model_info_summary_rndforest(model_type, model_hyperpara):
    tmp_dict = {}
    tmp_dict['type'] = model_type
    tmp_dict['num_estimators'] = model_hyperpara["num_estimators"]
    tmp_dict['split_criterion'] = model_hyperpara["split_crit"]

    return tmp_dict


def reformat_result_for_mat_rndforest(model_type, model_hyperpara, result_from_train_run):
    result_of_curr_run = {}
    #### 'model_specific_info' element
    result_of_curr_run['model_specific_info'] = model_info_summary_rndforest(model_type, model_hyperpara)

    num_run_per_model, train_error_list, valid_error_list, test_error_list, train_time_list = len(result_from_train_run), [], [], [], []
    result_of_curr_run['result_of_each_run'] = np.zeros((num_run_per_model,), dtype=np.object)

    for cnt in range(num_run_per_model):
        result_of_curr_run['result_of_each_run'][cnt] = result_from_train_run[cnt]
        train_error_list.append(result_from_train_run[cnt]['train_error'][-1])
        valid_error_list.append(result_from_train_run[cnt]['validation_error'][-1])
        test_error_list.append(result_from_train_run[cnt]['test_error'][-1])
        train_time_list.append(result_from_train_run[cnt]['training_time'])

    result_of_curr_run['train_error'] = train_error_list
    result_of_curr_run['train_error_mean'] = mean_of_list(train_error_list)
    result_of_curr_run['train_error_stddev'] = stddev_of_list(train_error_list)
    result_of_curr_run['valid_error'] = valid_error_list
    result_of_curr_run['valid_error_mean'] = mean_of_list(valid_error_list)
    result_of_curr_run['valid_error_stddev'] = stddev_of_list(valid_error_list)
    result_of_curr_run['test_error'] = test_error_list
    result_of_curr_run['test_error_mean'] = mean_of_list(test_error_list)
    result_of_curr_run['test_error_stddev'] = stddev_of_list(test_error_list)
    result_of_curr_run['training_time'] = train_time_list
    result_of_curr_run['training_time_mean'] = mean_of_list(train_time_list)
    result_of_curr_run['training_time_stddev'] = stddev_of_list(train_time_list)

    return result_of_curr_run


############################################
#### functions to run several training  ####
####       with same model setting      ####
############################################
def train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, data_hyperpara, mat_file_name, saved_result=None, useGPU=False, GPU_device=0, doLifelong=False):
    if not 'Result' in listdir(getcwd()):
        mkdir('Result')

    max_run_cnt = train_hyperpara['num_run_per_model']

    #### process results of previous training
    if (saved_result is None) and not (isfile('./Result/'+mat_file_name)):
        saved_result = np.zeros((1,), dtype=np.object)
    elif (saved_result is None):
        saved_result_tmp = spio.loadmat('./Result/'+mat_file_name)
        num_prev_test_model = len(saved_result_tmp['training_summary'][0])
        saved_result = np.zeros((num_prev_test_model+1,), dtype=np.object)
        for cnt in range(num_prev_test_model):
            saved_result[cnt] = saved_result_tmp['training_summary'][0][cnt]
    else:
        num_prev_result, prev_result_tmp = len(saved_result), saved_result
        saved_result = np.zeros((num_prev_result+1,), dtype=np.object)
        for cnt in range(num_prev_result):
            saved_result[cnt] = prev_result_tmp[cnt]

    #### process results saved in temporary file
    if not (isfile('./Result/temp_'+mat_file_name)):
        temp_result_exist = False
    else:
        saved_temp_result_tmp = loadmat('./Result/temp_'+mat_file_name)
        saved_temp_result = saved_temp_result_tmp['training_summary_temp']
        print('\nModel in temp file')
        print_model_info(saved_temp_result['model_specific_info'])
        print('\nModel for current train')
        print_model_info(model_info_summary(model_architecture, model_hyperpara, train_hyperpara))
        #use_temp_result = input('\nUse temp result? (T/F): ')
        use_temp_result = 't'
        if (use_temp_result is 'T') or (use_temp_result is 't') or (use_temp_result is 'True') or (use_temp_result is 'true'):
            temp_result_exist = True
        else:
            temp_result_exist = False

    result_from_train_run = []
    if temp_result_exist:
        if not (isinstance(saved_temp_result['train_valid_data_group'], np.ndarray) or isinstance(saved_temp_result['train_valid_data_group'], list)):
            result_from_train_run.append(saved_temp_result['result_of_each_run'])
        else:
            for elem in saved_temp_result['result_of_each_run']:
                result_from_train_run.append(elem)
        run_cnt_init = len(result_from_train_run)
        print("\nTemporary Result Exists! Start from %d\n" %(run_cnt_init+1))
    else:
        run_cnt_init = 0
        print("\nTemporary Result is Discarded! Overwrite Temp File\n")

    #### run training procedure with different dataset
    for run_cnt in range(run_cnt_init, max_run_cnt):
        # dataset, datainfo = read_data_15k(data_hyperpara['folder_name'], data_hyperpara['train_file_name'], data_hyperpara['test_file_name'], data_hyperpara['train_valid_test_ratio'], data_hyperpara['all_output'])
        dataset, datainfo = read_data(data_hyperpara['folder_name'], data_hyperpara['train_file_name'], data_hyperpara['test_file_name'], data_hyperpara['train_valid_ratio'], data_hyperpara['all_output'])
        print("Training/Validation data group : %d\n" %(run_cnt))
        if 'rndforest' in model_architecture:
            train_result_tmp, num_model_para = train_rand_forest(model_architecture, model_hyperpara, dataset, datainfo)
        else:
            train_result_tmp, num_model_para = train(model_architecture, model_hyperpara, train_hyperpara, dataset, datainfo, useGPU, GPU_device, doLifelong)
        result_from_train_run.append(train_result_tmp)

        if run_cnt < max_run_cnt-1:
            if 'rndforest' in model_architecture:
                result_of_curr_run = reformat_result_for_mat_rndforest(model_architecture, model_hyperpara, result_from_train_run)
            else:
                result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, range(run_cnt+1), num_model_para, doLifelong)
            spio.savemat('./Result/temp_'+mat_file_name, {'training_summary_temp':result_of_curr_run})
        elif isfile('./Result/temp_'+mat_file_name):
            remove('./Result/temp_'+mat_file_name)

        print("%d-th training run\n\n" % (run_cnt+1))
        reset_default_graph()

    #### save training summary
    if 'rndforest' in model_architecture:
        result_of_curr_run = reformat_result_for_mat_rndforest(model_architecture, model_hyperpara, result_from_train_run)
    else:
        result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, range(max_run_cnt), num_model_para, doLifelong)
    saved_result[-1] = result_of_curr_run
    spio.savemat('./Result/'+mat_file_name, {'training_summary':saved_result})

    return saved_result

