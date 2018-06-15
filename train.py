import os
import pickle
import timeit
from random import shuffle

import numpy as np
import tensorflow as tf

from ffnn_baseline_model import *

#### function to generate appropriate deep neural network
def model_generation(model_architecture, model_hyperpara, train_hyperpara, data_info):
    learning_model, gen_model_success = None, True
    learning_rate = train_hyperpara['lr']
    learning_rate_decay = train_hyperpara['lr_decay']

    if data_info is not None:
        x_dim, y_dim, num_task = data_info
    layers_dimension = [x_dim] + model_hyperpara['hidden_layer'] + [y_dim]

    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']

    ###### FFNN models
    if model_architecture == 'ffnn_minibatch':
        print("Training mini-batch FFNN model")
        learning_model = FFNN_minibatch(dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)
    elif model_architecture == 'mtl_several_ffnn_minibatch':
        print("Training several MTL-FFNN model (Single task ver.)")
        learning_model = MTL_several_FFNN_minibatch(num_tasks=num_task, dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, l1_reg_scale=model_hyperpara['regularization_scale'])
    elif model_architecture == 'mtl_ffnn_minibatch':
        print("Training MTL-FFNN model (Single NN ver.)")
        learning_model = MTL_FFNN_minibatch(num_tasks=num_task, dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, l1_reg_scale=model_hyperpara['regularization_scale'])
    elif model_architecture == 'mtl_ffnn_hard_para_sharing_minibatch':
        print("Training MTL-FFNN model (Hard Parameter Sharing Ver.)")
        ts_layer_dim_tmp = model_hyperpara['task_specific_layer']
        layers_dimension = [[x_dim]+model_hyperpara['hidden_layer'], [ts_layer_dim_tmp[x]+[y_dim] for x in range(num_task)]]
        learning_model = MTL_FFNN_HPS_minibatch(num_tasks=num_task, dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, l1_reg_scale=model_hyperpara['regularization_scale'])
    elif model_architecture == 'mtl_ffnn_tensor_factor_minibatch':
        print("Training MTL-FFNN model (Tensor Factorization)")
        ts_layer_dim_tmp = model_hyperpara['task_specific_layer']
        layers_dimension = [[x_dim]+model_hyperpara['hidden_layer'], [ts_layer_dim_tmp[x] + [y_dim] for x in range(num_task)]]
        factor_type = model_hyperpara['tensor_factor_type']
        factor_eps_or_k = model_hyperpara['tensor_factor_error_threshold']
        learning_model = MTL_FFNN_Tensor_Factor_minibatch(num_tasks=num_task, dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, l1_reg_scale=model_hyperpara['regularization_scale'], factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)
    else:
        print("No such model exists!!")
        print("No such model exists!!")
        print("No such model exists!!")
        gen_model_success = False
    return (learning_model, gen_model_success)


#### module of training/testing one model
def train(model_architecture, model_hyperpara, train_hyperpara, dataset, data_info, useGPU, GPU_device, doLifelong):
    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.85
        print("GPU %d is used" %(GPU_device))
    else:
        print("CPU is used")

    ### set-up data
    train_data, validation_data, test_data = dataset
    x_dim, y_dim, num_task = data_info
    num_train, num_valid, num_test = [x[0].shape[0] for x in train_data], [x[0].shape[0] for x in validation_data], [x[0].shape[0] for x in test_data]

    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']

    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, data_info)
    if not generation_success:
        return (None, None, None, None)

    learning_step = -1
    if (('batch_size' in locals()) or ('batch_size' in globals())) and (('num_task' in locals()) or ('num_task' in globals())):
        if num_task > 1:
            indices = [np.arange(num_train[x]//batch_size) for x in range(num_task)]
        else:
            indices = [np.arange(num_train//batch_size)]

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []
    train_total_error_hist, valid_total_error_hist, test_total_error_hist = [], [], []
    task_for_train, task_change_epoch = 0, [1]
    best_param = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #tfboard_writer = tf.summary.FileWriter('./graphs', sess.graph)

        start_time = timeit.default_timer()
        while learning_step < min(learning_step_max, epoch_bias + patience):
            learning_step = learning_step+1

            #### training & performance measuring process
            if model_architecture == 'ffnn_minibatch':
                #### mini-batch FFNN for single task
                shuffle(indices[0])
                if learning_step > 0:
                    for batch_cnt in range(num_train[0]//batch_size):
                        sess.run(learning_model.update, feed_dict={learning_model.model_input: train_data[0][0][indices[0][batch_cnt]*batch_size:(indices[0][batch_cnt]+1)*batch_size, :], learning_model.true_output: train_data[0][1][indices[0][batch_cnt]*batch_size:(indices[0][batch_cnt]+1)*batch_size], learning_model.epoch: learning_step})

                train_error_tmp = 0.0
                for batch_cnt in range(num_train[0]//batch_size):
                    train_error_tmp = train_error_tmp + sess.run(learning_model.train_loss, feed_dict={learning_model.model_input: train_data[0][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output: train_data[0][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})

                validation_error_tmp = 0.0
                for batch_cnt in range(num_valid[0]//batch_size):
                    validation_error_tmp = validation_error_tmp + sess.run(learning_model.valid_loss, feed_dict={learning_model.model_input: validation_data[0][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output: validation_data[0][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})

                test_error_tmp = 0.0
                for batch_cnt in range(num_test[0]//batch_size):
                    test_error_tmp = test_error_tmp + sess.run(learning_model.test_loss, feed_dict={learning_model.model_input: test_data[0][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output: test_data[0][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})

                train_error, valid_error, test_error = np.sqrt(train_error_tmp/num_train[0]), np.sqrt(validation_error_tmp/num_valid[0]), np.sqrt(test_error_tmp/num_test[0])

            elif (model_architecture == 'mtl_several_ffnn_minibatch') or (model_architecture == 'mtl_ffnn_minibatch') or (model_architecture == 'mtl_ffnn_hard_para_sharing_minibatch') or (model_architecture == 'mtl_ffnn_tensor_factor_minibatch'):
                #### Multi-task models
                if not doLifelong:
                    #task_for_train = np.random.randint(0, num_task)
                    if learning_step > 0:
                        for task_cnt in range(num_task):
                            shuffle(indices[task_cnt])
                            for batch_cnt in range(num_train[task_cnt] // batch_size):
                                sess.run(learning_model.update[task_cnt], feed_dict={learning_model.model_input[task_cnt]: train_data[task_cnt][0][indices[task_cnt][batch_cnt]*batch_size:(indices[task_cnt][batch_cnt]+1)*batch_size, :], learning_model.true_output[task_cnt]: train_data[task_cnt][1][indices[task_cnt][batch_cnt]*batch_size:(indices[task_cnt][batch_cnt]+1)*batch_size], learning_model.epoch: learning_step})

                elif learning_step > 0:
                    shuffle(indices[task_for_train])
                    for batch_cnt in range(num_train[task_for_train]//batch_size):
                        sess.run(learning_model.update[task_for_train], feed_dict={learning_model.model_input[task_for_train]: train_data[task_for_train][0][indices[task_for_train][batch_cnt]*batch_size:(indices[task_for_train][batch_cnt]+1)*batch_size, :], learning_model.true_output[task_for_train]: train_data[task_for_train][1][indices[task_for_train][batch_cnt]*batch_size:(indices[task_for_train][batch_cnt]+1)*batch_size], learning_model.epoch: learning_step})

                model_train_error, model_valid_error, model_test_error = learning_model.train_loss, learning_model.valid_loss, learning_model.test_loss
                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                train_total_error_tmp, validation_total_error_tmp, test_total_error_tmp = 0.0, 0.0, 0.0
                for task_cnt in range(num_task):
                    for batch_cnt in range(num_train[task_cnt]//batch_size):
                        train_error_tmp[task_cnt] = train_error_tmp[task_cnt] + sess.run(model_train_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: train_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: train_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})
                    #### compute error for residual train data
                    train_error_tmp[task_cnt] = train_error_tmp[task_cnt] + sess.run(model_train_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]:train_data[task_cnt][0][(batch_cnt+1)*batch_size:, :], learning_model.true_output[task_cnt]:train_data[task_cnt][1][(batch_cnt+1)*batch_size:]})
                    train_error_tmp[task_cnt] = train_error_tmp[task_cnt]/num_train[task_cnt]

                    for batch_cnt in range(num_valid[task_cnt]//batch_size):
                        validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt] + sess.run(model_valid_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: validation_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: validation_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})
                    #### compute error for residual validation data
                    validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt] + sess.run(model_valid_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: validation_data[task_cnt][0][(batch_cnt+1)*batch_size:, :], learning_model.true_output[task_cnt]: validation_data[task_cnt][1][(batch_cnt+1)*batch_size:]})
                    validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt]/num_valid[task_cnt]

                    for batch_cnt in range(num_test[task_cnt]//batch_size):
                        test_error_tmp[task_cnt] = test_error_tmp[task_cnt] + sess.run(model_test_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: test_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: test_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})
                    #### compute error for residual test data
                    test_error_tmp[task_cnt] = test_error_tmp[task_cnt] + sess.run(model_test_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: test_data[task_cnt][0][(batch_cnt+1)*batch_size:, :], learning_model.true_output[task_cnt]: test_data[task_cnt][1][(batch_cnt+1)*batch_size:]})
                    test_error_tmp[task_cnt] = test_error_tmp[task_cnt]/num_test[task_cnt]

                train_total_error_tmp = np.sum(np.array(train_error_tmp, dtype=np.float64)*np.array(num_train, dtype=np.float64))/np.sum(np.array(num_train, dtype=np.float64))
                validation_total_error_tmp = np.sum(np.array(validation_error_tmp, dtype=np.float64)*np.array(num_valid, dtype=np.float64))/np.sum(np.array(num_valid, dtype=np.float64))
                test_total_error_tmp = np.sum(np.array(test_error_tmp, dtype=np.float64)*np.array(num_test, dtype=np.float64))/np.sum(np.array(num_test, dtype=np.float64))

                train_error, valid_error, test_error = np.sqrt(np.sum(np.array(train_error_tmp))/num_task), np.sqrt(np.sum(np.array(validation_error_tmp))/num_task), np.sqrt(np.sum(np.array(test_error_tmp))/num_task)
                train_error_tmp, validation_error_tmp, test_error_tmp = list(np.sqrt(np.array(train_error_tmp))), list(np.sqrt(np.array(validation_error_tmp))), list(np.sqrt(np.array(test_error_tmp)))
                train_total_error, validation_total_error, test_total_error = np.sqrt(train_total_error_tmp), np.sqrt(validation_total_error_tmp), np.sqrt(test_total_error_tmp)

                if doLifelong:
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = train_error_tmp[task_for_train], validation_error_tmp[task_for_train], test_error_tmp[task_for_train]
                else:
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = train_error, valid_error, test_error

            #### current parameter of model
            #curr_param = sess.run(learning_model.param)

            #### error related process
            print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_error_to_compare), abs(valid_error_to_compare)))

            if valid_error_to_compare < best_valid_error:
                str_temp = ''
                if valid_error_to_compare < best_valid_error * improvement_threshold:
                    patience = max(patience, (learning_step-epoch_bias)*patience_multiplier)
                    str_temp = '\t<<'
                best_valid_error, best_epoch = valid_error_to_compare, learning_step
                test_error_at_best_epoch = test_error_to_compare
                print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_error_at_best_epoch), str_temp))

            if doLifelong and learning_step >= epoch_bias+min(patience, learning_step_max//num_task) and task_for_train < num_task-1:
                print('\n\t>>Change to new task!<<\n')
                # update epoch_bias, task_for_train, task_change_epoch
                epoch_bias, task_for_train = learning_step, task_for_train + 1
                task_change_epoch.append(learning_step+1)

                # initialize best_valid_error, best_epoch, patience
                patience = train_hyperpara['patience']
                best_valid_error, best_epoch = np.inf, -1

            train_error_hist.append(train_error_tmp + [abs(train_error)])
            valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
            test_error_hist.append(test_error_tmp + [abs(test_error)])
            best_test_error_hist.append(abs(test_error_at_best_epoch))
            train_total_error_hist.append(train_total_error)
            valid_total_error_hist.append(validation_total_error)
            test_total_error_hist.append(test_total_error)

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))
    if not doLifelong:
        print("Best validation error : %.4f (at epoch %d)" %(abs(best_valid_error), best_epoch))
        print("Test error at that epoch (%d) : %.4f" %(best_epoch, abs(test_error_at_best_epoch)))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['best_epoch'] = best_epoch
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist
    result_summary['history_train_total_error'] = train_total_error_hist
    result_summary['history_validation_total_error'] = valid_total_error_hist
    result_summary['history_test_total_error'] = test_total_error_hist
    result_summary['best_validation_error'] = abs(best_valid_error)
    result_summary['test_error_at_best_epoch'] = abs(test_error_at_best_epoch)
    if doLifelong:
        result_summary['task_changed_epoch'] = task_change_epoch

    #tfboard_writer.close()
    return result_summary, learning_model.num_trainable_var
