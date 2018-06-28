import os
import timeit
from random import shuffle

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class STL_Random_Forest_Regressor():
    def __init__(self, model_hyperpara, num_tasks):
        self.num_tasks = num_tasks
        self.model = [RandomForestRegressor(n_estimators=model_hyperpara["num_estimators"], criterion=model_hyperpara["split_crit"]) for _ in range(self.num_tasks)]

    def fit(self, x, y, task):
        print("\tTraining Random Forest model for task %d" %(task))
        self.model[task].fit(x, y)

    def predict(self, x, task):
        return self.model[task].predict(x)

    def get_param(self, task):
        return self.model[task].get_params()

    def get_params(self):
        return [self.get_param(task_cnt) for task_cnt in range(self.num_tasks)]

    def score(self, x, y, task):
        return self.model[task].score(x, y)


#### function to generate appropriate deep neural network
def model_generation(model_type, model_hyperpara, data_info):
    learning_model, gen_model_success = None, True

    if data_info is not None:
        x_dim, y_dim, num_task = data_info

    if model_type == 'mtl_rndforest':
        print("Training MTL-Random Forest model (Single model ver.)")
        learning_model = RandomForestRegressor(n_estimators=model_hyperpara["num_estimators"], criterion=model_hyperpara["split_crit"])
    elif model_type == 'mtl_several_rndforest':
        print("Training several MTL-Random Forest model (Single task ver.)")
        learning_model = STL_Random_Forest_Regressor(model_hyperpara, num_task)
    else:
        print("No such model exists!!")
        print("No such model exists!!")
        print("No such model exists!!")
        gen_model_success = False
    return (learning_model, gen_model_success)


#### module of training/testing one model
def train_rand_forest(model_type, model_hyperpara, dataset, data_info):
    print("Training Random Forest")

    ### set-up data
    train_data, validation_data, test_data = dataset
    x_dim, y_dim, num_task = data_info
    num_train, num_valid, num_test = [x[0].shape[0] for x in train_data], [x[0].shape[0] for x in validation_data], [x[0].shape[0] for x in test_data]

    ### Generate Model
    learning_model, generation_success = model_generation(model_type, model_hyperpara, data_info)
    if not generation_success:
        return (None, None, None, None)

    ### Lists to store intermediate result
    train_error, valid_error, test_error = [], [], []
    train_predict, valid_predict, test_predict = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    train_true_output, valid_true_output, test_true_output = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    if model_type == 'mtl_rndforest':
        train_x, train_y = train_data[0][0], train_data[0][1]
        for task_cnt in range(1, num_task):
            train_x = np.concatenate((train_x, train_data[task_cnt][0]), axis=0)
            train_y = np.concatenate((train_y, train_data[task_cnt][1]), axis=0)

    start_time = timeit.default_timer()
    for task_cnt in range(num_task):
        print("\tStart training of %d-th task" %(task_cnt))

        ### Training and Prediction
        if model_type == 'mtl_rndforest':
            if task_cnt < 1:
                learning_model.fit(train_x, np.squeeze(train_y))

            train_predict_tmp = learning_model.predict(train_data[task_cnt][0])
            valid_predict_tmp = learning_model.predict(validation_data[task_cnt][0])
            test_predict_tmp = learning_model.predict(test_data[task_cnt][0])

        elif model_type == 'mtl_several_rndforest':
            learning_model.fit(train_data[task_cnt][0], np.squeeze(train_data[task_cnt][1]), task_cnt)

            train_predict_tmp = learning_model.predict(train_data[task_cnt][0], task_cnt)
            valid_predict_tmp = learning_model.predict(validation_data[task_cnt][0], task_cnt)
            test_predict_tmp = learning_model.predict(test_data[task_cnt][0], task_cnt)

        ### Task-wise Error Computation
        train_error.append(np.sqrt(np.mean(np.square(train_predict_tmp - train_data[task_cnt][1]))))
        valid_error.append(np.sqrt(np.mean(np.square(valid_predict_tmp - validation_data[task_cnt][1]))))
        test_error.append(np.sqrt(np.mean(np.square(test_predict_tmp - test_data[task_cnt][1]))))

        ### Collecting all prediction and true output
        train_predict = np.concatenate((train_predict, train_predict_tmp))
        train_true_output = np.concatenate((train_true_output, np.squeeze(train_data[task_cnt][1])))
        valid_predict = np.concatenate((valid_predict, valid_predict_tmp))
        valid_true_output = np.concatenate((valid_true_output, np.squeeze(validation_data[task_cnt][1])))
        test_predict = np.concatenate((test_predict, test_predict_tmp))
        test_true_output = np.concatenate((test_true_output, np.squeeze(test_data[task_cnt][1])))

    ### Overall Error Computation
    train_error.append(np.sqrt(np.mean(np.square(train_predict - train_true_output))))
    valid_error.append(np.sqrt(np.mean(np.square(valid_predict - valid_true_output))))
    test_error.append(np.sqrt(np.mean(np.square(test_predict - test_true_output))))

    param = learning_model.get_params()

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" % (end_time - start_time))
    print("Train Error : %.4f" % (train_error[-1]))
    print("Validation Error : %.4f" % (valid_error[-1]))
    print("Test Error : %.4f" % (test_error[-1]))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['train_error'] = train_error
    result_summary['validation_error'] = valid_error
    result_summary['test_error'] = test_error

    result_summary['train_prediction'] = list(np.squeeze(train_predict))
    result_summary['validation_prediction'] = list(np.squeeze(valid_predict))
    result_summary['test_prediction'] = list(np.squeeze(test_predict))

    result_summary['train_true_output'] = list(np.squeeze(train_true_output))
    result_summary['validation_true_outputn'] = list(np.squeeze(valid_true_output))
    result_summary['test_true_output'] = list(np.squeeze(test_true_output))

    #tfboard_writer.close()
    return result_summary, -1