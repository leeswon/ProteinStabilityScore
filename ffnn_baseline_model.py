import numpy as np
import tensorflow as tf

from utils_nn import *

#################################################
############ Simple Feedforward Net #############
#################################################
#### Feedforward Neural Net
class FFNN_minibatch():
    def __init__(self, dim_layers, learning_rate, learning_rate_decay=-1):
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay

        #### placeholder of model
        self.model_input = tf.placeholder(tf.float32, [None, self.layers_size[0]])
        self.true_output = tf.placeholder(tf.float32, [None, self.layers_size[-1]])
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model
        self.layers, self.param = new_fc_net(self.model_input, self.layers_size[1:], params=None)

        #### functions of model
        self.eval = self.layers[-1]
        self.loss = 2.0 * tf.nn.l2_loss(self.eval - self.true_output)
        if self.learn_rate_decay <= 0:
            self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        else:
            self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0 + self.epoch*self.learn_rate_decay)).minimize(self.loss)

        self.num_trainable_var = count_trainable_var()

########################################################
####    Feedforward Net for Single-task Learning    ####
########################################################
#### FFNN3 model for MTL
class MTL_several_FFNN_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers) - 1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.layers_size[-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model for train data
        self.train_models, self.param, reg_param = [], [], []
        for task_cnt in range(self.num_tasks):
            model_tmp, param_tmp = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=None)
            self.train_models.append(model_tmp)
            self.param.append(param_tmp)
            reg_param.append(param_tmp[0::2])

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=self.param[task_cnt])
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=self.param[task_cnt])
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, _, _, _ = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [self.true_output, self.true_output, self.true_output], num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = []
            for param_list in reg_param:
                reg_term = 0.0
                for p in param_list:
                    reg_term = reg_term + tf.reduce_sum(tf.abs(p))
                reg_loss.append(reg_term)

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss[x]) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate / (1.0 + self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()


########################################################
#### Single Feedforward Net for Multi-task Learning ####
########################################################
#### FFNN3 model for MTL
class MTL_FFNN_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.layers_size[-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.param = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=None)
            else:
                model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=self.param)
            self.train_models.append(model_tmp)
        reg_param = self.param[0::2]

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=self.param)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.layers_size[1:], params=self.param)
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, _, _, _ = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [self.true_output, self.true_output, self.true_output], num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = 0.0
            for p in reg_param:
                reg_loss = reg_loss + tf.reduce_sum(tf.abs(p))

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()


########################################################
#### Hard Parameter Sharing for Multi-task Learning ####
########################################################
class MTL_FFNN_HPS_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0):
        self.num_tasks = num_tasks
        self.shared_layers_size = dim_layers[0]
        self.task_specific_layers_size = dim_layers[1]
        self.num_layers = [len(self.shared_layers_size)-1] + [len(self.task_specific_layers_size[x]) for x in range(self.num_tasks)]

        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.shared_layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.task_specific_layers_size[0][-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model for train data
        self.train_models, self.specific_param = [], []
        for task_cnt in range(self.num_tasks):
            #### generate network common to tasks
            if task_cnt == 0:
                shared_model_tmp, self.shared_param = new_fc_net(self.model_input[task_cnt], self.shared_layers_size[1:], params=None)
            else:
                shared_model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.shared_layers_size[1:], params=self.shared_param)

            #### generate task-dependent network
            specific_model_tmp, ts_params = new_fc_net(shared_model_tmp[-1], self.task_specific_layers_size[task_cnt], params=None)

            self.train_models.append(shared_model_tmp + specific_model_tmp)
            self.specific_param.append(ts_params)
        self.param = self.shared_param + sum(self.specific_param, [])
        reg_param = self.shared_param[0::2]

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            #### generate network common to tasks
            shared_model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.shared_layers_size[1:], params=self.shared_param)

            #### generate task-dependent network
            specific_model_tmp, _ = new_fc_net(shared_model_tmp[-1], self.task_specific_layers_size[task_cnt], params=self.specific_param[task_cnt])

            self.valid_models.append(shared_model_tmp + specific_model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            #### generate network common to tasks
            shared_model_tmp, _ = new_fc_net(self.model_input[task_cnt], self.shared_layers_size[1:], params=self.shared_param)

            #### generate task-dependent network
            specific_model_tmp, _ = new_fc_net(shared_model_tmp[-1], self.task_specific_layers_size[task_cnt], params=self.specific_param[task_cnt])

            self.test_models.append(shared_model_tmp + specific_model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, _, _, _ = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [self.true_output, self.true_output, self.true_output], num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = 0.0
            for p in reg_param:
                reg_loss = reg_loss + tf.reduce_sum(tf.abs(p))

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()


########################################################
####  Tensor Factorization for Multi-task Learning  ####
########################################################
class MTL_FFNN_Tensor_Factor_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay=-1, l1_reg_scale=0.0, factor_type='Tucker', factor_eps_or_k=0.01):
        self.num_tasks = num_tasks
        self.shared_layers_size = dim_layers[0]
        self.task_specific_layers_size = dim_layers[1]
        self.num_layers = [len(self.shared_layers_size)-1] + [len(self.task_specific_layers_size[x]) for x in range(self.num_tasks)]

        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.l1_reg_scale = l1_reg_scale
        self.batch_size = batch_size

        #### placeholder of model
        self.model_input = [tf.placeholder(tf.float32, [None, self.shared_layers_size[0]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(tf.float32, [None, self.task_specific_layers_size[0][-1]]) for _ in range(self.num_tasks)]
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model for train data
        self.train_models, self.shared_param, self.specific_param = new_tensorfactored_fc_fc_nets(self.model_input, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, activation_fn=tf.nn.relu, shared_params=None, specific_params=None, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)
        self.param = self.shared_param + self.specific_param
        reg_param = self.shared_param[0::2]

        #### layers of model for validation data
        self.valid_models, _, _ = new_tensorfactored_fc_fc_nets(self.model_input, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, activation_fn=tf.nn.relu, shared_params=self.shared_param, specific_params=self.specific_param, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)

        #### layers of model for test data
        self.test_models, _, _ = new_tensorfactored_fc_fc_nets(self.model_input, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, activation_fn=tf.nn.relu, shared_params=self.shared_param, specific_params=self.specific_param, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, _, _, _ = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [self.true_output, self.true_output, self.true_output], num_tasks)

        with tf.name_scope('L1_regularization'):
            reg_loss = 0.0
            for p in reg_param:
                reg_loss = reg_loss + tf.reduce_sum(tf.abs(p))

        if learning_rate_decay <= 0:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]
        else:
            self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x] + self.l1_reg_scale * reg_loss) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()