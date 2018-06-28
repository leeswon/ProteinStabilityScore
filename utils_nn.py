import tensorflow as tf
import numpy as np
from utils_tensor_factorization import TensorProducer

#### function to count trainable parameters in computational graph
def count_trainable_var():
    total_para_cnt = 0
    for variable in tf.trainable_variables():
        para_cnt_tmp = 1
        for dim in variable.get_shape():
            para_cnt_tmp = para_cnt_tmp * dim.value
        total_para_cnt = total_para_cnt + para_cnt_tmp
    return total_para_cnt


#### function to generate metrics of performance (eval, loss, accuracy)
def mtl_model_output_functions(models, y_batches, num_tasks, classification=False):
    if classification:
        with tf.name_scope('Model_Eval'):
            train_eval = [tf.nn.softmax(models[0][x][-1]) for x in range(num_tasks)]
            valid_eval = [tf.nn.softmax(models[1][x][-1]) for x in range(num_tasks)]
            test_eval = [tf.nn.softmax(models[2][x][-1]) for x in range(num_tasks)]

            train_output_label = [tf.argmax(models[0][x][-1], 1) for x in range(num_tasks)]
            valid_output_label = [tf.argmax(models[1][x][-1], 1) for x in range(num_tasks)]
            test_output_label = [tf.argmax(models[2][x][-1], 1) for x in range(num_tasks)]

        with tf.name_scope('Model_Loss'):
            train_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[0][x], tf.int32), logits=models[0][x][-1]) for x in range(num_tasks)]
            valid_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[1][x], tf.int32), logits=models[1][x][-1]) for x in range(num_tasks)]
            test_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[2][x], tf.int32), logits=models[2][x][-1]) for x in range(num_tasks)]

        with tf.name_scope('Model_Accuracy'):
            train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[0][x][-1], 1), tf.cast(y_batches[0][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
            valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[1][x][-1], 1), tf.cast(y_batches[1][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
            test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[2][x][-1], 1), tf.cast(y_batches[2][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
    else:
        with tf.name_scope('Model_Eval'):
            train_eval = [models[0][x][-1] for x in range(num_tasks)]
            valid_eval = [models[1][x][-1] for x in range(num_tasks)]
            test_eval = [models[2][x][-1] for x in range(num_tasks)]

        with tf.name_scope('Model_Loss'):
            train_loss = [2.0* tf.nn.l2_loss(train_eval[x]-y_batches[0][x]) for x in range(num_tasks)]
            valid_loss = [2.0* tf.nn.l2_loss(valid_eval[x]-y_batches[1][x]) for x in range(num_tasks)]
            test_loss = [2.0* tf.nn.l2_loss(test_eval[x]-y_batches[2][x]) for x in range(num_tasks)]

        train_accuracy, valid_accuracy, test_accuracy = None, None, None
        train_output_label, valid_output_label, test_output_label = None, None, None
    return (train_eval, valid_eval, test_eval, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy, train_output_label, valid_output_label, test_output_label)


###############################################
#### Functions to generate Neural Networks ####
###############################################

#### leaky ReLu
def leaky_relu(x_in, leaky_alpha=0.01):
    return tf.nn.relu(x_in) - leaky_alpha*tf.nn.relu(-x_in)


#### function to generate weight parameter
def new_weight(shape, trainable=True):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.2), trainable=trainable, dtype=tf.float32)

#### function to generate bias parameter
def new_bias(shape, trainable=True):
    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape), trainable=trainable, dtype=tf.float32)


#### function to generate fully-connected layer
def new_fc_layer(x_in, n_hidden, weight=None, bias=None, activation_fn=tf.nn.relu, trainable=True, tf_name_scope='fc_layer'):
    with tf.name_scope(tf_name_scope):
        n_in = int(x_in.shape[1])
        if weight is None:
            weight = new_weight([n_in, n_hidden], trainable)
        if bias is None:
            bias = new_bias([n_hidden], trainable)

        y = tf.matmul(x_in, weight) + bias
        if activation_fn is not None:
            y = activation_fn(y)
        return y, [weight, bias]

#### function to generate fully-connected network
def new_fc_net(x_in, n_hiddens, params=None, activation_fn=tf.nn.relu, trainable=True, tf_name_scope='fc_net'):
    with tf.name_scope(tf_name_scope):
        num_layers = len(n_hiddens)
        if num_layers < 1:
            return x_in, []
        else:
            if not (type(activation_fn) == list):
                activation_fn = [activation_fn for _ in range(num_layers-1)] + [None]

            if params is None:
                params = [None for _ in range(2*num_layers)]

            layers, params_return = [], []
            for layer_cnt in range(num_layers):
                if layer_cnt < 1:
                    layer_tmp, param_tmp = new_fc_layer(x_in, n_hiddens[layer_cnt], weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], activation_fn=activation_fn[layer_cnt], trainable=trainable)
                else:
                    layer_tmp, param_tmp = new_fc_layer(layers[-1], n_hiddens[layer_cnt], weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], activation_fn=activation_fn[layer_cnt], trainable=trainable)
                layers.append(layer_tmp)
                params_return = params_return + param_tmp
            return layers, params_return


#### function to generate parameters of tensor factored convolutional layer
def new_tensorfactored_weight(shape, num_task, factor_type='Tucker', factor_eps_or_k=0.01):
    if len(shape) == 2:
        W_init = np.random.rand(shape[0], shape[1], num_task)
    elif len(shape) == 4:
        W_init = np.random.rand(shape[0], shape[1], shape[2], shape[3], num_task)
    else:
        return (None, None)

    W_tmp, W_dict = TensorProducer(W_init, factor_type, eps_or_k=factor_eps_or_k, return_true_var=True)

    if len(shape) == 2:
        W = [W_tmp[:, :, i] for i in range(num_task)]
    elif len(shape) == 4:
        W = [W_tmp[:, :, :, :, i] for i in range(num_task)]
    return (W, W_dict)


def new_tensorfactored_fc_weights(hid_sizes, num_task, factor_type='Tucker', factor_eps_or_k=0.01):
    num_layers = len(hid_sizes)-1
    param_tmp = [[] for i in range(num_task)]
    for layer_cnt in range(num_layers):
        W_tmp, _ = new_tensorfactored_weight(hid_sizes[layer_cnt:layer_cnt+2], num_task, factor_type, factor_eps_or_k)
        bias_tmp = [new_bias(shape=[hid_sizes[layer_cnt+1]]) for i in range(num_task)]
        for task_cnt in range(num_task):
            param_tmp[task_cnt].append(W_tmp[task_cnt])
            param_tmp[task_cnt].append(bias_tmp[task_cnt])

    param = []
    for task_cnt in range(num_task):
        param = param + param_tmp[task_cnt]
    return param


def new_tensorfactored_fc_nets(net_inputs, hid_sizes, num_task, activation_fn=tf.nn.relu, params=None, factor_type='Tucker', factor_eps_or_k=0.01, trainable=True):
    num_para_per_model = 2*(len(hid_sizes)-1)

    with tf.name_scope('TF_fc_net'):
        if len(hid_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            return (net_inputs, [])
        elif params is None:
            #### network & parameters are new
            params = new_tensorfactored_fc_weights(hid_sizes, num_task, factor_type, factor_eps_or_k)
            fc_models = []
            for task_cnt in range(num_task):
                fc_model_tmp, _ = new_fc_net(net_inputs[task_cnt], hid_sizes[1:], activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], trainable=trainable)
                fc_models.append(fc_model_tmp)
        else:
            #### network generated from existing parameters
            fc_models = []
            for task_cnt in range(num_task):
                fc_model_tmp, _ = new_fc_net(net_inputs[task_cnt], hid_sizes[1:], activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], trainable=trainable)
                fc_models.append(fc_model_tmp)

    return (fc_models, params)


def new_tensorfactored_fc_fc_nets(net_inputs, hid_sizes_shared, hid_sizes_specific, num_task, activation_fn=tf.nn.relu, shared_params=None, specific_params=None, factor_type='Tucker', factor_eps_or_k=0.01, trainable=True):
    tf_shared_net, tf_shared_param = new_tensorfactored_fc_nets(net_inputs, hid_sizes_shared, num_task, activation_fn, shared_params, factor_type, factor_eps_or_k, trainable=trainable)

    num_acc_specific_fc_params, num_specific_fc_params_tmp = [0], 0
    for a in hid_sizes_specific:
        num_specific_fc_params_tmp += 2*len(a)
        num_acc_specific_fc_params.append(num_specific_fc_params_tmp)

    overall_net, tf_specific_param = [], []
    for task_cnt in range(num_task):
        if specific_params is None:
            fc_net_tmp, fc_param_tmp = new_fc_net(tf_shared_net[task_cnt][-1], hid_sizes_specific[task_cnt], activation_fn=activation_fn, trainable=trainable)
        else:
            fc_net_tmp, fc_param_tmp = new_fc_net(tf_shared_net[task_cnt][-1], hid_sizes_specific[task_cnt], activation_fn=activation_fn, params=specific_params[num_acc_specific_fc_params[task_cnt]:num_acc_specific_fc_params[task_cnt+1]], trainable=trainable)
        overall_net.append(tf_shared_net[task_cnt]+fc_net_tmp)
        tf_specific_param = tf_specific_param + fc_param_tmp

    return (overall_net, tf_shared_param, tf_specific_param)