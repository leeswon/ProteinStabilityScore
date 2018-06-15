from read_data import read_data
from train_wrapper import train_run_for_each_model

_train_file_name = 'consistent_normalized_training_data_v1.csv'
_test_file_name = 'consistent_normalized_testing_data_v1.csv'

def model_setup(model_type, num_task, test_type):
    model_hyperpara = {}
    model_hyperpara['batch_size'] = 32
    model_hyperpara['hidden_layer'] = [64, 32, 16]

    if model_type.lower() == 'stl':
        model_architecture = 'mtl_several_ffnn_minibatch'
    elif model_type.lower() == 'snn':
        model_architecture = 'mtl_ffnn_minibatch'
    elif model_type.lower() == 'hps':
        model_architecture = 'mtl_ffnn_hard_para_sharing_minibatch'
        model_hyperpara['hidden_layer'] = [64, 32]
        model_hyperpara['task_specific_layer'] = [[16] for _ in range(num_task)]
    elif model_type.lower() == 'tf':
        model_architecture = 'mtl_ffnn_tensor_factor_minibatch'
        model_hyperpara['hidden_layer'] = [64, 32]
        model_hyperpara['task_specific_layer'] = [[16] for _ in range(num_task)]
        model_hyperpara['tensor_factor_type'] = 'Tucker'
        model_hyperpara['tensor_factor_error_threshold'] = 1e-3
    else:
        model_architecture = 'wrong_model'

    if test_type == 0:
        model_hyperpara['regularization_scale'] = 1e-7
    elif test_type == 1:
        model_hyperpara['regularization_scale'] = 1e-9
    elif test_type == 2:
        model_hyperpara['regularization_scale'] = 1e-11
    elif test_type == -1:
        model_hyperpara['regularization_scale'] = 0.0

    return model_architecture, model_hyperpara


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', help='Use GPU to train NN', action='store_true', default=False)
    parser.add_argument('--gpu_device', help='GPU device ID', type=int, default=0)
    parser.add_argument('--model_type', help='Architecture of Model(STL/SNN/HPS/TF/PROG/Deconv/DeconvTM/DeconvTM2)', type=str, default='STL')
    parser.add_argument('--test_type', help='Type of test (including regularization scale or etc)', type=int, default=0)
    parser.add_argument('--all_output', help='Train on all outputs, not final stability score', action='store_true', default=False)
    parser.add_argument('--save_mat_name', help='Name of file to save training results', type=str, default='delete_this.mat')
    parser.add_argument('--lifelong', help='Train in lifelong learning setting', action='store_true', default=False)
    args = parser.parse_args()

    do_lifelong = args.lifelong
    mat_file_name = args.save_mat_name

    train_hyperpara = {}
    train_hyperpara['improvement_threshold'] = 1.002  # for accuracy (maximizing it)
    train_hyperpara['patience_multiplier'] = 1.5
    train_hyperpara['lr'] = 0.01
    train_hyperpara['lr_decay'] = 1.0 / 100.0
    train_hyperpara['num_run_per_model'] = 5
    train_hyperpara['learning_step_max'] = 10000
    train_hyperpara['patience'] = 500

    data_hyperpara = {}
    data_hyperpara['folder_name'] = 'Data'
    data_hyperpara['train_file_name'] = _train_file_name
    data_hyperpara['test_file_name'] = _test_file_name
    data_hyperpara['train_valid_ratio'] = [0.8, 0.2]
    data_hyperpara['all_output'] = args.all_output

    _, datainfo = read_data(data_hyperpara['folder_name'], data_hyperpara['train_file_name'], data_hyperpara['test_file_name'], data_hyperpara['train_valid_ratio'], data_hyperpara['all_output'])
    model_architecture, model_hyperpara = model_setup(args.model_type, datainfo[2], args.test_type)
    train_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, data_hyperpara, mat_file_name, useGPU=args.use_gpu, GPU_device=args.gpu_device, doLifelong=do_lifelong)

if __name__ == '__main__':
    main()
