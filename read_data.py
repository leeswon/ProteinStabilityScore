from csv import reader
from os import listdir, getcwd
from random import shuffle
from math import floor
import numpy as np


## process type of each column in csv file
class Column_Summary():
    def __init__(self, column_name):
        if column_name == 'stabilityscore':
            self.type = 'output_min'
        elif 'stabilityscore' in column_name:
            self.type = 'output'
        elif 'library' == column_name:
            self.type = 'lifelong'
        elif ('dslf_fa13' == column_name) or ('entropy' == column_name) or ('name' == column_name) or ('description' == column_name) or ('sequence' == column_name) or ('dssp' == column_name):
            self.type = 'no_use'
        else:
            self.type = column_name

    def isInput(self):
        return not ((self.type == 'output') or (self.type == 'output_min') or (self.type == 'no_use') or (self.type == 'lifelong'))

    def isOutput(self):
        return (self.type == 'output')

    def isOutputMin(self):
        return (self.type == 'output_min')

    def isMTLlabel(self):
        return ('lifelong' in self.type)


## organize columns of data according to their types
class Data_Summary():
    def __init__(self, title_row, use_all_output=True):
        self.num_columns = len(title_row)
        self.cols, self.input_indices, self.output_indices = [], [], []
        for col_cnt in range(self.num_columns):
            self.cols.append(Column_Summary(title_row[col_cnt]))
            if self.cols[-1].isInput():
                self.input_indices.append(col_cnt)
            elif self.cols[-1].isOutput() and use_all_output:
                self.output_indices.append(col_cnt)
            elif self.cols[-1].isOutputMin() and not use_all_output:
                self.output_indices.append(col_cnt)
            elif self.cols[-1].isMTLlabel():
                self.mtl_index = col_cnt


## split data into training/validation set according to given ratio
def split_train_valid(data, dratio):
    # list of (X, Y) as many as tasks
    num_tasks = len(data)

    a = sum(dratio)
    if a > 1.0:
        dratio = [x/a for x in dratio]

    train_data, valid_data = [], []
    for task_cnt in range(num_tasks):
        num_data = data[task_cnt][0].shape[0]
        num_train, num_valid = int(floor(num_data*dratio[0])), int(floor(num_data*dratio[1]))

        ind = np.arange(num_data)
        shuffle(ind)

        train_x, train_y = data[task_cnt][0][ind[0:num_train], :], data[task_cnt][1][ind[0:num_train], :]
        valid_x, valid_y = data[task_cnt][0][ind[num_train:num_train+num_valid], :], data[task_cnt][1][ind[num_train:num_train+num_valid], :]

        train_data.append((train_x, train_y))
        valid_data.append((valid_x, valid_y))
    return train_data, valid_data


## process data file
def process_data(csv_raw_data, use_all_output):
    mtl_data, curr_task, task_list = [], '', []
    for row_cnt, (row) in enumerate(csv_raw_data):
        if row_cnt == 0:
            title_row = list(row)
            train_data_descriptor = Data_Summary(title_row, use_all_output=use_all_output)
            num_input, num_output = len(train_data_descriptor.input_indices), len(train_data_descriptor.output_indices)
        else:
            if not (curr_task == row[train_data_descriptor.mtl_index]):
                if not (curr_task == ''):
                    num_data = len(data_bin[0])
                    X, Y = np.zeros((num_data, num_input), dtype=np.float32), np.zeros((num_data, num_output), dtype=np.float32)
                    for i, (cnt) in enumerate(train_data_descriptor.input_indices):
                        X[:, i] = np.array(data_bin[cnt])

                    for i, (cnt) in enumerate(train_data_descriptor.output_indices):
                        Y[:, i] = np.array(data_bin[cnt])
                    mtl_data.append((X, Y))

                curr_task = row[train_data_descriptor.mtl_index]
                task_list.append(curr_task)
                data_bin = [[] for _ in range(train_data_descriptor.num_columns)]

            for col_cnt in range(len(row)):
                data_bin[col_cnt].append(row[col_cnt])

    # for the data of the last task (it doesn't have the next task which differs from current task)
    num_data, num_input, num_output = len(data_bin[0]), len(train_data_descriptor.input_indices), len(train_data_descriptor.output_indices)
    X, Y = np.zeros((num_data, num_input), dtype=np.float32), np.zeros((num_data, num_output), dtype=np.float32)
    for i, (cnt) in enumerate(train_data_descriptor.input_indices):
        X[:, i] = np.array(data_bin[cnt])

    for i, (cnt) in enumerate(train_data_descriptor.output_indices):
        Y[:, i] = np.array(data_bin[cnt])
    mtl_data.append((X, Y))
    return mtl_data, task_list, num_input, num_output


## read csv file and organize into usable data
def read_data(folder_name, train_file_name, test_file_name, train_valid_ratio, use_all_output=True):
    if train_file_name in listdir(getcwd()+'/'+folder_name):
        with open(getcwd()+'/'+folder_name+'/'+train_file_name) as csv_obj:
            csv_raw_train_data = reader(csv_obj, delimiter=',')
            mtl_train_data, mtl_train_tasks, num_train_input, num_train_output = process_data(csv_raw_train_data, use_all_output)
            num_train_task = len(mtl_train_data)
            print("Successfully read train data! (Dim : %d/%d)" %(num_train_input, num_train_output))

    if test_file_name in listdir(getcwd()+'/'+folder_name):
        with open(getcwd()+'/'+folder_name+'/'+test_file_name) as csv_obj:
            csv_raw_test_data = reader(csv_obj, delimiter=',')
            mtl_test_data, mtl_test_tasks, num_test_input, num_test_output = process_data(csv_raw_test_data, use_all_output)
            num_test_task = len(mtl_test_data)
            print("Successfully read test data! (Dim : %d/%d)" %(num_test_input, num_test_output))

        assert (num_train_input == num_test_input and num_train_output == num_test_output), "Two dataset has different dimensionality!"
        assert (num_train_task == num_test_task), "Two dataset has different number of tasks!"
        assert (all([i==j for (i, j) in zip(mtl_train_tasks, mtl_test_tasks)])), "Order of tasks of train/test set!"

        mtl_train_data_set, mtl_valid_data_set = split_train_valid(mtl_train_data, train_valid_ratio)
        return [mtl_train_data_set, mtl_valid_data_set, mtl_test_data], [num_train_input, num_train_output, num_train_task]
    else:
        print("Failed to read data!!!")
        return None, None

#####################################################
#### Deprecated functions for older Rocklin data ####
#####################################################
class Column_Summary_15k():
    def __init__(self, input1, input2):
        self.name = input1
        if 'output' in input2:
            if '(' in input2:
                self.type = 'output_min'
            else:
                self.type = 'output'
        elif 'input' in input2:
            if 'all' in input2:
                self.type = 'input_all'
            elif '?' in input2:
                self.type = 'input?'
            elif 'EHEE' in input2:
                self.type = 'input_ehee'
            else:
                self.type = 'input_literature'
        elif 'use' in input2:
            self.type = 'no_use'
        elif 'stabilityscore' in input2:
            self.type = ''
        else:
            self.type = input2

    def isInput(self):
        return (self.type == 'input_all')

    def isOutput(self):
        return (self.type == 'output')

    def isOutputMin(self):
        return (self.type == 'output_min')

    def isMTLlabel(self):
        return ('lifelong' in self.type)


class Data_Summary_15k():
    def __init__(self, row1, row2, use_all_output=True):
        assert (len(row1) == len(row2)), 'Given two rows have different numbers of columns!'
        self.num_columns = len(row1)-1
        self.cols, self.input_indices, self.output_indices = [], [], []
        for col_cnt in range(1, self.num_columns+1):
            self.cols.append(Column_Summary_15k(row1[col_cnt], row2[col_cnt]))
            if self.cols[-1].isInput():
                self.input_indices.append(col_cnt-1)
            elif self.cols[-1].isOutput() and use_all_output:
                self.output_indices.append(col_cnt-1)
            elif self.cols[-1].isOutputMin() and not use_all_output:
                self.output_indices.append(col_cnt-1)
            elif self.cols[-1].isMTLlabel():
                self.mtl_index = col_cnt-1


def split_train_valid_test_15k(data, dratio):
    # list of (X, Y) as many as tasks
    num_tasks = len(data)

    a = sum(dratio)
    if a > 1.0:
        dratio = [x/a for x in dratio]

    train_data, valid_data, test_data = [], [], []
    for task_cnt in range(num_tasks):
        num_data = data[task_cnt][0].shape[0]
        num_train, num_valid, num_test = floor(num_data*dratio[0]), floor(num_data*dratio[1]), floor(num_data*dratio[2])

        ind = np.arange(num_data)
        shuffle(ind)

        train_x, train_y = data[task_cnt][0][ind[0:num_train], :], data[task_cnt][1][ind[0:num_train], :]
        valid_x, valid_y = data[task_cnt][0][ind[num_train:num_train+num_valid], :], data[task_cnt][1][ind[num_train:num_train+num_valid], :]
        test_x, test_y = data[task_cnt][0][ind[num_train+num_valid:num_train+num_valid+num_test], :], data[task_cnt][1][ind[num_train+num_valid:num_train+num_valid+num_test], :]

        train_data.append((train_x, train_y))
        valid_data.append((valid_x, valid_y))
        test_data.append((test_x, test_y))
    return [train_data, valid_data, test_data]


def read_data_15k(folder_name, file_name, train_valid_test_ratio, use_all_output=True):
    if file_name in listdir(getcwd()+'/'+folder_name):
        mtl_data, curr_task = [], ''
        with open(getcwd()+'/'+folder_name+'/'+file_name) as csv_obj:
            csv_raw_data = reader(csv_obj, delimiter=',')
            for row_cnt, (row) in enumerate(csv_raw_data):
                if row_cnt == 0:
                    row1 = list(row)
                elif row_cnt == 1:
                    row2 = list(row)
                    data_descriptor = Data_Summary_15k(row1, row2, use_all_output=use_all_output)
                    num_input, num_output = len(data_descriptor.input_indices), len(data_descriptor.output_indices)
                else:
                    if not (curr_task == row[data_descriptor.mtl_index+1]):
                        if not (curr_task == ''):
                            num_data = len(data_bin[0])
                            X, Y = np.zeros((num_data, num_input), dtype=np.float32), np.zeros((num_data, num_output), dtype=np.float32)
                            for i, (cnt) in enumerate(data_descriptor.input_indices):
                                X[:, i] = np.array(data_bin[cnt])

                            for i, (cnt) in enumerate(data_descriptor.output_indices):
                                Y[:, i] = np.array(data_bin[cnt])
                            mtl_data.append((X, Y))

                        curr_task = row[data_descriptor.mtl_index+1]
                        data_bin = [[] for _ in range(data_descriptor.num_columns)]

                    for col_cnt in range(1, len(row)):
                        data_bin[col_cnt-1].append(row[col_cnt])

            # for the data of the last task (it doesn't have the next task which differs from current task)
            num_data, num_input, num_output = len(data_bin[0]), len(data_descriptor.input_indices), len(data_descriptor.output_indices)
            X, Y = np.zeros((num_data, num_input), dtype=np.float32), np.zeros((num_data, num_output), dtype=np.float32)
            for i, (cnt) in enumerate(data_descriptor.input_indices):
                X[:, i] = np.array(data_bin[cnt])

            for i, (cnt) in enumerate(data_descriptor.output_indices):
                Y[:, i] = np.array(data_bin[cnt])
            mtl_data.append((X, Y))
            print("Successfully read data! (Dim : %d/%d)" %(num_input, num_output))
        return split_train_valid_test_15k(mtl_data, train_valid_test_ratio), [num_input, num_output, len(mtl_data)]
    else:
        print("Failed to read data!!!")
        return None, None


if __name__ == '__main__':
    folder_name = 'Data'
    #file_name = listdir(getcwd()+'/'+folder_name)
    #multi_task_data = read_data_15k('Data', file_name[0], [0.6, 0.1, 0.3])
    train_file_name, test_file_name = 'consistent_normalized_training_data_v1.csv', 'consistent_normalized_testing_data_v1.csv'
    multi_task_data = read_data('Data', train_file_name, test_file_name, [0.7, 0.3])
    print("Done!")