"""This script passes the set of configurations to the lstm.ipynb notebook (i.e., DL code)"""

import os
import csv
import time
import subprocess
import datetime
import sys


architectures_implemented = ['lenet5', 'lstm']

architecture = sys.argv[1]

if architecture not in architectures_implemented:
    sys.exit('Please enter the name of an architecture that was implemented (i.e., {})'.format(architectures_implemented))

def is_failed(file_name):
    """This function makes sure that the file that contains the results of the
    corresponding model is complete (e.g., if for some reason the GPU utilization
    is not collected, the function will return true).
    
    Arguments
    ---------
    file_name: string 
        The name of the .txt that contains the results of the corresponding model.
    """
    is_fail = False
    with open(file_name, 'r', encoding='utf-8') as f:
        s = f.read()
        lines = s.split('\n')
        fieldNames = ['training_time', 'inference_time', 'accuracy',
                      'train_start_timestamp', 'train_end_timestamp',
                      'inference_start_timestamp', 'inference_end_timestamp',
                      'cpu_utilization_train', 'cpu_mem_train', 'gpu_utilization_train',
                      'gpu_mem_train', 'cpu_utilization_infer', 'cpu_mem_infer',
                      'gpu_utilization_infer', 'gpu_mem_infer']
        field_names_copy = fieldNames.copy()

        for line in lines:
            for fieldName in fieldNames:
                if fieldName in line:
                    field_names_copy.remove(fieldName)
                    val = line.split(' = ')[1]
                    if val == '[]' or val == '': 
                        is_fail = True
                        break
                        
    return is_fail or len(field_names_copy) != 0


def generate_arguments(dictionary):
    """Create a 'arguments.py' module containing a set of configurations.
    
    Arguments
    ---------
    dictionary: dict
        The values of the configurations of the model to be trained.
    """
    with open('arguments.py', 'w') as fid:
        for key in dictionary:
            fid.write(f'{key} = {repr(dictionary[key])}\n')


if architecture == 'lstm':
    dropout_prob_list = [0, 0.15, 0.25]

    training_size_list = [0.2, 0.35, 0.5]

    batch_size_list = [64, 256, 512]

    n_epochs_list = [3, 5, 10]

    learning_rate_list = [0.01, 0.005, 0.001]

    data_type_list = ['float32', 'mixed']

    device_list = ['gpu', 'cpu']

    weight_initialization_list = ['xavier', 'he']

    frameworks = ['PyTorch', 'TensorFlow', 'Keras']
    
    phases = ['training', 'inference']
else:
    dropout_prob_list = [0, 0.15, 0.25]

    training_size_list = [0.2, 0.35, 0.5]

    batch_size_list = [64, 256, 512]

    n_epochs_list = [5, 15, 30]

    learning_rate_list = [0.01, 0.05, 0.1]

    data_type_list = ['float32', 'mixed']

    device_list = ['gpu', 'cpu']

    weight_initialization_list = ['xavier', 'he']

    frameworks = ['PyTorch', 'TensorFlow', 'Keras']
    
    phases = ['training', 'inference']

for i in range(5):
    for dropout in dropout_prob_list:
        for training_size in training_size_list:
            for n_epochs in n_epochs_list:
                for device in device_list:
                    for batch_size in batch_size_list:
                        for weight_initialization in weight_initialization_list:
                            for data_type in data_type_list:
                                for learning_rate in learning_rate_list:
                                    for framework in frameworks:
                                        
                                        
                                        #cpu doesn't fully support float16
                                        if device == 'cpu' and data_type == 'mixed':
                                            continue

                                        experiment = '{}_{}{}_{}ts_{}batch_{}epochs_{}lr_{}dtype_{}_{}wi_{}dp'.format(architecture, framework, i, training_size,
                                                                                                                        batch_size, n_epochs,
                                                                                                                        learning_rate, data_type, device,
                                                                                                                        weight_initialization, dropout)

                                        #If we find that that we already have results for a model that corresponds to the given set of configurations
                                        #and that the results are complete, we skip to the next set of configurations.
                                        if os.path.isfile('./Results/{}/{}.txt'.format(architecture, experiment)):
                                            if not is_failed('./Results/{}/{}.txt'.format(architecture, experiment)):
                                                continue

                                        while True:
                                            for phase in phases:
                                                generate_arguments({
                                                    'i': i,
                                                    'training_size': training_size,
                                                    'batch_size': batch_size,
                                                    'n_epochs': n_epochs,
                                                    'learning_rate': learning_rate,
                                                    'data_type': data_type,
                                                    'device': device,
                                                    'weight_initialization': weight_initialization,
                                                    'framework': framework,
                                                    'dropout': dropout,
                                                    'phase': phase
                                                })

                                                os.system('jupyter nbconvert --execute --to notebook {}.ipynb --ExecutePreprocessor.timeout=9999'.format(architecture))


                                            #The while loop is terminated only if the model is succesfully trainined and evaluated.
                                            status = 'fail'
                                            if os.path.isfile('./Results/{}/{}.txt'.format(architecture, experiment)):
                                                if not is_failed('./Results/{}/{}.txt'.format(architecture, experiment)):
                                                    status = 'success' 
                                            if status == 'success':
                                                break