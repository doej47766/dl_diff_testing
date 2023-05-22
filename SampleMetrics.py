"""This script is used to sample the hardware utilization metrics during the training and inference processes of the model.
The script takes the time interval at which the metrics are sampled (in seconds)"""

import os
import time
import csv
import GPUtil
import psutil
import datetime
import sys

architectures_implemented = ['lenet5', 'lstm']

architecture = sys.argv[1]

if architecture not in architectures_implemented:
    sys.exit('Please enter the name of an architecture that was implemented (i.e., {})'.format(architectures_implemented))

#The default time interval is 1 second
time_interval = 1

if len(sys.argv) > 2:
    #If the argument is invalid, we take the default value of 1 second
    try:
        time_interval = int(sys.argv[2])
    except:
        time_interval = 1


collect_time = time.time()
i = 2


file_path = './Results/{}/metric_sampling.csv'.format(architecture)

#This samples the hardware utilization metrics according to the time interval and writes the samples in a file, alongside
#the time stamp at which the metric was sampled
with open(file_path, 'w+', encoding='utf-8') as csv_file:
    fieldNames = ['CPU Utilization', 'CPU Memory', 'GPU Utilization', 'GPU Memory', 'Timestamp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldNames, lineterminator = '\n')
    
    firstRow = {}
    for fieldName in fieldNames:
        firstRow[fieldName] = fieldName
    writer.writerow(firstRow)
    
    while True:
        if time.time() - collect_time >= time_interval:
            
            metrics = [psutil.cpu_percent(),
                       (psutil.virtual_memory().used / psutil.virtual_memory().total) * 100,
                       GPUtil.getGPUs()[0].load * 100.0, 
                       (GPUtil.getGPUs()[0].memoryUsed / GPUtil.getGPUs()[0].memoryTotal) * 100,
                       datetime.datetime.now()]
            
            row = {}
            
            for fieldName, metric in zip(fieldNames, metrics):
                row[fieldName] = metric
            
            writer.writerow(row)
            
            # if os.path.getsize(file_path) / 1000000 > 50:
            #     os.remove(file_path)