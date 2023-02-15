#!/bin/bash

echo $PYTHONPATH
export PYTHONPATH=/home/nunzio/caffe/python

preprocessing_debug_mode=1
behavior_characterization_debug_mode=1
anomaly_detection_debug_mode=1
mode=inference
anomalous_dataset_type=A
n_monitor_layers=2
preprocessing_test_percentage=0.5
normalize=Y
final_compression=N
final_compression_components=128
ad_technique=HC
behavior_characterization_validation_percentage=0.5
numero_monitor=2
tipo_dataset=A

#rm -f Input/Inference/PP/Data/*
#rm -f Output/Inference/PP/Data/*
#rm -f Input/Inference/AD/Data/*

python3 cnn_inference_3.py $numero_monitor $tipo_dataset 

python3 preprocessing_3.py $preprocessing_debug_mode $mode $anomalous_dataset_type $n_monitor_layers $preprocessing_test_percentage $normalize $final_compression $final_compression_components 

cp Output/Inference/PP/Data/*  Input/Inference/AD/Data

python3 anomaly_detection_3.py $anomaly_detection_debug_mode $mode $ad_technique 

