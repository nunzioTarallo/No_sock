import numpy as np
import glob
import sys
import caffe


input_inference_dir= "Input/Inference/CI/"
input_inference_data_dir= input_inference_dir + "Data/"
input_inference_model_dir= input_inference_dir + "Model/"

numero_monitor = int(sys.argv[1])
tipo_dataset = sys.argv[2]

caffe.set_mode_cpu() 

model_def3 = input_inference_model_dir+ 'resnet56_cifar10_' +str(numero_monitor)+ '.prototxt' 
model_weights3 = input_inference_model_dir+ 'resnet56_cifar10.caffemodel' 
net3 = caffe.Net(model_def3, model_weights3,caffe.TEST)

mean= np.array([0.49139,0.48215,0.44653])

transformer3 = caffe.io.Transformer({'data': net3.blobs['data'].data.shape})
transformer3.set_transpose('data', (2,0,1)) 
transformer3.set_mean('data', mean)  
transformer3.set_raw_scale('data',255)  

image3= caffe.io.load_image(input_inference_data_dir +str(tipo_dataset)+'_dataset_cifar10/ae1.png')
transformed_image3 = transformer3.preprocess('data', image3)
net3.blobs['data'].data[...] =transformed_image3

output3 = net3.forward()

output_prob3 = output3['prob'][0] 

print ('predicted class is net3 :', output_prob3.argmax())



