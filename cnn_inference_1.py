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

model_def1 = input_inference_model_dir +'resnet20_cifar10_' +str(numero_monitor)+ '.prototxt' 
model_weights1=input_inference_model_dir+ 'resnet20_cifar10.caffemodel' 
net1 = caffe.Net(model_def1, model_weights1,caffe.TEST)

mean= np.array([0.49139,0.48215,0.44653])

transformer1 = caffe.io.Transformer({'data': net1.blobs['data'].data.shape})
transformer1.set_transpose('data', (2,0,1)) 
transformer1.set_mean('data', mean)  
transformer1.set_raw_scale('data',1) 

        
image1= caffe.io.load_image(input_inference_data_dir + str(tipo_dataset)+'_dataset_cifar10/ae1.png')

transformed_image1 = transformer1.preprocess('data', image1)

net1.blobs['data'].data[...] =transformed_image1

output1 = net1.forward()

output_prob1 = output1['prob'][0] 

print ('predicted class is net1 :', output_prob1.argmax())




