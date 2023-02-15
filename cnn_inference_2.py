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



model_def2 = input_inference_model_dir +'resnet32_cifar10_' +str(numero_monitor)+ '.prototxt' 
model_weights2 =input_inference_model_dir +'resnet32_cifar10.caffemodel' 
net2 = caffe.Net(model_def2, model_weights2,caffe.TEST)


mean= np.array([0.49139,0.48215,0.44653])


transformer2 = caffe.io.Transformer({'data': net2.blobs['data'].data.shape})
transformer2.set_transpose('data', (2,0,1)) 
transformer2.set_mean('data', mean)  
transformer2.set_raw_scale('data',1)   

image2= caffe.io.load_image(input_inference_data_dir +str(tipo_dataset)+'_dataset_cifar10/ae1.png')

transformed_image2 = transformer2.preprocess('data', image2)

net2.blobs['data'].data[...] =transformed_image2

output2 = net2.forward()

output_prob2 = output2['prob'][0] 

print ('predicted class is net2 :', output_prob2.argmax())




