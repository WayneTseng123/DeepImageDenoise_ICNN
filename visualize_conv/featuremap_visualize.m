clear;
clc;    %清除当前command区域的命令
close all;

caffe.set_mode_cpu();   %选择gpu模式

net_model =  './SRCNN_deploy.prototxt';
net_weights = './sigma25/model-train2/SRCNN_iter_200000.caffemodel';
phase = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

im=imread('./test14/lenna.bmp');
im=rgb2gray(im);
figure();imshow(im);title('Original Image');

input_data={prepare_image(im)};

scores=net.forward(input_data);

blob_names={'data','conv1','conv2','conv3'};
for i=1:length(blob_names)
    visualize_feature_maps(net,blob_names{i},1);
end