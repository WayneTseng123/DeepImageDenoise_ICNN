close all;
clear all;

%% settings
model = './deploy.prototxt';
image_path = '../data/Set12bmp/02.bmp';
step = 500;
min_iter = 500;
max_iter = 50000;

%% read ground truth image
im = imread(image_path);
%% work on illuminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end

% im_gnd = modcrop(im, up_scale);
im_gnd = single(im)/255;
% im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_input = imnoise(im_gnd,'gaussian',0,25^2/255^2);
% im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
% im_b = imresize(im_l, up_scale, 'bicubic');
im_gnd = shave(uint8(im_gnd * 255), [6, 6]);
[input_height ,input_width] = size(im_input);
input_channel = 1;
batch =1;
count = fix((max_iter - min_iter)/step) +1;
psnr = zeros(1,count);
count = 1;

%% load model
caffe.reset_all(); 
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(model,'test');
net.blobs('data').reshape([input_height input_width input_channel batch]); % reshape blob 'data'
net.reshape();
net.blobs('data').set_data(im_input);
for fid = min_iter : step : max_iter
    weights = ['./yuan_model/multi_scale_iter_' num2str(fid) '.caffemodel'];
    %% load weights
    net.copy_from(weights);
    %% denoise
    net.forward_prefilled();
    output = net.blobs('conv8').get_data();
    [output_height, output_width, output_channel] = size(output);
    % scaled_height = up_scale * output_height;
     % scaled_width = up_scale * output_width;
     im_h = zeros(output_height, output_width);
    
   % for m = 1 : up_scale
        %for n = 1 : up_scale
    im_h(1:output_height,1:output_width) = output(:,:,1);   
        %end
   % end
    im_h = uint8(im_h * 255);
    %% compute PSNR    
    psnr_srcnn = compute_psnr(im_gnd,im_h);
    psnr(count) = psnr_srcnn;
    count = count+1;
end
%% plot picture
x = min_iter : step : max_iter;
plot(x,psnr);
imwrite(im_h, ['compare' '.bmp']);