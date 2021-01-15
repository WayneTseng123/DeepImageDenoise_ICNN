function crops_data = prepare_image(im)

CROPPED_DIM = 256;
im_data = im(:,:);
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [CROPPED_DIM CROPPED_DIM], 'bilinear');  % resize im_data

crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 1, 1, 'single');
crops_data(:,:,1,1) = im_data(:,:);

