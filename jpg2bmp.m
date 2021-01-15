jpgs = dir('./data/VOCtest100/*.jpg');  
 num_jpgs = length( jpgs );  
 for i = 1 : num_jpgs  
   jpg_file = fullfile( './data//VOCtest100/' , jpgs(i).name );  
   jpg     = imread( jpg_file );  
  
  % 第一步，解析文件名 jpg_file ,注意，pgm_file 包括路径+文件名+后缀 
   
   [ path , name , ext ] = fileparts( jpg_file ) ;  
  
  % 第二步，生成新的文件名  
   filename = strcat( name , '.bmp' );  
  
  % 第三步，生成文件全称  
   bmp_file = fullfile( './data/VOC_testbmp' , filename ) ;  
    
  imwrite( jpg , bmp_file , 'bmp' );  
  
 end  