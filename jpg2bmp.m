jpgs = dir('./data/VOCtest100/*.jpg');  
 num_jpgs = length( jpgs );  
 for i = 1 : num_jpgs  
   jpg_file = fullfile( './data//VOCtest100/' , jpgs(i).name );  
   jpg     = imread( jpg_file );  
  
  % ��һ���������ļ��� jpg_file ,ע�⣬pgm_file ����·��+�ļ���+��׺ 
   
   [ path , name , ext ] = fileparts( jpg_file ) ;  
  
  % �ڶ����������µ��ļ���  
   filename = strcat( name , '.bmp' );  
  
  % �������������ļ�ȫ��  
   bmp_file = fullfile( './data/VOC_testbmp' , filename ) ;  
    
  imwrite( jpg , bmp_file , 'bmp' );  
  
 end  