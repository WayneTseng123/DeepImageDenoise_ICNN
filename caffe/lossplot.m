%my loss  
clear;  
clc;  
close all;  
  
train_log_file = '200000train.log' ;  
train_interval = 100 ;  
test_interval = 500 ;  
  
[~, string_output] = dos(['type ' , train_log_file ]) ;  
pat='1 = .*? loss';  
o1=regexp(string_output,pat,'start');%用'start'参数指定输出o1为匹配正则表达式的子串的起始位置  
o2=regexp(string_output,pat,'end');%用'end'参数指定输出o2为匹配正则表达式的子串的结束位置  
o3=regexp(string_output,pat,'match');%用'match'参数指定输出o3为匹配正则表达式的子串   
  
loss=zeros(1,size(o1,2));  
for i=1:size(o1,2)  
    loss(i)=str2num(string_output(o1(i)+4:o2(i)-5));  
end 
plot(loss);
axis([0,2400,0,30.0]);
xlabel('iteration(x10^2)');
ylabel('loss');
grid on;