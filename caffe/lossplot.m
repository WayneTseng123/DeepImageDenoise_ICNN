%my loss  
clear;  
clc;  
close all;  
  
train_log_file = '200000train.log' ;  
train_interval = 100 ;  
test_interval = 500 ;  
  
[~, string_output] = dos(['type ' , train_log_file ]) ;  
pat='1 = .*? loss';  
o1=regexp(string_output,pat,'start');%��'start'����ָ�����o1Ϊƥ��������ʽ���Ӵ�����ʼλ��  
o2=regexp(string_output,pat,'end');%��'end'����ָ�����o2Ϊƥ��������ʽ���Ӵ��Ľ���λ��  
o3=regexp(string_output,pat,'match');%��'match'����ָ�����o3Ϊƥ��������ʽ���Ӵ�   
  
loss=zeros(1,size(o1,2));  
for i=1:size(o1,2)  
    loss(i)=str2num(string_output(o1(i)+4:o2(i)-5));  
end 
plot(loss);
axis([0,2400,0,30.0]);
xlabel('iteration(x10^2)');
ylabel('loss');
grid on;