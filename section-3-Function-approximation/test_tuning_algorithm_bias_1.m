%% The technical details can be seen in the paper
% @article{liu2021adaptive,
%   title={Adaptive bias RBF neural network control for a robotic manipulator},
%   author={Liu, Qiong and Li, Dongyu and Ge, Shuzhi Sam and Ji, Ruihang and Ouyang, Zhong and Tee, Keng Peng},
%   journal={Neurocomputing},
%   volume={447},
%   pages={213--223},
%   year={2021},
%   publisher={Elsevier}
% }


clear
clc
close all

%define function 
step=0.05;
x_min=-pi;
x_max=pi;
x=x_min:step:x_max;
constans=ones(length(x),1);
approximation_aim=-10*constans-10*sin(2*x')+10*x'+0*(x.^2)';
% function_dimension=1;
%parameter for train
gamma=0.05;  %learning rate
variance=0.5; 
hidden_node=x_min-0.25:0.5:x_max+0.7;
number_node=length(hidden_node);
train_loop_number=1000;

% test bias on time   0.4 0.3  0.5 0.225 0.6 0.226

% in term of bias is add for each activation
test_bias_1=[0,0.6,0.7,0.8,0.9];
number_bias_1=length(test_bias_1);
W1=zeros(number_node,number_bias_1);  % function_i =W(:,i,j)'*S(Z);
 
for j=1:train_loop_number    
    for i=1:length(x)    
        S1(:,i,1)=RBF(x(i),hidden_node, variance,number_node);    
        for k=1:number_bias_1
        S1(:,i,k)=S1(:,i,1)+test_bias_1(k);
        e1(i,k)=approximation_aim(i)-W1(:,k)'*S1(:,i,k);
        dotW1(:,k)=gamma*S1(:,i,k)*e1(i,k);
        W1(:,k)=W1(:,k)+step*dotW1(:,k);
        end
    end
end
%test in same data
for k=1:number_bias_1
   error1(:,k)=approximation_aim'-W1(:,k)'*S1(:,:,k);
end


% in term of bias is add as another activation which is can be seen as PID+RBF
test_bias_2=[5.5,6,6.5,7,7.5];
number_bias_2=length(test_bias_2);
W2=zeros(number_node+1,number_bias_2);  

for j=1:train_loop_number    
    for i=1:length(x)    
        for k=1:number_bias_2
        S2(:,i,k)=[S1(:,i,1);test_bias_2(k)];
        e2(i,k)=approximation_aim(i)-W2(:,k)'*S2(:,i,k);
        dotW2(:,k)=gamma*S2(:,i,k)*e2(i,k);
        W2(:,k)=W2(:,k)+step*dotW2(:,k);
        end
    end
end
for k=1:number_bias_2
   error2(:,k)=approximation_aim'-W2(:,k)'*S2(:,:,k);
end


% test data
x=x+step/2;
y_test=-10*constans-10*sin(2*x')+10*x'+0*(x.^2)';
% test for 1
for i=1:length(x)    
        S1(:,i,1)=RBF(x(i),hidden_node, variance,number_node);    
        for k=1:number_bias_1
        S1(:,i,k)=S1(:,i,1)+test_bias_1(k);
        error_test1(i,k)=y_test(i)-W1(:,k)'*S1(:,i,k);
        end
end

for k=1:number_bias_1
       norm_test1(k)=sqrt(sum(error_test1(:,k).^2)/length(x));
end

% test for 2
for i=1:length(x)    
        S1(:,i,1)=RBF(x(i),hidden_node, variance,number_node);    
        for k=1:number_bias_2
        S2(:,i,k)=[S1(:,i,1);test_bias_2(k)];
        error_test2(i,k)=y_test(i)-W2(:,k)'*S2(:,i,k);        
        end
end

for k=1:number_bias_2
       norm_test2(k)=sqrt(sum(error_test2(:,k).^2)/length(x));
end

figure
plot(e1);
hold on 
plot(e2);
legend

figure
plot(error1);
hold on 
plot(error2);
legend

figure
plot(error_test1);
hold on 
plot(error_test2);
legend

figure
plot([norm_test1,norm_test2])

%plot(x,e1,'r',x,e2,'g',x,e3,'b',x,en1,'c',x,en2,'m')
% legend;
% norme=[norm(e1),norm(e2),norm(e3),norm(en1),norm(en2)]
% figure
% error1=approximation_aim'-W1'*S1;
% error2=approximation_aim'-W2'*S2;
% error3=approximation_aim'-W3'*S3;
% 
% error_n1=approximation_aim'-Wn1'*Sn1;
% error_n2=approximation_aim'-Wn2'*Sn2;
% 
% 
% plot(x,error1,'r',x,error2,'g',x,error3,'b',x,error_n1,'c',x,error_n2,'m')
% legend
% 
% normerror=[norm(error1),norm(error2),norm(error3),norm(error_n1),norm(error_n2)]
% 
% % plot(approximation_aim)
