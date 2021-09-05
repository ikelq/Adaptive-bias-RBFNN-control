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
x_min=-2.1;
x_max=2.1;
step=(x_max-x_min)/100;
x=linspace(x_min,x_max,100);
constans=ones(1,length(x));
approximation_aim=3*x.*(x-1).*(x-1.9).*(x+0.7).*(x+1.8);
hidden_node=linspace(x_min,x_max,25);
number_node=length(hidden_node);
train_loop_number=10;
mode=2;
switch mode
    case 0
        bias=-sum(approximation_aim)/length(approximation_aim);
        approximation_aim=approximation_aim+bias;
        gamma=6;  %learning rate
        variance=0.15; 
        test_bias_1=[0,0.01];
        test_bias_2=[0.1];
    case 1
        bias=max(abs(approximation_aim));
        approximation_aim=approximation_aim+bias;
        gamma=6;  %learning rate
        variance=0.15; 
        test_bias_1=[0,0.04,0.045,0.05,0.055,0.06];
        test_bias_2=[0.25,0.3,0.35,0.4,0.45,0.5];
    case 2
        bias=2*max(abs(approximation_aim));
        approximation_aim=approximation_aim+bias;
        gamma=6;  %learning rate
        variance=0.15; 
%         test_bias_1=[0,0.02,0.03,0.04,0.05,0.06];
%         test_bias_2=[0.3,0.35,0.4,0.45,0.5,0.55];
        test_bias_1=[0,0.03,0.04,0.05,0.055,0.06];
        test_bias_2=[0.3,0.35,0.4,0.45,0.5,0.55];
    case 3
    bias=4*max(abs(approximation_aim));
    approximation_aim=approximation_aim+bias;
    gamma=6;  %learning rate
    variance=0.15; 
%         test_bias_1=[0,0.02,0.03,0.04,0.05,0.06];
%         test_bias_2=[0.3,0.35,0.4,0.45,0.5,0.55];
    test_bias_1=[0,0.03,0.04,0.05,0.055,0.06];
    test_bias_2=[0.3,0.35,0.4,0.45,0.5,0.55];
end
mean_y=sum(approximation_aim)/length(approximation_aim);
        
% for reduce the bias of function
%  bias=sum(approximation_aim)/length(approximation_aim);


% test bias on time
% test_bias_1=[0,0.1,0.65,0.7,0.75,0.8];
% test_bias_2=[3,3.5,4,4.5,5];
% reduce the bias

% % original function
% test_bias_1=[0,0.01,0.014,0.018,0.022];
% test_bias_2=[0.15,0.017,0.2,0.023,0.25,0.3];
% in term of bias is add for each activation
number_bias_1=length(test_bias_1);
W1=zeros(number_node,number_bias_1);  % function_i =W(:,i,j)'*S(Z);
 
for j=1:train_loop_number    
    for i=1:length(x)    
        S(:,i,1)=RBF(x(i),hidden_node, variance,number_node);    
        for k=1:number_bias_1
        S1(:,i,k)=S(:,i,1)+test_bias_1(k);
        e1(i,k)=approximation_aim(i)-W1(:,k)'*S1(:,i,k);
        dotW1(:,k)=gamma*S1(:,i,k)*e1(i,k);
        W1(:,k)=W1(:,k)+step*dotW1(:,k);
        end
    end
end
%test in same data
for k=1:number_bias_1
   error1(:,k)=approximation_aim-W1(:,k)'*S1(:,:,k);
end


% in term of bias is add as another activation which is can be seen as PID+RBF
number_bias_2=length(test_bias_2);
W2=zeros(number_node+1,number_bias_2);  

for j=1:train_loop_number    
    for i=1:length(x)    
        for k=1:number_bias_2
        S2(:,i,k)=[S(:,i,1);test_bias_2(k)];
        e2(i,k)=approximation_aim(i)-W2(:,k)'*S2(:,i,k);
        dotW2(:,k)=gamma*S2(:,i,k)*e2(i,k);
        W2(:,k)=W2(:,k)+step*dotW2(:,k);
        end
    end
end
for k=1:number_bias_2
   error2(:,k)=approximation_aim-W2(:,k)'*S2(:,:,k);
end


% test data
x_t=linspace(x_min,x_max,1000);
y_test=3*x_t.*(x_t-1).*(x_t-1.9).*(x_t+0.7).*(x_t+1.8)+bias;
% test for 1
for i=1:length(x_t)    
        S(:,i,1)=RBF(x_t(i),hidden_node, variance,number_node);    
        for k=1:number_bias_1
        S1(:,i,k)=S(:,i,1)+test_bias_1(k);
        error_test1(i,k)=y_test(i)-W1(:,k)'*S1(:,i,k);
        mean_error_1(i,k)=y_test(i)-mean_y;
        end
end



for k=1:number_bias_1
       norm_test1(k)=sqrt(sum(error_test1(:,k).^2)/length(x_t));
       %NRMSE_1(k)=sqrt(sum(error_test1(:,k).^2)/sum( mean_error_1(:,k).^2));
       NRMSE_1(k)=sqrt(sum(error_test1(:,k).^2)/sum(y_test(:).^2));
end

% test for 2
for i=1:length(x_t)    
        S(:,i,1)=RBF(x_t(i),hidden_node, variance,number_node);    
        for k=1:number_bias_2
        S2(:,i,k)=[S1(:,i,1);test_bias_2(k)];
        error_test2(i,k)=y_test(i)-W2(:,k)'*S2(:,i,k);   
        mean_error_2(i,k)=y_test(i)-mean_y;
        end
end

for k=1:number_bias_2
       norm_test2(k)=sqrt(sum(error_test2(:,k).^2)/length(x_t));
      % NRMSE_2(k)=sqrt(sum(error_test2(:,k).^2)/sum( mean_error_2(:,k).^2));
       NRMSE_2(k)=sqrt(sum(error_test2(:,k).^2)/sum(y_test(:).^2));
end

vecnorm(W1)
vecnorm(W2)


mean_y
Fbl=sum(W1,1).*test_bias_1
Fbg=W2(26,:).*test_bias_2

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
figure
plot([ NRMSE_1, NRMSE_2])
[norm_test1,norm_test2]
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
