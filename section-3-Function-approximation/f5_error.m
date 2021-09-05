
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
x_min=-0;
x_max=1;
step=(x_max-x_min)/20;
x1=linspace(x_min,x_max,20);
x2=linspace(x_min,x_max,20);
[x1,x2]=meshgrid(x1,x2);
x=[reshape(x1,1,numel(x1));reshape(x2,1,numel(x2))];
approximation_aim= 1.9*(1.35+exp(x1).*sin(13*(x1-0.6).^2).*exp(-x2).*sin(7*x2));
%approximation_aim=4*x1.*x2;42.695
approximation_aim=reshape(approximation_aim,1,numel(approximation_aim));
hidden_node_x1=linspace(x_min,x_max,10);
hidden_node_x2=linspace(x_min,x_max,10);
[hidden_node_x1,hidden_node_x2]=meshgrid(hidden_node_x1,hidden_node_x2);
hidden_node=[reshape(hidden_node_x1,1, numel(hidden_node_x1));reshape(hidden_node_x2,1, numel(hidden_node_x2))];
number_node=length(hidden_node);
train_loop_number=10;

mode=0;
switch mode
    case 0
        bias=-sum(approximation_aim)/length(approximation_aim);
        approximation_aim=approximation_aim+bias;
        gamma=4;  %learning rate
        variance=0.08; 
        test_bias_1=[0];
        test_bias_2=[0.1];
    case 1
        bias=-sum(approximation_aim)/length(approximation_aim);
        approximation_aim=approximation_aim+bias;
        gamma=2;  %learning rate
        variance=0.2; 
        test_bias_1=[0,0.0005,0.001,0.002,0.003];
        test_bias_2=[0.002,0.006,0.01];
    case 2
        bias=10*max(abs(approximation_aim));
        approximation_aim=approximation_aim+bias;
        gamma=2;  %learning rate
        variance=0.2; 
        test_bias_1=[0,0.02,0.025,0.03,0.035,0.04,0.045,0.05];
        test_bias_2=[0.3,0.35,0.4,0.45,0.5,0.55,0.6];
end
        

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
        S(:,i,1)=RBF(x(:,i),hidden_node, variance,number_node);    
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
x1_t=linspace(x_min,x_max,31);
x2_t=linspace(x_min,x_max,31);
[x1_t,x2_t]=meshgrid(x1_t,x2_t);
x_t=[reshape(x1_t,1,numel(x1_t));reshape(x2_t,1,numel(x2_t))];
 y_test= 1.9*(1.35+exp(x1_t).*sin(13*(x1_t-0.6).^2).*exp(-x2_t).*sin(7*x2_t))+bias;
%y_test=4*x1_t.*x2_t;
y_test=reshape(y_test,1,numel(x1_t));

% y_test_r=reshape(y_test,size(x1_t));
% y_r=reshape(approximation_aim,size(x1));
% e_r=reshape(error_test1,size(x1_t));
% 
% figure
% plot3(x1_t,x2_t,y_test_r);
% figure
% plot3(x1,x2,y_r);
% hold on 
% plot3(x1_t,x2_t,e_r);

% test for 1
for i=1:length(x_t)    
        S(:,i,1)=RBF(x_t(:,i),hidden_node, variance,number_node);    
        for k=1:number_bias_1
        S1(:,i,k)=S(:,i,1)+test_bias_1(k);
        error_test1(i,k)=y_test(i)-W1(:,k)'*S1(:,i,k);
        end
end

for k=1:number_bias_1
       norm_test1(k)=sqrt(sum(error_test1(:,k).^2)/length(x_t));
end

for k=1:number_bias_1
       norm_error1(k)=sqrt(sum(error1(:,k).^2)/length(x_t));
end

% test for 2
for i=1:length(x_t)    
        S(:,i,1)=RBF(x_t(i),hidden_node, variance,number_node);    
        for k=1:number_bias_2
        S2(:,i,k)=[S(:,i,1);test_bias_2(k)];
        error_test2(i,k)=y_test(i)-W2(:,k)'*S2(:,i,k);        
        end
end

for k=1:number_bias_2
       norm_test2(k)=sqrt(sum(error_test2(:,k).^2)/length(x_t));
end

% figure
% plot3(x1,x2,reshape(e1(:,1),26,26))



% figure
% e1=reshape(e1,[size(x1),number_bias_1]);
% e2=reshape(e2,[size(x1),number_bias_2]);
% plot3(x1,x2,e1);
% hold on 
% plot3(x1,x2,e2);
% legend
% 
% figure
% error1=reshape(error1,[size(x1),number_bias_1]);
% error2=reshape(error2,[size(x1),number_bias_2]);
% plot3(x1,x2,error1);
% hold on 
% plot3(x1,x2,error2);
% legend

% figure
% error_test1=reshape(error_test1,[size(x1),number_bias_1]);
% error_test2=reshape(error_test2,[size(x1),number_bias_2]);
% plot3(x1_t,x2_t,error_test1);
% hold on 
% plot3(x1_t,x2_t,error_test2);
% legend
% 

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