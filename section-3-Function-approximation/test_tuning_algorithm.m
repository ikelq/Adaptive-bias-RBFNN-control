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
approximation_aim=-10*constans-10*sin(2*x')+0*x'+0*(x.^2)';
function_dimension=1;
%parameter for train
gamma=0.1;  %learning rate
variance=0.3; 
hidden_node=-x_min-0.25:0.5:x_max+0.7;
number_node=length(hidden_node);
train_loop_number=1000;



% test bias on time
test_bias=[0,0.1,0.4,0.8,1.6];
number_bias=length(test_bias);

% in term of bias is add for each activation
W=zeros(number_node,function_dimension,number_bias);  % function_i =W(:,i,j)'*S(Z);
% in term of bias is add as another activation which is can be seen as PID+RBF
W=zeros(number_node+1,function_dimension,number_bias);  


W1=zeros(node,1);
W2=W1;
W3=W1;
Wn1=[W1;0];
Wn2=[W1;0];
for j=1:train_loop_number
for i=1:length(x)
    S1(:,i)=RBF(x(i),hidden_node, variance,node);
    S2(:,i)=S1(:,i)+0.5;
    e1(i)=approximation_aim(i)-W1'*S1(:,i);
    e2(i)=approximation_aim(i)-W2'*S2(:,i);
    dotW1=gamma*S1(:,i)*e1(i);
    dotW2=gamma*S2(:,i)*e2(i);
    W1=W1+step*dotW1;
    W2=W2+step*dotW2;
    
    S3(:,i)=S1(:,i)+0.6;
    e3(i)=approximation_aim(i)-W3'*S3(:,i);
    dotW3=gamma*S3(:,i)*e3(i);
    W3=W3+step*dotW3;

    Sn1(:,i)=[S1(:,i);2];
    en1(i)=approximation_aim(i)-Wn1'*Sn1(:,i);
    dotWn1=gamma*Sn1(:,i)*en1(i);
    Wn1=Wn1+step*dotWn1;
    
    Sn2(:,i)=[S1(:,i);4];
    en2(i)=approximation_aim(i)-Wn2'*Sn2(:,i);
    dotWn2=gamma*Sn2(:,i)*en2(i);
    Wn2=Wn2+step*dotWn2;
    
end
end
plot(x,e1,'r',x,e2,'g',x,e3,'b',x,en1,'c',x,en2,'m')
legend;
norme=[norm(e1),norm(e2),norm(e3),norm(en1),norm(en2)]
figure
error1=approximation_aim'-W1'*S1;
error2=approximation_aim'-W2'*S2;
error3=approximation_aim'-W3'*S3;

error_n1=approximation_aim'-Wn1'*Sn1;
error_n2=approximation_aim'-Wn2'*Sn2;


plot(x,error1,'r',x,error2,'g',x,error3,'b',x,error_n1,'c',x,error_n2,'m')
legend

normerror=[norm(error1),norm(error2),norm(error3),norm(error_n1),norm(error_n2)]

% plot(approximation_aim)