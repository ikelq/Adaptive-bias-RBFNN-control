%% full state feedback based on the classic structure proposed by Slotine and Li
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

%% gamma=10*eye(Node); Variance=4; % for RBF ,variance =width^2   K1=0.01*diag([16  8]);K2=0.01*diag([16  8]);

%% 50 0.1
%% RBF neural network
Node=3^6;   % number of nodes
sigma1=0.0000; % for updating weight
sigma2=0.0000;
Variance=1; % for RBF ,variance =width^2
%bg=[0.4 0.1];
bg=[1.5 0.08]
%bg=[1.5 0.1]

%bg=[0 0];
% 2 8  0.2   (12 6 )
gamma1=8; % for updating Weight
gamma2=4; % for updating Weight
% Mu=2*rand(6,Node)-1;
k=1;
for i1=-1:1:1   % i1=-1 or 1
    for i2=-1:1:1
        for i3=-1:1:1
            for i4=-1:1:1
                for i5=-1:1:1
                    for i6=-1:1:1                      
                                 Mu(:,k)=[i1;i2;i3;i4;i5;i6];
                                 k=k+1;
                    end
                end
            end
        end
    end
end
% Mu=Mu.*[1 1 0.2 0.2 0.2 0.2]';
W=zeros(Node+1,2);
WW(:,:,1)=W;

% W1_crt=zeros(Node,1);
% W2_crt=zeros(Node,1);

%% parameter of dynamics
m1 = 2;  %unit is 'kg' %The model is referred "Aptive Nerual Network Contol..." written by Sam Ge on page57 
m2 = 0.85;

l1 = 0.35;  %unit is 'm'
l2 = 0.31;  % li is the length of ith link i
lc1 = 1/2 * l1; % lci the center of mass to joint of ith link
lc2 = 1/2 * l2;

I1 = 1/4*m1*l1^2;%1825795.31e-09 ;  % moment of inertial
I2 = 1/4*m2*l2^2;%26213426.68e-09 ;

g = 9.81;

p(1) = m1 * lc1.^2 + m2 * l1^2 + I1;
p(2) = m2 * lc2.^2 + I2;
p(3) = m2 * l1 * lc2;
p(4) = m1 * lc2 + m2 * l1;
p(5) = m2 * lc2;

%% begin simulation 
T=200;
size=0.001;
t=0:size:T;
n=length(t);
% 16 8
K1=1*diag([10  5]);
K2=0.5*diag([10  5]);
% K1=2*diag([9  3]);
% K2=1*diag([9  3]);
% K1=2*diag([5  5]);
% K2=diag([5  5]);
%initial state
i=1;
x(1)=0;x(2)=0;x(3)=0;x(4)=0; 
e1=[0;0];
de1=[0;0];
dw=zeros(Node+1,2);
dwf=dw;
q=zeros(2,length(t));
%q(:,1)=[x(1);x(2)];
dq=zeros(2,length(t));
ddq=zeros(2,length(t));
e=zeros(2,length(t));
de=zeros(2,length(t));
Tau=zeros(2,length(t));
normW=zeros(2,length(t));
Td=[0,0]';
% qr=[sin(t);sin(t)];
% dqr=[cos(t);cos(t)];
% ddqr=-[sin(t);sin(t)];
qr=    [sin(t);cos(t)];
dqr=  [cos(t);-sin(t)];
ddqr=[-sin(t);-cos(t)];

% qr=[sin(0.5*t);sin(0.5*t)];
% dqr=[0.5*cos(0.5*t);0.5*cos(0.5*t)];
% ddqr=-0.25*[sin(0.5*t);0.25*sin(0.5*t)];
% parameter for integral
% K3=5*diag([2  1]);
% Ir=[0;0];
normS(1)=0;
Norm_W1(1) = 0
Norm_W2(1) = 0
%% the first step
for i=2:n

M=[p(1)+p(2)+2*p(3)*cos(x(3)) p(2)+p(3)*cos(x(3));
    p(2)+p(3)*cos(x(3)) p(2)];
C=[-p(3)*x(4)*sin(x(3)) -p(3)*(x(2)+x(4))*sin(x(3));
    p(3)*x(2)*sin(x(3))  0];
G=[p(4)*g*cos(x(1)) + p(5)*g*cos(x(1)+x(3)); p(5)*g*cos(x(1)+x(3))];
J=[-l1*sin(x(1))+l2*sin(x(1)+x(3))  -l2*sin(x(1)+x(3));  l1*cos(x(1))+l2*cos(x(1)+x(3))  l2*cos(x(1)+x(3)) ];

e1=[qr(1,i-1)-x(1);qr(2,i-1)-x(2)];
de1=[dqr(1,i-1)-x(3);dqr(2,i-1)-x(4)];
r=de1+K2*e1;
%  te1=0.3*tanh(K2*e1);
%  tde1=0.3*tanh(K2*de1);
% te1=min(max(K2*e1,-0.1),0.1);
% tde1=min(max(K2*de1,-0.1),0.1);
%r=K2*e1;
    %Z=[sin(q(:,i-1));(dqr(:,i)+K2*e1)*0.2;(ddqr(:,i)+K2*de1)*0.2];
    Z=[q(:,i-1);(dqr(:,i)+K2*e1);(ddqr(:,i)+K2*de1)];
    %Z=[sin(q(:,i-1));(dqr(:,i)+te1);(ddqr(:,i)+tde1)];
    %Z=min(max(Z,-1.2),1.2);
  % Z=1.1*tanh(Z);
    %Z=[q(:,i-1);(dqr(:,i));ddqr(:,i)];
    %function [ S ] = RBF( Z, Mu,variance,node )     %prototype of function RBF
    %b=min(max(50*abs(e1)),0.1) ;
    
    Sm=RBF(Z,Mu,Variance,Node ) ;                     % RBF method is used in calculating S     
    S=[Sm , Sm;  bg];
    
    normS(i)=norm(S);
    dw(:,1)=gamma1*S(:,1)*r(1);              % updating law as stated
    dw(:,2)=gamma2*S(:,2)*r(2);
    if Norm_W1(i-1) >10 || Norm_W2(i-1) >10
    dw(:,1)=gamma1*S(:,1)*r(1)    - 0.001*  W;              % updating law as stated
    dw(:,2)=gamma2*S(:,2)*r(2) - 0.001*  W;
    end
    
    %dw(:,1)=gamma1*(S*de1(1)+sigma1*W(:,1));              % updating law as stated
    %dw(:,2)=gamma2*(S*de1(2)+sigma2*W(:,2));
e(:,i-1)=e1;
de(:,i-1)=de1;

% Td=J'*[0,20]';
if i>100001
    Td=J'*[0,8]';
end
% de(:,i-1)=de1;
%dW(1)=-gamma*(S*r(1)+sigma1*W1_crt);
% the next step
%Tau(:,i)=K1*r+M*(ddqr(:,i)+K2*de1)+C*(dqr(:,i)+K2*e1)+G;
Ta=[W(:,1)'*S(:,1); W(:,2)'*S(:,2)];


% Calculator C(qr,dqr)
dqr1=dqr(:,i)+K2*e1;
x(3)=dqr1(1);
x(4)=dqr1(2);
C1=[-p(3)*x(4)*sin(x(3)) -p(3)*(x(2)+x(4))*sin(x(3));
    p(3)*x(2)*sin(x(3))  0];


e_RBF=M*(ddqr(:,i)+K2*de1)+C*(dqr(:,i)+K2*e1)+G+Td-Ta;
%e_RBF=M*(ddqr(:,i))+C*(dqr(:,i))+G-Ta;
ee_RBF(:,i)=e_RBF;
TTa(:,i)=Ta;
%for interal 
% Ir=Ir+r*size;
% end
Tau(:,i)=K1*r+Ta+ 0*randn(2,1) ;
ddq(:,i)=M\(Tau(:,i)-Td-C*dq(:,i-1)-G);
dq(:,i)=dq(:,i-1)+size*ddq(:,i-1);
q(:,i)=q(:,i-1)+size*dq(:,i-1)+1/2*size^2*ddq(:,i-1);

x(1)=q(1,i);
x(2)=q(2,i);
x(3)=dq(1,i);
x(4)=dq(2,i);

    W(:,1)=dw(:,1)*size+W(:,1);                     % Weights for next iteration
    W(:,2)=dw(:,2)*size+W(:,2);
    Norm_W1(i)=sqrt(W(:,1)'*W(:,1));                    % Norm W1 & W2
    Norm_W2(i)=sqrt(W(:,2)'*W(:,2));
    
    
    WW(:,:,i)=W;
end
e_q=qr-q;

label_y=["Trajectory  [rad]","Tracking error [rad]"];
legend_y=["q_1","q_2","error_1","error_2"];
plot_2line(t,q',e_q','t [s]',label_y,legend_y,[-14,0;-14,0.4])

plot_local_detial ([0.2 0.2 0.2 0.2], t,e_q',[90 100])
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\e_q_b_g'...
    ,'-depsc','-r600')

label_y=["Control torque [N\cdotm]","Control torque [N\cdotm]"];
legend_y=["\tau_1","q_2","\tau_2","error_2"];
plot_2line(t,Tau(1,:),Tau(2,:),'t [s]',label_y,legend_y,[-14,7;-14,7])

print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\T_b_g'...
    ,'-depsc','-r600')


label_y=["Output of RBFNNs [N\cdotm]","Errors of RBFNNs [N\cdotm]"];
legend_y=["RBF_1","RBF_2","error_1","error_2"];
plot_2line(t,TTa',ee_RBF','t [s]',label_y,legend_y,[-14,5;-14,4])
plot_local_detial ([0.2 0.25 0.2 0.2], t,ee_RBF',[90 100])
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\error_b_g',...
    '-depsc','-r600')



last_10_seconds_e_q=e_q(:,100000-10000:100000);
mean_e_q_last_10=mean (( last_10_seconds_e_q').^2)
max_eq_10= max(last_10_seconds_e_q')
last_10_seconds_ee_RBF=ee_RBF(:,100000-10000:100000);
mean_ee_RBF_last_10=mean (( last_10_seconds_ee_RBF).^2,2)
max_eRBF_10= max(last_10_seconds_ee_RBF')


last_20_seconds_e_q=e_q(:,200000-10000:200000);
mean_e_q_last_20=mean (( last_20_seconds_e_q').^2)
max_eq_20= max(last_20_seconds_e_q')
last_20_seconds_ee_RBF=ee_RBF(:,200000-10000:200000);
mean_ee_RBF_last_20=mean (( last_20_seconds_ee_RBF).^2,2)
max_eRBF_20= max(last_20_seconds_ee_RBF')
 
 
 
 figure
 plot(t,normS)
 set (gca,'position',[0.1,0.1,0.8,0.8] );
 legend('Norm S')
 print('Norm_s_g','-depsc')
% figure
% plot(t,ddq(1,:),t,ddq(2,:))
% xlabel('t [s]'); ylabel('accleration');
% figure
% plot(t,dq(1,:),t,dq(2,:))
% xlabel('t [s]'); ylabel('velocity');
% figure; 
% subplot(2,1,1);
% plot(t,q(1,:)',t,qr(1,:)');xlabel('t [s]'); ylabel('q1 and qd1');
% title('Model based Control with the Full State Feedback');
% subplot(2,1,2);
% plot(t,e_q(1,:)');xlabel('t [s]'); ylabel('error e1');
% 
% 
% figure; 
% subplot(2,1,1);
% plot(t,q(2,:)',t,qr(2,:));xlabel('t [s]'); ylabel('q2 and qd2');
% title('Model based Control with the Full State Feedback');
% subplot(2,1,2);
% plot(t,e_q(2,:)');xlabel('t [s]'); ylabel('error e2');
% 
% 
% figure;
% plot(t,Tau(1,:)',t,Tau(2,:)');title('Adaptive Neural Netwok Control with the Full State Feedback');
% legend('\tau_1','\tau_2'); xlabel('t [s]'); ylabel('Control inputs');

 
 WW1=WW(700:730,1,:);
 WW2=WW(700:730,2,:);
 WW11=WW1(:,:);
 WW22=WW2(:,:);
 


 
label_y=["Evolving W_{700-730}";"Norm W"];
legend_y=[""];
plot_2line(t,WW11',Norm_W1','t [s]',label_y,legend_y,[-13,3;-13,4]);
ylabel("\textbf{Norm $\hat{W}$}",'interpreter','latex')

ylabel("\textbf{Evolving $\hat{W}_{700-730}$}",'interpreter','latex')
annotation('arrow',[0.4 0.3],[0.83 0.8]);
text(75,4,'$\hat{W}_{bg}$','interpreter','latex');
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\Norm_W1_bg',...
    '-depsc', '-r600')

% the same as the last section codes
plot_line(t,WW11','t [s]',"  ",legend_y,[-20,3])
ylabel("\textbf{Evolving $\hat{W}_{700-730}$}",'interpreter','latex','fontsize', 14)
annotation('arrow',[0.4 0.3],[0.70 0.65]);
text(75,4,'$\hat{W}_{bg}$','interpreter','latex','fontsize', 14);
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\Norm_W1_bg',...
    '-depsc', '-r600')



label_y=["\textbf{Evolving $\hat{W}_{700-730}$}";"Norm W"];
legend_y=[""];
plot_2line(t,WW22',Norm_W2','t [s]',label_y,legend_y,[-13,0.1;-13,2]);
ylabel("\textbf{Norm $\hat{W}$}",'interpreter','latex')

ylabel("\textbf{Evolving $\hat{W}_{700-730}$}",'interpreter','latex')
annotation('arrow',[0.35 0.25],[0.8 0.76]);
text(65,0.15,'$\hat{W}_{bg}$','interpreter','latex');
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\Norm_W2_bg', ...
    '-depsc', '-r600')


plot_line(t,WW22','t [s]'," a ",legend_y,[-20,0.1])
ylabel("\textbf{Evolving $\hat{W}_{700-730}$}",'interpreter','latex','fontsize', 14)
annotation('arrow',[0.35 0.25],[0.61 0.55]);
text(65,0.15,'$\hat{W}_{bg}$','interpreter','latex','fontsize', 14);
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\Norm_W2_bg', ...
    '-depsc', '-r600') 
%  figure;
% subplot(2,1,1)
% set (gca,'position',[0.1,0.55,0.8,0.4] )
% plot(t,Norm_W1')
% xlabel('t [s]'); ylabel('Norm W1','Position',[-13,5])
% subplot(2,1,2)
% set (gca,'position',[0.1,0.1,0.8,0.4] )
% plot(t,Norm_W2');
% xlabel('t [s]');ylabel('Norm W2','Position',[-13,2]) 
%  print('Norm_W_b_g','-depsc')

 

% figure;
% plot(linspace(0,Time,STeps),eps');
% title('Adaptive Neural Netwok Control with the Full State Feedback');
% xlabel('t [s]'); ylabel('Approximation errors');% approximation error btw Neural network and the model
% 
% figure;plot(tout,eout); title('Adaptive Neural Netwok Control with the Full State Feedback');
% xlabel('t [s]'); ylabel('Norm of errors ||z_1||');   % Norm Errors
 save("RBF_global_bias")

