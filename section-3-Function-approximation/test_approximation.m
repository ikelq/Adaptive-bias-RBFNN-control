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
x=-2:0.01:2;
lengthx=ones(length(x),1);

variance=0.5;
hidden_node=-2:0.5:2;
hidden_length=length(hidden_node);
node=length(hidden_node);

approximation_aim=0*lengthx+10*sin(3*x')+10*x'+0*(x.^2)';
for i=1:length(x)
    S(i,:)=RBF(x(i),hidden_node, variance,node);
end
W(:,1)= S\approximation_aim;
W(:,2)=(S+0.2)\approximation_aim;
W(:,3)=(S+0.4)\approximation_aim;
W(:,4)=(S+0.6)\approximation_aim;
W(:,5)=(S+10)\approximation_aim;
Wb1  =[S,0.1*lengthx]\approximation_aim;
Wb2 =[S,1*lengthx]\approximation_aim;


y(:,1)= S*W(:,1);
y(:,2)=(S+0.2)*W(:,2);
y(:,3)=(S+0.4)*W(:,3);
y(:,4)=(S+0.6)*W(:,4);
y(:,5)=(S+10)*W(:,5);
y(:,6)=[S,0.1*lengthx]*Wb1;
y(:,7)=[S,1*lengthx]*Wb2;

error(:,1)=approximation_aim-y(:,1);
error(:,2)=approximation_aim-y(:,2);
error(:,3)=approximation_aim-y(:,3);
error(:,4)=approximation_aim-y(:,4);
error(:,5)=approximation_aim-y(:,5);
error(:,6)=approximation_aim-y(:,6);
error(:,7)=approximation_aim-y(:,7);

for i=1:7
    Norm(i)=0.01*norm(error(:,i),1);
end
t=1:7;
figure 
plot(t,Norm)
% hold on 
% plot(x,approximation_aim,'y')
% plot(x,y)
% hold on
figure
plot(x,error(:,1),'r',x,error(:,2),'b',x,error(:,3),'g',x,error(:,4),'c',x,error(:,5),'m',x,error(:,6),'y',x,error(:,7),'k')
legend
 figure
 plot(x,S(:,1),x,S(:,2),x,S(:,3),x,S(:,4),x,S(:,5),x,S(:,6),x,S(:,7),x,S(:,8),x,S(:,9))
 hold on
  plot(x,approximation_aim,'r')




