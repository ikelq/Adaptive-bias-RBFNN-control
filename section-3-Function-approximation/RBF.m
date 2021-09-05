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

function [ S ] = RBF( Z, Mu,variance,node )
%RBF Summary of this function goes here
%   Detailed explanation goes here
S=zeros(node,1);
for i =1:node
    S(i,1)=exp(-(Z-Mu(:,i))'*(Z-Mu(:,i))/variance^2);
end

