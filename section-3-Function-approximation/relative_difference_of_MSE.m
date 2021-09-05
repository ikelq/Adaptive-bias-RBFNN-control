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
f=zeros(7,3,3);
f(1,1,:)=[0.43739   0.43808  0.43755];
f(1,2,:)=[0.4994     0.46018  0.4530];
f(1,3,:)=[0.6849     0.5257    0.4836];
f(2,1,:)=[0.02060   0.02067  0.02062];
f(2,2,:)=[0.03703   0.0267    0.02397];
f(2,3,:)=[0.06292   0.03662  0.02817];
f(3,1,:)=[0.01041   0.01049  0.01043];
f(3,2,:)=[0.01560   0.01381  0.01279];
f(3,3,:)=[0.02347   0.01844  0.01588];
f(4,1,:)=[0.1025     0.1026    0.1025 ];
f(4,2,:)=[0.2645     0.1565    0.1180];
f(4,3,:)=[0.4202     0.2241    0.1428];
f(5,1,:)=[0.03484   0.03486  0.03484];
f(5,2,:)=[0.2400     0.1309    0.07968];
f(5,3,:)=[0.4124     0.2234    0.1359];
f(6,1,:)=[0.01463   0.01477  0.01464];
f(6,2,:)=[0.07655   0.03254  0.03085];
f(6,3,:)=[0.12045   0.04852  0.04558];
f(7,1,:)=[0.07933   0.08286  0.07957];
f(7,2,:)=[0.1104     0.09155  0.09021];
f(7,3,:)=[0.1401     0.09700   0.09530];


d=zeros(7,3,3);
%d(:,:,1)=1;
for i=1:7
    for j=1:3
        for k=2:3
        d(i,j,k)=(f(i,j,k)-f(i,j,1))/f(i,j,1)
        end
        dd=d(i,j,:);
        plot(dd(:));
        hold on
    end
end
legend
a=['a','b','c'];
for j=1:3
    figure
    for i=1:7
        dd=d(i,j,:);
        plot(dd(:));
        hold on
    end
    legend
end
    

    
        

