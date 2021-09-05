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


f_b1(1,:)= [0.43739, 0.43739,0.43739];
f_b2(1,:)= [0.4994, 0.4601,0.4530];
f_b3(1,:)= [0.6849, 0.5257,0.4836];

f_b1(2,:)= [0.0206, 0.0206, 0.0206];
f_b2(2,:)= [0.03703, 0.0267,0.02397];
f_b3(2,:)= [0.06292, 0.03662,0.02817];

f_b1(3,:)= [0.01041, 0.01041, 0.01041];
f_b2(3,:)= [0.01560, 0.01381,0.01297];
f_b3(3,:)= [0.02347, 0.01844,0.01588];

f_b1(4,:)= [0.1025, 0.1025, 0.1025];
f_b2(4,:)= [0.2645, 0.1565,0.1180];
f_b3(4,:)= [0.4202, 0.2241,0.1428];

f_b1(5,:)= [0.03484,0.03484, 0.03484];
f_b2(5,:)= [0.24, 0.1309, 0.07968];
f_b3(5,:)= [0.4124, 0.2234,0.1359];

f_b1(6,:)= [0.01463,0.01463,0.01463];
f_b2(6,:)= [0.07655, 0.03254, 0.03085];
f_b3(6,:)= [0.12045, 0.04852,0.04558];


f_b1(7,:)= [0.07933, 0.07933, 0.07933];
f_b2(7,:)= [0.1104,   0.09155, 0.09021];
f_b3(7,:)= [0.1401,  0.09700, 0.09530];

x=[1,2,3];
y=f_b1;
label_x="three kinds of scheme";
label_y="MSE value";
legend_y = ["f_{11}", "f_{21}", "f_{31}", "f_{41}", "f_{51}", "f_{61}", "f_{71}"]
ylabel_position=[0.8,0.25];
plot_line(x,y,label_x,label_y,legend_y,ylabel_position)
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\approximation_0',...
   '-depsc','-r600')

y=f_b2;
legend_y = ["f_{12}", "f_{22}", "f_{32}", "f_{42}", "f_{52}", "f_{62}", "f_{72}"]
ylabel_position=[0.8,0.25];
plot_line(x,y,label_x,label_y,legend_y,ylabel_position)
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\approximation_1',...
   '-depsc','-r600')

y=f_b3;
legend_y = ["f_{13}", "f_{23}", "f_{33}", "f_{43}", "f_{53}", "f_{63}", "f_{73}"]
ylabel_position=[0.8,0.35];
plot_line(x,y,label_x,label_y,legend_y,ylabel_position)
print('D:\GE\robot control\04-Adaptive Bias RBF Neural Network Control for ELSs\approximation_2',...
   '-depsc','-r600')












