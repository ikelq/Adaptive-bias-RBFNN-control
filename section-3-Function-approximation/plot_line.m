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

function plot_line(x,y,label_x,label_y,legend_y,ylabel_position)
figure
%set(gcf,'DefaultTextInterpreter','latex' );
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[100 100 520 440]);
set(gcf,'PaperPositionMode','auto')

plot(x,y,'-o','linewidth',1)
%xlabel(label_x);
ylabel(label_y,'position',ylabel_position);
legend(legend_y)
xticks([1,2,3])
%,'interpreter','latex'

set(gcf,'PaperPositionMode','auto');
set(gca,'XTickLabel',{'b_0','b_l','b_g'});
set (gca,'position',[0.12,0.12,0.8,0.8],'fontsize', 14,'linewidth',0.5) 
end
