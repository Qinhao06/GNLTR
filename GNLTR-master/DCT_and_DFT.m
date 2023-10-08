X = ['Beach-1','Beach-1','Beach-2','Beach-2','Beach-3','Beach-3','Airport-4','Airport-4','Average','Average' ];
% data1 = [94.61;96.91;95.76;99.37;96.66];
% data2 = [94.67;97.21;95.66; 99.53;96.89];

% data = [94.61, 94.67; 96.91,97.21;95.76,95.66;99.37,99.53;96.66,96.89];


% data1 = [99.36;
% 99.83;
% 99.23;
% 98.66;
% 98.37;
% 99.09;
% 
% ];
% data2 = [99.49;
% 99.83;
% 99.20;
% 98.89;
% 98.40;
% 99.16;
% 
% ];
% 
% data = [
% 99.36, 99.49;
% 99.83,99.83;
% 99.23,99.20;
% 98.66,98.89;
% 98.37,98.40;
% 99.09,99.16;
% 
% ];


data1 = [96.95;99.23; 99.63; 99.58];
data2 = [96.74;97.86; 99.58; 98.28];
data = [96.95,96.74;99.23,97.86;99.63, 99.58; 99.58, 98.28 ];

% data1 = [0.9695;0.9923; 0.9963; 0.9958];
% data2 = [0.9500;0.9786; 0.9958; 0.9828];
% data = [0.9695,0.9500;0.9923,0.9786;0.9963, 0.9958; 0.9958, 0.9828 ];
% data1 = [99.37;98.84;99.83;99.40;99.57];
% data2 = [99.53;98.99;99.83;99.53;99.56];
% data = [99.37, 99.53;
%         98.84,98.99;
%         99.83,99.83;
%         99.40,99.53;
%         99.57,99.56;
% 
% ];

% data1 = [2.075 ;
% 12.618 ;
% 9.401 ;
% 2.611;
% 2.145;
% ];
% data2 = [
% 1.870;
% 11.566;
% 8.985;
% 2.472;
% 2.087;
% ];
% data = [2.075, 1.870 ;
% 12.618 , 11.566;
% 9.401 , 8.985;
% 2.611, 2.472;
% 2.145, 2.087;
% ];






h = bar(data,'FaceColor','flat','LineWidth',1.25, 'ShowBaseLine', 'off');
% set(h(1),'FaceColor','r','BarWidth');
set(h(1),'facecolor',[.01, .72, .77]);
set(h(2),'facecolor',[.99, .49, .00]);
set(h, 'edgecolor', [0.5,0.5,0.5]);
box off  
% set(gca,'Visible','off');
% set(h(1),'facecolor',[0.1, 0.1 ,0.1]);
% for k = 1:size(data,2)
%     b(k).CData = k;
% end
% h=bar(data1,data2,1,'stack')
% set(h(1),'FaceColor',color1);
% set(h(2),'FaceColor',color2);
% set(h(3),'FaceColor',color3); 
% set(h(4),'FaceColor',color4);
% set(h(5),'FaceColor',color5);

for i=1:length(data1)
    num=num2str(data1(i), '%.2f');
    num1=[num];			% 转换成字符串的74.95和百分号合并，形成字符串的74.95%
    text(i,data1(i),num1,'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',9);
%     //% data1(i)+0.5，数字显示的位置在data1(i)上+0.5的位置
%     //% right，和下面data2(i)中的left，是显示的数字相对于两个柱的中间位置的偏移，可以自己改变一下试试看效果
%     //% 字号设置为15
end
for i=1:length(data2)
    num=num2str(data2(i));
    num1=[num];
    text(i,data2(i),num1,'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',9);
end
legend('Nonconvex Dictionary', 'HSI Dictionary ')
x_name={
    'Beach-2',...
    'Urban-3',...
    'HYDICE-Urban',...
    'San Diego',...
    }; 			% 新的横坐标要显示的值
ylabel('AUC(%)');
xt = get(gca,'XTick');	% 获取横坐标轴刻度句柄		% xt为[1 2 3 4]
yt = get(gca,'YTick');	% 获取纵坐标轴刻度句柄 	% yt为[0~100]的间隔为10的11个值，[0 10 20 30 40 50 60 70 80 90 100]
yt=[yt,110];			% 考虑后面legend图标会挡住柱状图上的数字，所以将纵轴加长一个格，到110
% yt = [90,100,110];
xtextp=xt;                    
ytextp=-0.2*yt(3)*ones(1,length(xt)) + 0.5 + 96.3;			% -0.2 为柱状图上的数字距离柱状图顶的距离，可调
% //% 设置显示标签的位置，写法不唯一，这里其实是在为每个标签找放置位置的纵坐标
% //% rotation，正的旋转角度代表逆时针旋转，旋转轴可以由HorizontalAlignment属性来设定，
% //% 有3个属性值：left，right，center，这里可以改这三个值，以及rotation后的角度，这里写的是-5
% //% 不同的角度对应不同的旋转位置了，依自己的需求而定了。
ylim([93, 100]);
text(xtextp-0.1,ytextp,x_name,'HorizontalAlignment','left','rotation',-15,'FontSize',11)  % rotation 旋转角度设置为-5，可使横坐标值斜着-5度的角度显示
% text(0.5,86,x_name,'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',15);
set(gca,'XTickLabel',[]);  % 将原坐标(1,2,3,..)去掉，可以试一下，就看出效果了


% xticklabels({'Airport-1','Airport-2','Airport-3','Airport-4','Average'})
% xtickangle(45)
% ytickangle(90)

