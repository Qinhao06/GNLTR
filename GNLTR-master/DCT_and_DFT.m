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
    num1=[num];			% ת�����ַ�����74.95�Ͱٷֺźϲ����γ��ַ�����74.95%
    text(i,data1(i),num1,'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',9);
%     //% data1(i)+0.5��������ʾ��λ����data1(i)��+0.5��λ��
%     //% right��������data2(i)�е�left������ʾ��������������������м�λ�õ�ƫ�ƣ������Լ��ı�һ�����Կ�Ч��
%     //% �ֺ�����Ϊ15
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
    }; 			% �µĺ�����Ҫ��ʾ��ֵ
ylabel('AUC(%)');
xt = get(gca,'XTick');	% ��ȡ��������̶Ⱦ��		% xtΪ[1 2 3 4]
yt = get(gca,'YTick');	% ��ȡ��������̶Ⱦ�� 	% ytΪ[0~100]�ļ��Ϊ10��11��ֵ��[0 10 20 30 40 50 60 70 80 90 100]
yt=[yt,110];			% ���Ǻ���legendͼ��ᵲס��״ͼ�ϵ����֣����Խ�����ӳ�һ���񣬵�110
% yt = [90,100,110];
xtextp=xt;                    
ytextp=-0.2*yt(3)*ones(1,length(xt)) + 0.5 + 96.3;			% -0.2 Ϊ��״ͼ�ϵ����־�����״ͼ���ľ��룬�ɵ�
% //% ������ʾ��ǩ��λ�ã�д����Ψһ��������ʵ����Ϊÿ����ǩ�ҷ���λ�õ�������
% //% rotation��������ת�Ƕȴ�����ʱ����ת����ת�������HorizontalAlignment�������趨��
% //% ��3������ֵ��left��right��center��������Ը�������ֵ���Լ�rotation��ĽǶȣ�����д����-5
% //% ��ͬ�ĽǶȶ�Ӧ��ͬ����תλ���ˣ����Լ�����������ˡ�
ylim([93, 100]);
text(xtextp-0.1,ytextp,x_name,'HorizontalAlignment','left','rotation',-15,'FontSize',11)  % rotation ��ת�Ƕ�����Ϊ-5����ʹ������ֵб��-5�ȵĽǶ���ʾ
% text(0.5,86,x_name,'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',15);
set(gca,'XTickLabel',[]);  % ��ԭ����(1,2,3,..)ȥ����������һ�£��Ϳ���Ч����


% xticklabels({'Airport-1','Airport-2','Airport-3','Airport-4','Average'})
% xtickangle(45)
% ytickangle(90)

