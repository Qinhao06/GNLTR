clear all;
close all;
clc;
addpath(genpath('./inexact_alm_rpca'));
addpath(genpath('./ConstructW'));
addpath(genpath('./tensor_toolbox_2.5'));
addpath(genpath('./tSVD'));
addpath(genpath('./Datasets'));
addpath(genpath('./proximal_operator'));
addpath(genpath('./nonconvex_funs'))
addpath(genpath('./LRENmat'))
% 
% load Sandiego_new
% load Sandiego_gt
% load HYDICE_urban.mat
load abu-beach-2.mat
% load abu-urban-3.mat
% data=hsi;
% mask=hsi_gt;
mask = map;
[rows,cols] = size(mask);

f_show=data(:,:,[37,18,8]);
for i=1:3
    max_f=max(max(f_show(:,:,i)));
    min_f=min(min(f_show(:,:,i)));
    f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
end
% figure,imshow(f_show);imwrite(f_show,'im.jpg');
% figure,imshow(mask,[]);imwrite(mask,'gt.jpg');
DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

NBOXPLOT=zeros(H*W,10);

%%%
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';
%%%

%%%%%%%GTVLRR
tic;
Dict=ConstructionD_lilu(Y,15,20);
lambda = 0.6;
beta = 0.2;
gamma =0.05;% 0;%
display = true;
[X,S] = lrr_tv_manifold(Y,Dict,lambda,beta,gamma,[H,W],display);
toc
disp('GTVLRR');
r_gtvlrr=sqrt(sum(S.^2));
r_max = max(r_gtvlrr(:));
r_min = min(r_gtvlrr(:));
R_gtvlrr = (r_gtvlrr - r_min)/(r_max - r_min);



% %%%%%%LRASR

beta=0.1;
lamda=0.001;
tic;
[S,E]=LRASR(Y,Dict,beta,lamda,1);
toc
disp('LRSAR');
r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
r_min = min(r_new(:));
R_lrasr = (r_new - r_min) / (r_max - r_min);






% %%%%%%
% tic;
% [T_all,Anomaly]=AD_lilu5(DataTest);
% toc
% Y0=reshape(Anomaly, num, Dim)';
% r4 = RX(Y0);  % input: num_dim x num_sam    rx
% r_max = max(r4(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r4 > tau);
%   PF4(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD4(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% f_show=reshape(r4,[H,W]);
% f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% figure('name','TDAD'), imshow(f_show);imwrite(f_show,'TDAD.jpg');
% area_TDAD = sum((PF4(1:end-1)-PF4(2:end)).*(PD4(2:end)+PD4(1:end-1))/2);
% NBOXPLOT(:,6)=f_show(:);


tic;
Anomaly=AD_lilu7(DataTest);
toc
disp('TPCA');
r5 = RX(abs(Anomaly)');  % input: num_dim x num_sam    rx
r_max = max(r5(:));
r_min = min (r5(:));
R_tpca = (r5 - r_min)/(r_max -r_min);




tic;
r3 = RX(Y);  % input: num_dim x num_sam    rx
toc
disp('RX');
r_max = max(r3(:));
r_min = min(r3(:));
R_rx = (r3 - r_min)/(r_max - r_min);

% 
% LSMAD

rank = 25; 

tic;
[L,S,RMSE,error]=GoDec(Y',rank,floor(0.0022*Dim)*9,2);
toc
disp('LSMAD');
L=L';
S=S';

mu=mean(L,2);
r_new2=(diag((Y-repmat(mu,[1,num]))'*pinv(cov(L'))*(Y-repmat(mu,[1,num]))))';

r_max = max(r_new2(:));
r_min = min(r_new2(:));
R_lsmad = (r_new2 - r_min)/(r_max - r_min);


%%%rpca   rx
tic;
[r0 ,Output_S, Output_L] = Unsupervised_RPCA_Detect_v1(DataTest);
toc
disp('RPCA');
XS = reshape(Output_S, num, Dim);
r2 = RX(XS');  % input: num_dim x num_sam

r_max = max(r2(:));  
r_min = min(r2(:));
R_rpca = (r2 - r_min)/(r_max - r_min);




%%% PCA_TLRSR
f_show=data(:,:,[37,18,8]);
for i=1:3
    max_f=max(max(f_show(:,:,i)));
    min_f=min(min(f_show(:,:,i)));
    f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
end

DataTest=data;
% numb_dimension = 6;%San Diego
% numb_dimension = 4;%airport-1  
%  numb_dimension =5;%airport-2 
%   numb_dimension =17;%airport-3
% numb_dimension = 17;%airport-4

% numb_dimension = 19;%ubran1
% numb_dimension = 21;%ubran2
%  numb_dimension =15;%ubran3 
%   numb_dimension =6;%ubran4
% numb_dimension = 10;%ubran5


% numb_dimension = 15;%beach-1  
 numb_dimension =15;%beach-2 
%   numb_dimension =12;%beach-3
% numb_dimension = 4;%beach-4

% numb_dimension = 15;%HY-urban

DataTest = PCA_img(DataTest, numb_dimension);

[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim 
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end 

mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                        
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';

X=DataTest;  
[n1,n2,n3]=size(X);

% opts.lambda =0.06;%airport-1 
% opts.lambda =0.06; %airport-2 
% opts.lambda =0.06; %airport-3
% opts.lambda =0.06; %airport-4

% opts.lambda =0.2;%beach-1 
opts.lambda =0.3; %beach-2 
% opts.lambda =0.3; %beach-3
% opts.lambda =0.03; %beach-4

% opts.lambda =0.07;%urban-1 
% opts.lambda =0.06; %urban-2 
% opts.lambda =0.2; %urban-3
% opts.lambda =0.02; %urban-4
% opts.lambda =0.08; %urban-4

% opts.lambda =0.02;%san
% opts.lambda =0.06;%Urban




opts.mu = 1e-4;
opts.tol = 1e-8;    
opts.rho = 1.5;
opts.max_iter = 100;
opts.DEBUG = 0;
% learn-dictionary
tic;
[ L,S,rank] = dictionary_learning_tlrr1( X, opts);
max_iter=100;
Debug = 0;


% lambda=0.01;%airport-1  
% lambda=0.01;%airport-2 
% lambda=0.05;%airport-3 
% lambda=0.05;%airport-4 % :)

% lambda=0.04;%beach-1  
lambda=0.05;%beach-2 
% lambda=0.05;%beach-3 
% lambda=0.0006;%beach-4 % :)

% lambda=0.2;%urban-1  
% lambda=0.03;%urban-2 
% lambda=0.03;%urban-3 
% lambda=0.001;%urban-4 % :)
% lambda=0.02;%urban-5 % :)

% lambda=0.01;%HY-Urban
% lambda=0.01;%San :)




[Z,tlrr_E,Z_rank,err_va ] = TLRSR(X,L,max_iter,lambda,Debug);
toc;
disp('PCA-TLRSR');
E=reshape(tlrr_E, num, Dim)';
r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
r_min = min(r_new(:));
R_tlrsr = (r_new - r_min)/(r_max - r_min);






% figure,
% semilogx(taus, PF10, 'b-', 'LineWidth', 2);
% % % % % %   LREN
load('abu-beach-2Energy-map-cluster.mat')
r_new = array;
[m,n] = size(r_new);
r_new = reshape(r_new,[1, m*n]);
r_max = max(r_new(:));
r_min = min(r_new(:));
R_LREN = (r_new - r_min)/(r_max - r_min);


% % % % % DECNN
run './matlab/vl_setupnn'
% load('pavia_car');
K=6; % number of clusters, pavia car
beta=0.02; % beta
lamda=0.04; % lambda

load('FFDNet_gray.mat');
% mask=map;
% map=mask;

% f_show=data(:,:,[37,18,8]);
% for i=1:3
%     max_f=max(max(f_show(:,:,i)));
%     min_f=min(min(f_show(:,:,i)));
%     f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
% end

% figure,imshow(f_show);
% figure,imshow(mask,[]);
DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

Y=reshape(DataTest, num, Dim)';
tic
E=DeCNNAD(Y,H,W,K,beta,lamda,net);
toc
r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
r_min = min(r_new(:));
R_DEC = (r_new - r_min)/(r_max - r_min);
taus = linspace(0, r_max, 5000);
PF11=zeros(1,5000);
PD11=zeros(1,5000);
% taus = linspace(0, r_max, 10000);
% PF11=zeros(1,10000);
% PD11=zeros(1,10000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new> tau);
  PF11(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD11(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_dec1 = sum((PF11(1:end-1)-PF11(2:end)).*(PD11(2:end)+PD11(1:end-1))/2)
% area_dec=ROC(r_new,map,1) % AUC
% f_anomaly=reshape(r_new,[H,W]);
% f_anomaly=(f_anomaly-min(f_anomaly(:)))/(max(f_anomaly(:))-min(f_anomaly(:)));
% figure, imshow(f_anomaly);








% %%% GNLTR


opts.mu = 1e-4;
opts.tol = 1e-8;
opts.rho = 1.1;
opts.max_iter = 100;
opts.DEBUG = 0;
max_iter=100;
Debug = 0;

% opts.lambda = 0.05; % HY
% lambda = 0.01;
% numb_dimension = 7; 
% fun = 'lp' ;        gamma = 0.5;
% fun_dict = 'lp' ;        gamma_dict = 0.5; 
% % 
% opts.lambda = 0.001; % san
% lambda = 0.05;
% numb_dimension = 6; 
% fun_dict = 'lp' ;       gamma_dict = 0.5;
% fun = 'logarithm' ; gamma = 10 ;

% 
% opts.lambda = 0.049; % air1
% lambda = 0.002;
% numb_dimension = 19; 
% 
% opts.lambda = 0.016; % air2
% lambda = 0.002;
% numb_dimension = 13;

% opts.lambda = 0.063; % air3
% lambda = 0.177;
% numb_dimension = 16; 
% 
% opts.lambda = 0.1; % air4
% lambda = 0.013;
% numb_dimension = 5; 
% 




% 
% opts.lambda = 0.03; % beach1
% lambda = 0.03;
% numb_dimension = 15;
% fun_dict = 'mcp' ; gamma_dict = 10 ;
% fun = 'lp' ;        gamma = 0.5;
% 
opts.lambda = 0.05; % beach2
lambda = 0.001;
numb_dimension = 10;
fun_dict = 'lp' ;        gamma_dict = 0.5; 
fun = 'geman' ;  gamma = 10 ;
% 
% opts.lambda = 0.05; % beach3
% lambda = 0.001;
% numb_dimension = 18; 
% fun_dict = 'scad' ;      gamma_dict = 100 ;
% fun = 'lp' ;        gamma = 0.5; 


% opts.lambda = 0.001; % beach4
% lambda = 0.03;
% numb_dimension = 9; 
% fun_dict = 'etp' ;  gamma_dict = 0.1 ;
% fun = 'geman' ;  gamma = 10 ; 
% 

% opts.lambda = 0.01; % urban1
% lambda = 0.01;
% numb_dimension = 19;
% fun_dict = 'scad' ;      gamma_dict = 100 ;
% fun = 'mcp' ; gamma = 10 ;

 
% opts.lambda = 0.005; % urban2
% lambda = 0.0005;
% numb_dimension = 20; 
% fun_dict = 'lp' ;        gamma_dict = 0.5;
% fun = 'lp' ;        gamma = 0.5; 

 
% opts.lambda = 0.03; % urban3
% lambda = 0.03;
% numb_dimension = 16; 
% fun_dict = 'scad' ;      gamma_dict = 100 ;
% fun = 'geman' ;  gamma = 10 ;

% opts.lambda = 0.01; % urban4
% lambda = 0.03;
% numb_dimension = 6; 
% fun_dict = 'lp' ;        gamma_dict = 0.5;
% fun = 'lp' ;        gamma = 0.5;  

% opts.lambda = 0.01; % urban5
% lambda = 0.0005;
% numb_dimension = 9; 
% fun_dict = 'etp' ;  gamma_dict = 0.1 ;
% fun = 'logarithm' ; gamma = 10 ;

% fun = 'lp' ;        gamma = 0.5;  % air2 san beach2 urban2 urban3
% fun = 'scad' ;      gamma = 100 ;  % beach4
% fun = 'logarithm' ; gamma = 10 ;   % HY urban1 urban5
% fun = 'mcp' ; gamma = 10 ;
% fun = 'cappedl1' ; gamma = 1000 ;
% fun = 'etp' ;  gamma = 0.1 ;
% fun = 'geman' ;  gamma = 10 ; % air4 urban4
% fun = 'laplace' ; gamma = 10 ; % air1 air3 beach1 beach3


% fun_dict = 'lp' ;        gamma_dict = 0.5;  % air2 san beach2 urban2 urban3
% fun_dict = 'scad' ;      gamma_dict = 100 ;  % beach4
% fun_dict = 'logarithm' ; gamma_dict = 10 ;   % HY urban1 urban5
% fun_dict = 'mcp' ; gamma_dict = 10 ;
% fun_dict = 'cappedl1' ; gamma_dict = 1000 ;
% fun_dict = 'etp' ;  gamma_dict = 0.1 ;
% fun_dict = 'geman' ;  gamma_dict = 10 ; % air4 urban4
% fun_dict = 'laplace' ; gamma_dict = 10 ; % air1 air3 beach1 beach3

DataTest=data;

DataTest = PCA_img(DataTest, numb_dimension);
    
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim 
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end 


%%%%
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                        
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';


X=DataTest;  
[n1,n2,n3]=size(X);

%% test R-TPCA

        tic;
    [ L,S,rank] = dictionary_construction( X, opts, fun_dict, gamma_dict);

    %% test PCA-TLRSR

 
    [Z,tlrr_E,Z_rank,err_va ] = GNLTA(X,L,max_iter,lambda,Debug, fun, gamma);

toc
disp('GNLTR');
    %% compute AUC
    E = reshape(tlrr_E, num, Dim)';
    r_new = sqrt(sum(E.^2, 1));
    r_max = max(r_new(:));
    r_min = min(r_new(:));
    R_gnltr = (r_new - r_min)/(r_max- r_min);



% label_value = reshape(hsi_gt,1,rows*cols);

label_value = reshape(mask,1,rows*cols);
R1_value = reshape(R_rx,1,rows*cols);
R2_value = reshape(R_rpca,1,rows*cols);
R3_value = reshape(R_lsmad,1,rows*cols);
R4_value = reshape(R_lrasr,1,rows*cols);
R5_value = reshape(R_tpca,1,rows*cols);
R6_value = reshape(R_gtvlrr,1,rows*cols);
R7_value = reshape(R_tlrsr,1,rows*cols);
R8_value = reshape(R_LREN,1,rows*cols);
R9_value = reshape(R_DEC,1,rows*cols);
R10_value = reshape(R_gnltr,1,rows*cols);

ind_tar = find(label_value == 1);
ind_bac = find(label_value == 0);
% targ back为列向量
num_targ = length(ind_tar);
num_back = length(ind_bac);
num_meth = 10;   % 异常探测方法种类

targ1 = R1_value(ind_tar);
targ2 = R2_value(ind_tar);
targ3 = R3_value(ind_tar);
targ4 = R4_value(ind_tar);
targ5 = R5_value(ind_tar);
targ6 = R6_value(ind_tar);
targ7 = R7_value(ind_tar);
targ8 = R8_value(ind_tar);
targ9 = R9_value(ind_tar);
targ10 = R10_value(ind_tar);

back1 = R1_value(ind_bac);
back2 = R2_value(ind_bac);
back3 = R3_value(ind_bac);
back4 = R4_value(ind_bac);
back5 = R5_value(ind_bac);
back6 = R6_value(ind_bac);
back7 = R7_value(ind_bac);
back8 = R8_value(ind_bac);
back9 = R9_value(ind_bac);
back10 = R10_value(ind_bac);


X_targ = [targ1;targ2;targ3;targ4;targ5;targ6;targ7;targ8;targ9;targ10]';
X_back = [back1;back2;back3;back4;back5;back6;back7;back8;back9;back10]';
X = [X_targ(:);X_back(:)];
X = X(:);
g1_targ = [ones(1,num_targ); 2*ones(1, num_targ); 3*ones(1, num_targ);4*ones(1, num_targ);...
    5*ones(1, num_targ);6*ones(1, num_targ);7*ones(1, num_targ);8*ones(1, num_targ);9*ones(1, num_targ);10*ones(1, num_targ)]'; 
g1_back = [ones(1, num_back); 2*ones(1, num_back); 3*ones(1, num_back);4*ones(1, num_back);...
    5*ones(1, num_back);6*ones(1, num_back);7*ones(1, num_back);8*ones(1, num_back);9*ones(1, num_back);10*ones(1, num_back)]'; 
g1 = [g1_targ(:); g1_back(:)];
g1 = g1(:);
g2 = [ones(num_meth*num_targ,1);2*ones(num_meth*num_back,1)];
g2 = g2(:);
positions = [[1:num_meth],[1:num_meth]+0.3];

%%
figure(2);
bh=boxplot(X, {g2,g1} ,'whisker',10000,'colorgroup',g2, 'symbol','.','outliersize',4,'widths',0.2,'positions',positions);
set(bh,'LineWidth',1.7)
ylabel('Detection test statistic range');


% grid on
% set(gca,'YLim',[0,0.5],'gridLineStyle', '-.');

% ylim([0,0.0065])  % 用于y轴的坐标轴显示范围的控制

Xtick_pos = [1:num_meth]+0.15;% 确定label显示的位置
Xtick_label  ={'GRXD','RPCA-RX','LSMAD','LRASR','TPCA','GTVLRR', 'PCA-TLRSR','LREN', 'DeCNN-AD' , 'GNLTR'};
set(gca,'XTickLabel',Xtick_label, 'XTick',Xtick_pos);
xtickangle(15)% 旋转标签角度

% 显示图例
h=findobj(gca,'tag','Outliers');
delete(h) 
legend(findobj(gca,'Tag','Box'),{'Background','Anomaly'})


%% 最大值与最小值（箱须至高与至低点：whisker 为0-100%）
p_targ = prctile(X_targ,[0 100]);
p_back = prctile(X_back,[0 100]);
% p_targ = prctile(X_targ,[10 90]);
% p_back = prctile(X_back,[10 90]);
p = [];
for i = 1:num_meth
    p = [p,p_targ(:,i),p_back(:,i)];
end

% 箱子的上边缘与下边缘 (异常、背景区域10% 与 90% 统计)
q_targ = quantile(X_targ,[0.1 0.9]);  
q_back = quantile(X_back,[0.1 0.9]);  
% q_targ = quantile(X_targ,[0.09 0.81]);  
% q_back = quantile(X_back,[0.09 0.81]);  
q = [];
for i = 1:num_meth
    q = [q,q_targ(:,i),q_back(:,i)];
end

h = flipud(findobj(gca,'Tag','Box'));
for j = 1:length(h)
    q10 = q(1,j);
    q90 = q(2,j);
    set(h(j),'YData',[q10 q90 q90 q10 q10]);
end

% Replace upper end y value of whisker
h = flipud(findobj(gca,'Tag','Upper Whisker'));
for j=1:length(h)
%     ydata = get(h(j),'YData');
%     ydata(2) = p(2,j);
%     set(h(j),'YData',ydata);
    set(h(j),'YData',[q(2,j) p(2,j)]);
end

% Replace all y values of adjacent value
h = flipud(findobj(gca,'Tag','Upper Adjacent Value'));
for j=1:length(h)
%     ydata = get(h(j),'YData');
%     ydata(:) = p(2,j);
    set(h(j),'YData',[p(2,j) p(2,j)]);
end

% Replace lower end y value of whisker
h = flipud(findobj(gca,'Tag','Lower Whisker'));
for j=1:length(h)
%     ydata = get(h(j),'YData');
%     ydata(1) = p(1,j);
    set(h(j),'YData',[q(1,j) p(1,j)]);
end

% Replace all y values of adjacent value
h = flipud(findobj(gca,'Tag','Lower Adjacent Value'));
for j=1:length(h)
%     ydata = get(h(j),'YData');
%     ydata(:) = p(1,j);
    set(h(j),'YData',[p(1,j) p(1,j)]);
end

