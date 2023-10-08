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
% load abu-beach-1.mat
load abu-urban-5.mat
% data=hsi;
% mask=hsi_gt;
mask = map;

f_show=data(:,:,[37,18,8]);
for i=1:3
    max_f=max(max(f_show(:,:,i)));
    min_f=min(min(f_show(:,:,i)));
    f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
end
figure,imshow(f_show);imwrite(f_show,'im.jpg');
figure,imshow(mask,[]);imwrite(mask,'gt.jpg');
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
% 
% %%%%%%%GTVLRR
% tic;
% Dict=ConstructionD_lilu(Y,15,20);
% lambda = 0.4;
% beta = 0.2;
% gamma =0.05;% 0;%
% display = true;
% [X,S] = lrr_tv_manifold(Y,Dict,lambda,beta,gamma,[H,W],display);
% toc
% disp('GTVLRR');
% r_gtvlrr=sqrt(sum(S.^2));
% r_max = max(r_gtvlrr(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r_gtvlrr > tau);
%   PF_gtvlrr(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD_gtvlrr(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% % f_show=reshape(r_gtvlrr,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','GTVLRR'), imshow(f_show);imwrite(f_show,'GTVLRR.jpg');
% 
% area_GTVLRR = sum((PF_gtvlrr(1:end-1)-PF_gtvlrr(2:end)).*(PD_gtvlrr(2:end)+PD_gtvlrr(1:end-1))/2)
% % NBOXPLOT(:,8)=f_show(:);
% 
% 
% % %%%%%%LRASR
% 
% beta=1;
% lamda=0.1;
% tic;
% [S,E]=LRASR(Y,Dict,beta,lamda,1);
% toc
% disp('LRSAR');
% r_new=sqrt(sum(E.^2,1));
% r_max = max(r_new(:));
% taus = linspace(0, r_max, 5000);
% PF40=zeros(1,5000);
% PD40=zeros(1,5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r_new> tau);
%   PF40(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD40(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area_LRASR = sum((PF40(1:end-1)-PF40(2:end)).*(PD40(2:end)+PD40(1:end-1))/2)
% % f_show=reshape(r_new,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','LRASR'), imshow(f_show);imwrite(f_show,'LRASR.jpg');
% % NBOXPLOT(:,4)=f_show(:)';
% 
% 
% 
% % %%%%%%
% % tic;
% % [T_all,Anomaly]=AD_lilu5(DataTest);
% % toc
% % Y0=reshape(Anomaly, num, Dim)';
% % r4 = RX(Y0);  % input: num_dim x num_sam    rx
% % r_max = max(r4(:));
% % taus = linspace(0, r_max, 5000);
% % for index2 = 1:length(taus)
% %   tau = taus(index2);
% %   anomaly_map_rx = (r4 > tau);
% %   PF4(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
% %   PD4(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% % end
% % f_show=reshape(r4,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','TDAD'), imshow(f_show);imwrite(f_show,'TDAD.jpg');
% % area_TDAD = sum((PF4(1:end-1)-PF4(2:end)).*(PD4(2:end)+PD4(1:end-1))/2);
% % NBOXPLOT(:,6)=f_show(:);
% 
% 
% tic;
% Anomaly=AD_lilu7(DataTest);
% toc
% disp('TPCA');
% r5 = RX(abs(Anomaly)');  % input: num_dim x num_sam    rx
% r_max = max(r5(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r5 > tau);
%   PF7(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD7(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% % f_show=reshape(r5,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','TPCA'), imshow(f_show);imwrite(f_show,'TPCA.jpg');
% area_TPCA = sum((PF7(1:end-1)-PF7(2:end)).*(PD7(2:end)+PD7(1:end-1))/2)
% % NBOXPLOT(:,7)=f_show(:);
% 
% 
% tic;
% r3 = RX(Y);  % input: num_dim x num_sam    rx
% toc
% disp('RX');
% r_max = max(r3(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r3 > tau);
%   PF3(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD3(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% % f_show=reshape(r3,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','RX'), imshow(f_show);imwrite(f_show,'RX.jpg');
% area_RX = sum((PF3(1:end-1)-PF3(2:end)).*(PD3(2:end)+PD3(1:end-1))/2)
% % NBOXPLOT(:,2)=f_show(:);
% % 
% % LSMAD
% 
% rank = 2; 
% 
% tic;
% [L,S,RMSE,error]=GoDec(Y',rank,floor(0.0022*Dim)*9,2);
% toc
% disp('LSMAD');
% L=L';
% S=S';
% 
% mu=mean(L,2);
% r_new2=(diag((Y-repmat(mu,[1,num]))'*pinv(cov(L'))*(Y-repmat(mu,[1,num]))))';
% 
% r_max = max(r_new2(:));
% taus = linspace(0, r_max, 5000);
% PF_41=zeros(1,5000);
% PD_41=zeros(1,5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r_new2> tau);
%   PF_41(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD_41(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area_LSMAD = sum((PF_41(1:end-1)-PF_41(2:end)).*(PD_41(2:end)+PD_41(1:end-1))/2)
% % f_show=reshape(r_new2,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','LSMAD'), imshow(f_show);imwrite(f_show,'LSMAD.jpg');
% % NBOXPLOT(:,5)=f_show(:);
% 
% %%%rpca   rx
% tic;
% [r0 ,Output_S, Output_L] = Unsupervised_RPCA_Detect_v1(DataTest);
% toc
% disp('RPCA');
% XS = reshape(Output_S, num, Dim);
% r2 = RX(XS');  % input: num_dim x num_sam
% 
% r_max = max(r2(:));                                              
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r2 > tau);
%   PF2(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD2(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% % f_show=reshape(r2,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','RPCA-RX'), imshow(f_show);imwrite(f_show,'RPCA-RX.jpg');
% % NBOXPLOT(:,3)=f_show(:);
% area_RPCA = sum((PF2(1:end-1)-PF2(2:end)).*(PD2(2:end)+PD2(1:end-1))/2)
% 
%%% PCA_TLRSR

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

numb_dimension = 10;
% numb_dimension = 15;%beach-1  
%  numb_dimension =15;%beach-2 
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
% opts.lambda =0.3; %beach-2 
% opts.lambda =0.3; %beach-3
% opts.lambda =0.03; %beach-4

% opts.lambda =0.07;%urban-1 
% opts.lambda =0.06; %urban-2 
% opts.lambda =0.8; %urban-3
% opts.lambda =0.02; %urban-4
opts.lambda =0.08; %urban-4

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
% lambda=0.05;%beach-2 
% lambda=0.05;%beach-3 
% lambda=0.0006;%beach-4 % :)

% lambda=0.2;%urban-1  
% lambda=0.03;%urban-2 
% lambda=0.03;%urban-3 
% lambda=0.001;%urban-4 % :)
lambda=0.02;%urban-5 % :)

% lambda=0.01;%HY-Urban
% lambda=0.01;%San :)




[Z,tlrr_E,Z_rank,err_va ] = TLRSR(X,L,max_iter,lambda,Debug);
toc;
disp('PCA-TLRSR');
E=reshape(tlrr_E, num, Dim)';
r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
taus = linspace(0, r_max, 5000);
PF10=zeros(1,5000);
PD10=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new> tau);
  PF10(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD10(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_TLRR = sum((PF10(1:end-1)-PF10(2:end)).*(PD10(2:end)+PD10(1:end-1))/2)
area_TLRR_taus = sum((taus(1:end-1)-taus(2:end)).*(PF10(2:end)+PF10(1:end-1))/2)
% area_TLRR_tus = trapz(taus, PF10)
f_show=reshape(r_new,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','PCA_TLRR'), imshow(f_show);imwrite(f_show,'PCA-TLRSR.jpg');
NBOXPLOT(:,9)=f_show(:); 

% figure,
% semilogx(taus, PF10, 'b-', 'LineWidth', 2);
% % % % % % %   LREN
% load('Sandie_goEnergy-map-cluster.mat')
% r_new = array;
% [m,n] = size(r_new);
% r_new = reshape(r_new,[1, m*n]);
% r_max = max(r_new(:));
% taus = linspace(0, r_max, 5000);
% PF21=zeros(1,5000);
% PD21=zeros(1,5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r_new> tau);
%   PF21(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD21(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area_LREN = sum((PF21(1:end-1)-PF21(2:end)).*(PD21(2:end)+PD21(1:end-1))/2)
% % f_show=reshape(r_new,[H,W]);
% % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% % figure('name','LREN'), imshow(f_show);imwrite(f_show,'LREN.jpg');
% % NBOXPLOT(:,9)=f_show(:); 

% 
% 
% % % % % % DECNN
% run './matlab/vl_setupnn'
% % load('pavia_car');
% K=10; % number of clusters, pavia car
% beta=0.03; % beta
% lamda=0.04; % lambda
% 
% load('FFDNet_gray.mat');
% % mask=map;
% % map=mask;
% 
% % f_show=data(:,:,[37,18,8]);
% % for i=1:3
% %     max_f=max(max(f_show(:,:,i)));
% %     min_f=min(min(f_show(:,:,i)));
% %     f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
% % end
% 
% % figure,imshow(f_show);
% % figure,imshow(mask,[]);
% DataTest=data;
% [H,W,Dim]=size(DataTest);
% num=H*W;
% for i=1:Dim
%     DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
% end
% 
% Y=reshape(DataTest, num, Dim)';
% tic
% E=DeCNNAD(Y,H,W,K,beta,lamda,net);
% toc
% r_new=sqrt(sum(E.^2,1));
% r_max = max(r_new(:));
% taus = linspace(0, r_max, 5000);
% PF11=zeros(1,5000);
% PD11=zeros(1,5000);
% % taus = linspace(0, r_max, 10000);
% % PF11=zeros(1,10000);
% % PD11=zeros(1,10000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (r_new> tau);
%   PF11(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD11(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area_dec1 = sum((PF11(1:end-1)-PF11(2:end)).*(PD11(2:end)+PD11(1:end-1))/2)
% % area_dec=ROC(r_new,map,1) % AUC
% % f_anomaly=reshape(r_new,[H,W]);
% % f_anomaly=(f_anomaly-min(f_anomaly(:)))/(max(f_anomaly(:))-min(f_anomaly(:)));
% % figure, imshow(f_anomaly);
% 
% % 


% %%% GNLTR


opts.mu = 1e-4;
opts.tol = 1e-8;
opts.rho = 1.1;
opts.max_iter = 100;
opts.DEBUG = 0;
max_iter=100;
Debug = 0;
% 
% opts.lambda = 0.05; % HY
% lambda = 0.01;
% numb_dimension = 7; 
% fun = 'lp' ;        gamma = 0.5;
% fun_dict = 'lp' ;        gamma_dict = 0.5; 
% 
opts.lambda = 0.001; % san
lambda = 0.05;
numb_dimension = 6; 
fun_dict = 'lp' ;       gamma_dict = 0.5;
fun = 'logarithm' ; gamma = 10 ;

% 
% % opts.lambda = 0.049; % air1
% % lambda = 0.002;
% % numb_dimension = 19; 
% % 
% % opts.lambda = 0.016; % air2
% % lambda = 0.002;
% % numb_dimension = 13;
% 
% % opts.lambda = 0.063; % air3
% % lambda = 0.177;
% % numb_dimension = 16; 
% % 
% % opts.lambda = 0.1; % air4
% % lambda = 0.013;
% % numb_dimension = 5; 
% % 




% 
% opts.lambda = 0.03; % beach1
% lambda = 0.03;
% numb_dimension = 15;
% fun_dict = 'mcp' ; gamma_dict = 10 ;
% fun = 'lp' ;        gamma = 0.5;
% % 
% opts.lambda = 0.05; % beach2
% lambda = 0.001;
% numb_dimension = 10;
% fun_dict = 'lp' ;        gamma_dict = 0.5; 
% fun = 'geman' ;  gamma = 10 ;

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

%  
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

 
    [Z,tlrr_E,Z_rank,err_va ] = GNLTR(X,L,max_iter,lambda,Debug, fun, gamma);

toc
disp('GNLTR');
    %% compute AUC
    E = reshape(tlrr_E, num, Dim)';
    r_new = sqrt(sum(E.^2, 1));
    r_max = max(r_new(:));
    taus = linspace(0, r_max, 5000);
    PF20 = zeros(1, 5000);
    PD20 = zeros(1, 5000);
  
    for index2 = 1:length(taus)
        tau = taus(index2);
        anomaly_map_rx = (r_new > tau);
        PF20(index2) = sum(anomaly_map_rx & normal_map) / sum(normal_map);
        PD20(index2) = sum(anomaly_map_rx & anomaly_map) / sum(anomaly_map);
    end

    area_GNLTA = sum((PF20(1:end - 1) - PF20(2:end)) .* (PD20(2:end) + PD20(1:end - 1)) / 2)
f_show=reshape(r_new,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','GNLTR'), imshow(f_show);imwrite(f_show,'GNLTR.jpg');
NBOXPLOT(:,10)=f_show(:);




%%%%PTA with LTV-norm

% tol1=1e-4;
% tol2=1e-6;
% maxiter=400;
% truncate_rank=1;
% alphia=1.7;
% beta=0.069;
% tau=0.1;
% 
%  tic;
%  [X,S,area] = AD_Tensor_LILU1(DataTest,alphia,beta,tau,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
% toc
% disp(PTA);
% f_show=sqrt(sum(S.^2,3));
% f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% 
% figure('name','Proposed'),imshow(f_show);imwrite(f_show,'PTA.jpg');
% NBOXPLOT(:,1)=f_show(:);
% 
% r_max = max(f_show(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (f_show(:)> tau)';
%   PF0(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD0(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area03=sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
%%%%

%%%%PTA with anisotropic TV-norm

% tol1=1e-4;
% tol2=1e-3;
% maxiter=29;
% truncate_rank=34;
% alphia=2.76;%2.76;
% beta=0.17;%1/sqrt(H*W);
% tau=0.009;
% tic;
%    [X,S,area] = AD_Tensor_LILU3(DataTest,alphia,beta,tau,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
% toc
% 
% f_show=sqrt(sum(S.^2,3));
% f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% 
% %figure('name','Proposed'),imshow(f_show);imwrite(f_show,'PTR.jpg');
% NBOXPLOT(:,1)=f_show(:);
% 
% r_max = max(f_show(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (f_show(:)> tau)';
%   PF02(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD02(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area02=sum((PF02(1:end-1)-PF02(2:end)).*(PD02(2:end)+PD02(1:end-1))/2);
% 
% %%%%PTA with isotropic TV-norm
% 
% tol1=1e-4;
% tol2=1e-3;
% maxiter=400;
% truncate_rank=10;
% alphia=0.01;%2.76;
% beta=172;%1/sqrt(H*W);
% tau=10;
% tic;
%    [X,S,area] = AD_Tensor_LILU2(DataTest,alphia,beta,tau,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
% toc
% 
% f_show=sqrt(sum(S.^2,3));
% f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% 
% %figure('name','Proposed'),imshow(f_show);imwrite(f_show,'PTR.jpg');
% NBOXPLOT(:,2)=f_show(:);
% 
% r_max = max(f_show(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (f_show(:)> tau)';
%   PF01(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD01(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area01=sum((PF01(1:end-1)-PF01(2:end)).*(PD01(2:end)+PD01(1:end-1))/2);
% 
% 
% figure,
% plot(PF01, PD01, 'b-', 'LineWidth', 2);hold on;
% plot(PF02, PD02, 'r-', 'LineWidth', 2);  
% plot(PF03, PD03, 'g-', 'LineWidth', 2);  
% hold off;
% xlabel('False alarm rate'); ylabel('Probability of detection');
% legend('anisotropic','isotropic','LTV-norm');
% axis([0 0.1 0 1]);hold off;
% 
% figure,
% semilogx(PF01, PD01, 'b-', 'LineWidth', 2);hold on;
% semilogx(PF02, PD02, 'r-', 'LineWidth', 2);  
% semilogx(PF03, PD03, 'g-', 'LineWidth', 2);  
% hold off;
% xlabel('False alarm rate'); ylabel('Probability of detection');
% legend('anisotropic','isotropic','LTV-norm');
% axis([0 0.1 0 1]);hold off;



figure,
plot(PF3, PD3, 'c-', 'LineWidth', 2);hold on;
plot(PF2, PD2, 'r-', 'LineWidth', 2);
plot(PF_41, PD_41, 'y-', 'LineWidth', 2);
plot(PF40, PD40, 'm-', 'LineWidth', 2);
% plot(PF4, PD4, 'c-', 'LineWidth', 2);
plot(PF7, PD7,  'Color',[0,0.75,0.25], 'LineWidth', 2);
plot(PF_gtvlrr, PD_gtvlrr, 'Color',[0.5,0.5,1], 'LineWidth', 2);
% plot(PF0, PD0, 'g-', 'LineWidth', 2);  
plot(PF10, PD10, 'g-', 'LineWidth', 2);
plot(PF21, PD21, 'Color',[0,0.75,1], 'LineWidth', 2);
plot(PF11, PD11, 'k-', 'LineWidth', 2);
plot(PF20, PD20, 'b-', 'LineWidth', 4);
hold off;
xlabel('False alarm rate'); ylabel('Probability of detection');
% legend('GRXD','RPCA-RX','LSMAD','LRASR','TDAD','TPCA','GTVLRR','PTA');
% legend('GRXD','RPCA-RX','LSMAD','LRASR','TPCA','GTVLRR','PTA', 'PCA-TLRSR', 'GNLTA');
legend('GRXD','RPCA-RX','LSMAD','LRASR','TPCA','GTVLRR', 'PCA-TLRSR','LREN', 'DeCNN-AD','GNLTR');
axis([0 0.1 0 1]);hold off;

figure, 
semilogx(PF3, PD3, 'c-', 'LineWidth', 3); hold on;
semilogx(PF2, PD2, 'r-', 'LineWidth', 2);  
semilogx(PF_41, PD_41, 'y-', 'LineWidth', 2);
semilogx(PF40, PD40, 'm-', 'LineWidth', 2);
% semilogx(PF4, PD4, 'c-', 'LineWidth', 2);
semilogx(PF7, PD7, 'Color',[0,0.75,0.25], 'LineWidth', 2);
semilogx(PF_gtvlrr, PD_gtvlrr, 'Color',[0.5,0.5,1], 'LineWidth', 2);
% semilogx(PF0, PD0, 'g-', 'LineWidth', 2);
semilogx(PF10, PD10, 'g-', 'LineWidth', 2);
semilogx(PF21, PD21, 'Color',[0,0.75,1], 'LineWidth', 2);
semilogx(PF11, PD11, 'k-', 'LineWidth', 2);
semilogx(PF20, PD20, 'b-', 'LineWidth', 4);
hold off;
xlabel('False alarm rate'); ylabel('Probability of detection');
% legend('GRXD','RPCA-RX','LSMAD','LRASR','TDAD','TPCA','GTVLRR','PTA');
% legend('GRXD','RPCA-RX','LSMAD','LRASR','TPCA','GTVLRR','PTA', 'PCA-TLRSR', 'GNLTA');
legend('GRXD','RPCA-RX','LSMAD','LRASR','TPCA','GTVLRR', 'PCA-TLRSR','LREN', 'DeCNN-AD','GNLTR');
axis([0 1 0 1]);hold off;
