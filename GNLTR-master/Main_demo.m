clear all;
close all;
 load('FFDNet_gray.mat');

load('pavia_car');
load('abu-beach-2.mat');
load('HYDICE_urban.mat');
K=16; % number of clusters, pavia car
beta=0.0004; % beta
lamda=0.04; % lambda

mask=map;
map=mask;

f_show=data(:,:,[37,18,8]);
for i=1:3
    max_f=max(max(f_show(:,:,i)));
    min_f=min(min(f_show(:,:,i)));
    f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
end

figure,imshow(f_show);
figure,imshow(mask,[]);
DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

Y=reshape(DataTest, num, Dim)';

E=DeCNNAD(Y,H,W,K,beta,lamda,net);

r_new=sqrt(sum(E.^2,1));
AUC=ROC(r_new,map,1) % AUC
f_anomaly=reshape(r_new,[H,W]);
f_anomaly=(f_anomaly-min(f_anomaly(:)))/(max(f_anomaly(:))-min(f_anomaly(:)));
figure, imshow(f_anomaly);

