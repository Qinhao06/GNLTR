function [area_TLRR] = PCA_TLRSR_function(data, mask, numb_dimension, opts, lambda)
 
%   f_show=data(:,:,[37,18,8]);
% for i=1:3
%     max_f=max(max(f_show(:,:,i)));
%     min_f=min(min(f_show(:,:,i)));
%     f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
% end
DataTest=data;


% TIR = load('HYDICE_urban.mat');
% DataTest1 = TIR.data;
% DataTest = DataTest1./max(DataTest1(:));
% mask = double(TIR.map);

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

% opts.lambda =0.06;%airport-1 
% opts.lambda =0.06; %airport-2 
% opts.lambda =0.06; %airport-3
% opts.lambda =0.06; %airport-4


% opts.lambda =0.02;%san
% opts.lambda =0.06;%Urban

%     opts.mu = 1e-4;
%     opts.tol = 1e-8;
%     opts.rho = 1.1;
%     opts.max_iter = 100;
%     opts.DEBUG = 0;
        tic;
    [ L,S,rank] = dictionary_learning_tlrr( X, opts);

    %% test PCA-TLRSR

  max_iter=100;
  Debug = 0;
% lambda=0.01;%airport-1  
% lambda=0.01;%airport-2 
% lambda=0.05;%airport-3 
% lambda=0.05;%airport-4 % :)


% lambda=0.01;%HY-Urban
% lambda=0.01;%San :)

    [Z,tlrr_E,Z_rank,err_va ] = GNLTA(X,L,max_iter,lambda,Debug);


    %% compute AUC
    E = reshape(tlrr_E, num, Dim)';
    r_new = sqrt(sum(E.^2, 1));
    r_max = max(r_new(:));
    taus = linspace(0, r_max, 5000);
    PF_40 = zeros(1, 5000);
    PD_40 = zeros(1, 5000);

    for index2 = 1:length(taus)
        tau = taus(index2);
        anomaly_map_rx = (r_new > tau);
        PF_40(index2) = sum(anomaly_map_rx & normal_map) / sum(normal_map);
        PD_40(index2) = sum(anomaly_map_rx & anomaly_map) / sum(anomaly_map);
    end

    area_TLRR = sum((PF_40(1:end - 1) - PF_40(2:end)) .* (PD_40(2:end) + PD_40(1:end - 1)) / 2);
    % disp("area_TLRR =");
    % disp(area_TLRR);
    % disp("TLRSR_lambda" );
    % disp(lambda);
    % disp("opts:");
    % disp(opts);

    % f_show=reshape(r_new,[H,W]);
    % f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
    % figure('name','PCA_TLRR'), imshow(f_show);

end
