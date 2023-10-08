function [ L,E,rank] = dictionary_construction( XX, opts, fun, gamma)
[L,E,rank,obj,err,iter] = trpca_tnn_w(XX,opts.lambda,opts, fun, gamma);
end

