function [ L,E,rank] = dictionary_learning_tlrr1( XX, opts)
[L,E,rank,obj,err,iter] = trpca_tnn_w1(XX,opts.lambda,opts);
end

