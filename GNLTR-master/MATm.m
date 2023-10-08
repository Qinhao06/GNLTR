load Z.mat


f_show=Z(:,:,[1,1,1]);
for i=1:3
    max_f=max(max(f_show(:,:,i)));
    min_f=min(min(f_show(:,:,i)));
    f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
end


figure('name','GTVLRR'), imshow(f_show);imwrite(f_show,'Z2.jpg');