function Mat2Tif(InputMatFileName,OutputTifFilename, data)
    load(InputMatFileName);
    InputMatImg=data;
    t = Tiff(OutputTifFilename,'w');
    if size(InputMatImg,3) == 3
        t.setTag('Photometric',Tiff.Photometric.RGB);
    else
        t.setTag('Photometric',Tiff.Photometric.MinIsBlack);%��ɫ�ռ���ͷ�ʽ
    end
    t.setTag('Compression',Tiff.Compression.None);%��ѹ��
    t.setTag('BitsPerSample',64);% ��������.matΪdouble���ͣ�����ѡ����64λ
    t.setTag('SamplesPerPixel',size(InputMatImg,3));% ÿ�����صĲ�����Ŀ
    t.setTag('SampleFormat',Tiff.SampleFormat.IEEEFP);% ���BitsPerSample64λdouble���ͣ�ѡ��IEEEFP����Ӧ
    t.setTag('ImageLength',size(InputMatImg,1));% Ӱ����
    t.setTag('ImageWidth',size(InputMatImg,2));% Ӱ��߶�
    t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);%ƽ������ѡ����ʽ
    t.write(InputMatImg);% ׼������ͷ�ļ�����ʼдӰ������
    t.close();% �ر�Ӱ��