function [ database ] = compute_feature( para )

img_dir=para.img_dir
data_dir=para.data_dir;
dataSet=para.dataSet;
skip_cal_sift=para.skip_cal_sift;

rt_img_dir = fullfile(img_dir, dataSet);
rt_data_dir = fullfile(data_dir, dataSet);

%% calculate sift features or retrieve the database directory
if skip_cal_sift,
    database = retr_database_dir(rt_data_dir);
else
    database = CalculateSiftDescriptor(rt_img_dir, rt_data_dir, para.gridSpacing, para.patchSize, para.maxImSize, para.nrml_threshold);
end;


end

