% Relevant techniques utilized in this implementation:
% ScSPM (http://www.ifp.illinois.edu/~jyang29/ScSPM.htm)
% K-SVD (http://www.cs.technion.ac.il/~ronrubin/software.html) 
% LC-KSVD (http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html)
% All codes are provided for noncommercial research use
% If you happen to use this code, please cite our work:
% [1] F. Zhu and L. Shao, ¡°Weakly-Supervised Cross-Domain Dictionary Learning for Visual Recognition¡±, 
% International Journal of Computer Vision (IJCV), vol. 109, no. 1-2, pp. 42-59, Aug. 2014.
% [2] F. Zhu and L. Shao, ¡°Enhancing Action Recognition by Cross-Domain Dictionary Learning¡±, 
% British Machine Vision Conference (BMVC), Bristol, UK, 2013.
% If you find any bugs in this code, please contact (fan.zhu@sheffield.ac.uk)
clear;clc;
% set path
addpath('sift');
addpath(genpath('sparse_coding'));
addpath(genpath('./'))
% parameter setting
nRounds=25;
set_tr_num = 30;                        % training number per category
para.img_dir = 'image';                  % directory for dataset images
para.data_dir = 'data';                  % directory to save the sift features of the chosen dataset
para.dataSet = 'your target data';

para.skip_cal_sift = 0;              % if 'skip_cal_sift' is false, set the following parameter
para.gridSpacing = 6;
para.patchSize = 16;
para.maxImSize = 300;
para.nrml_threshold = 1;                 % low contrast region normalization threshold (descriptor length)

% dictionary training for sparse coding
para.skip_dic_training = 0;
para.nBases = 512;
para.nsmp = 20000;
para.beta = 1e-5;                        % a small regularization for stablizing sparse coding
para.num_iters = 1;

% feature pooling parameters
% pyramid = [1, 2, 4];                % spatial block number on each level of the pyramid
para.pyramid = [1 2 4];                       % spatial block number on each level of the pyramid
para.gamma = 0.15;
para.knn = 0;                          % find the k-nearest neighbors for approximate sparse coding
% if set 0, use the standard sparse coding
sparsitythres = 100;
sqrt_alpha = 4;
sqrt_beta = 2;
dictsize = 300;
iterations = 1;
iterations4ini = 20;
if ~exist('sc_fea.mat')
    % compute feature
    database = compute_feature(para);
    % retrieve dictionary
    B = retrieve_dictionary(database,para);
    nBases = size(B, 2);
    % global representations
    [sc_fea,sc_label] = get_global(database,B,para);
    save('sc_fea','sc_fea');save('sc_label','sc_label')
% compute source feature
    para.dataSet = 'your source data';
    database_source = compute_feature(para);
    source_fea = get_global(database_source,B,para);
    save('source_fea','source_fea')
else
    load sc_fea;load sc_label;
    load source_fea
end

source_fea=source_fea(:,1:100);

source_fea=source_fea';
training.source=source_fea;
clabel = unique(sc_label);
nclass = length(clabel);

tt_confusion_matrix=[];tt_target_accuracy=[];al_target_accuracy=[];
for ii = 1:nRounds
    fprintf('Round: %d\n', ii);
    
    tr_idx = [];
    ts_idx = [];
    for jj = 1:nclass,
        idx_label = find(sc_label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        save_train_num(ii).num=idx_rand;
        %         idx_rand = 1:num;
        if set_tr_num>length(idx_rand)
            tr_num=length(idx_rand);
        else
            tr_num=set_tr_num;
        end
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    training_set = sc_fea(:, tr_idx)';
    training_label = sc_label(tr_idx);
    
    testing_set = sc_fea(:, ts_idx)';
    testing_label = sc_label(ts_idx);
    temp.data=[];
    temp.label=[]; 
    training.target=training_set;
    training_set=fuse_training(training);

    training_feats=training_set';
    testing_feats=testing_set';
    H_train=zeros(length(unique(length(training_label))),length(training_label));
    for label_i=1:length(training_label)
        H_train(training_label(label_i),label_i)=1;
    end
    fprintf('\nDTLC-KSVD initialization... ');
    [Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres);
    fprintf('done!');
    fprintf('\nDictionary learning by DTLC-KSVD1...');
    [D1,X1,T1,W1] = labelconsistentDTLC(training_feats,Dinit,Q_train,Tinit,H_train,iterations,sparsitythres,sqrt_alpha);
    fprintf('done!');
    H_test=zeros(11,length(testing_label));
    for label_i=1:length(testing_label)
        H_test(testing_label(label_i),label_i)=1;
    end
    [prediction1,accuracy1] = classification(D1, W1, testing_feats, H_test, sparsitythres);
    
    accuracy(ii) = accuracy1*100;
    fprintf('\nAccuracy: %8.2f%%',accuracy1*100);
end
fprintf('Mean accuracy: %f\n', mean(accuracy));
fprintf('Standard deviation: %f\n', std(accuracy));

