function [ sc_fea, sc_label ] = get_global( database,B,para )
nBases = size(B, 2);  
dimFea = sum(nBases*para.pyramid.^2);
numFea = length(database.path);

sc_fea = zeros(dimFea, numFea);
sc_label = zeros(numFea, 1);

disp('==================================================');
fprintf('Calculating the sparse coding feature...\n');
fprintf('Regularization parameter: %f\n', para.gamma);
disp('==================================================');

for iter1 = 1:numFea,  
    if ~mod(iter1, 50),
        fprintf('.\n');
    else
        fprintf('.');
    end;
    fpath = database.path{iter1};
    load(fpath);
    if para.knn,
        sc_fea(:, iter1) = sc_approx_pooling(feaSet, B, para.pyramid, para.gamma, para.knn);
    else
        sc_fea(:, iter1) = sc_pooling(feaSet, B, para.pyramid, para.gamma);
    end
    sc_label(iter1) = database.label(iter1);
end;

end

