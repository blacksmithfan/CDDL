function B = retrieve_dictionary(database,para )
Bpath = ['dictionary/dict_' para.dataSet '_' num2str(para.nBases) '.mat'];
Xpath = ['dictionary/rand_patches_' para.dataSet '_' num2str(para.nsmp) '.mat'];
if ~para.skip_dic_training,
    try
        load(Xpath);
    catch
        X = rand_sampling(database, para.nsmp);
        save(Xpath, 'X');
    end
    [B, S, stat] = reg_sparse_coding(X, para.nBases, eye(para.nBases), para.beta, para.gamma, para.num_iters);
    save(Bpath, 'B', 'S', 'stat');
else
    load(Bpath);
end
end

