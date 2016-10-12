function [D,X,T,W, err]=labelconsistentDTLC(Y,Dinit,Q_train,Tinit,H_train,iterations,sparsitythres,sqrt_alpha)

params.data = [Y;sqrt_alpha*Q_train];
params.Tdata = sparsitythres; 
params.iternum = iterations;
params.memusage = 'high';
D_ext2 = [Dinit;sqrt_alpha*Tinit];
D_ext2=normcols(D_ext2); 
params.initdict = D_ext2;
[Dksvd,X,err] = ksvd(params,'');

i_start_D = 1;
i_end_D = size(Dinit,1);
i_start_T = i_end_D+1;
i_end_T = i_end_D+size(Tinit,1);
D = Dksvd(i_start_D:i_end_D/2,:);
T = Dksvd(i_start_T:i_end_T,:);

l2norms = sqrt(sum(D.*D,1)+eps);
D = D./repmat(l2norms,size(D,1),1);
T = T./repmat(l2norms,size(T,1),1);
T = T./sqrt_alpha;

W = inv(X*X'+eye(size(X*X')))*X*H_train';
W = W';