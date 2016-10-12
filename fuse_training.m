function [ training_set ] = fuse_training( training )

K=zeros(size(training.target,1),size(training.source,1));
for i=1:size(training.target,1)
    for j=1:size(training.source,1)
        K(i,j)=1/(2*pi)*exp(-sqrt(sum((training.target(i,:)-training.source(j,:)).^2)/(2*2)));
    end
end
for i=1:size(K,1)
    [~,idx]=max(K(i,:));
    K(i,:)=0;
    K(i,idx)=1;
end
training_set=[training.target K*training.source];

end

