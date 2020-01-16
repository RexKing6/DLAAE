function [ H ] = get_H( trainClassLabels, train_labels )
% transform the train_labels to one-hot vectors
H=zeros(length(trainClassLabels),length(train_labels));
temp=unique(train_labels);

for i=1:length(temp)
    ind=find(train_labels==temp(i));
    H(i,ind)=1;
end

end

