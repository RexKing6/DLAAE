function [SIM] = get_sim(class_attributes, trainClassLabels)
A = single(class_attributes');
%%% linear kernel
A = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);
A = A' * A;
%%% projection
H = A(trainClassLabels,trainClassLabels); % 维数40x40，40个可见类之间的相似性
F = A(trainClassLabels,:); % 维数40x50，40个可见类与50个类别之间的相似性
alpha = zeros(length(trainClassLabels), size(F,2));
for i = 1:size(F,2)
    f = -F(:,i);
    alpha(:,i) = quadprog(double(H + 1e-2*eye(size(H,2))),double(f),[],[],ones(1,length(f)),1,zeros(length(f),1));
    % 56项限制相似性得分的和为1，7项限制相似性得分大于或等于0
end
alpha(abs(alpha)<1e-3) = 0;
SIM = alpha ./ repmat(sum(alpha,1), [size(alpha,1) 1]);% 重新归一化，将相似性之和置为1
end

