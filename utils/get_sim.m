function [SIM] = get_sim(class_attributes, trainClassLabels)
A = single(class_attributes');
%%% linear kernel
A = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);
A = A' * A;
%%% projection
H = A(trainClassLabels,trainClassLabels); % ά��40x40��40���ɼ���֮���������
F = A(trainClassLabels,:); % ά��40x50��40���ɼ�����50�����֮���������
alpha = zeros(length(trainClassLabels), size(F,2));
for i = 1:size(F,2)
    f = -F(:,i);
    alpha(:,i) = quadprog(double(H + 1e-2*eye(size(H,2))),double(f),[],[],ones(1,length(f)),1,zeros(length(f),1));
    % 56�����������Ե÷ֵĺ�Ϊ1��7�����������Ե÷ִ��ڻ����0
end
alpha(abs(alpha)<1e-3) = 0;
SIM = alpha ./ repmat(sum(alpha,1), [size(alpha,1) 1]);% ���¹�һ������������֮����Ϊ1
end

