function [none] = main(alpha, beta, phi,lambda, gammaw, gammay, iter, dnum)
addpath('utils');

% 读取数据
dataset_path = '.\datasets\AwA';
load(sprintf('%s/predicateMatrixContinuous.mat', dataset_path));
load(sprintf('%s/feat-imagenet-vgg-verydeep-19.mat', dataset_path));
load(sprintf('%s/trainTestSplit.mat', dataset_path));

% 归一化
class_attributes = predicateMatrixContinuous/max(max(predicateMatrixContinuous));
train_feature=double(train_feat);
test_feature=double(test_feat);

% PCA降维
% [pc,~,latent,~]=pca(train_feature');
% tran=pc(:,1:200);
% options.ReducedDim = 200;
% [tran, ~] = PCA_DengCai(train_feature', options);
% train_feature = bsxfun(@minus,train_feature,mean(train_feature,1));
% train_feature = train_feature'*tran;
% train_feature = train_feature';
% test_feature = bsxfun(@minus,test_feature,mean(train_feature,1));
% test_feature = test_feature'*tran;
% test_feature = test_feature';

train_label=double(class_attributes(train_labels,:))';
test_label=double(class_attributes(test_labels,:))';

test_class_label=double(class_attributes(testClassLabels,:))';
train_class_label=double(class_attributes(trainClassLabels,:))';

%% parameter settings
pars.alpha=alpha;
pars.beta=beta;
pars.phi=phi;
pars.lambda=lambda;
pars.gammaw=gammaw;
pars.gammay=gammay;
pars.iter=iter;
pars.numBaseU=dnum; % 字典列数
% pars.alpha=10;
% pars.beta=50;
% pars.phi=10;
% pars.lambda=10;
% pars.gammaw=0.5;
% pars.gammay=1;
% pars.iter=20;
% pars.numBaseU=200; % 字典列数
pars.dimFeature=size(train_feature,1);
pars.dimSemantic=size(class_attributes,2);
pars.numTrainClass=length(trainClassLabels);

% 初始化
name = 'datasets/initial_awa_ADS.mat';
if ~exist(name,'file')
    [D,~,~]= initial(pars, name);
else
    load(name);
end

SIM = get_sim(class_attributes, trainClassLabels); % 计算相似性空间
H = get_H(trainClassLabels, train_labels); % one hot标签

X0 = learn_coefficients_noise(D,train_feature,pars.gammay); % 先利用初始化得到的D计算出Ys
W  = learn_basis(X0,train_label,pars.gammaw); % 根据字典D与Ys用阿格朗日对偶问题计算W
U  = SAE(H,X0,pars.phi/pars.beta); % 根据字典D与Ys用自动编码器计算U

diary on
fprintf('pars: alpha:%f,beta:%f,phi:%f,lambda:%f,gammaw:%f,gammay:%f,iter:%d,dnum:%d\n',alpha, beta, phi, lambda, gammaw, gammay, iter, dnum);
% 第一次测试
accuracy = evaluate(test_feature,test_labels,testClassLabels,test_class_label,SIM,D,W,U,pars);
fprintf('first, test accuracy is %.2f%%\n',accuracy*100);

%% train models
fprintf('Optimizing the models...\n');
[D, W, U] = optimize(train_feature,train_label,H,D,W,U,pars,test_feature,test_labels,testClassLabels,test_class_label,SIM);
fprintf('Testing the models...\n');
accuracy = evaluate(test_feature,test_labels,testClassLabels,test_class_label,SIM,D,W,U,pars);
fprintf('finally, test accuracy is %.2f%%\n',accuracy*100);
