function [ D, W, U ] = optimize( Xs, As, Hs, D, W, U, pars, test_feature,test_labels,testClassLabels,test_class_label,SIM)
%%
% Xs: train features
% As: train semantics
% Hs: train labels

%%
% h = waitbar(0);
for i=1:pars.iter
    %% X, 更新Ys
%     waitbar(i/pars.iter,h)
   	X  = [Xs; sqrt(pars.alpha)*W*As; sqrt(pars.beta)*Hs; sqrt(pars.phi)*U'*Hs; sqrt(pars.lambda)*D'*Xs];
  	D1 = [D; sqrt(pars.alpha)*eye(size(D,2)); sqrt(pars.beta)*U; sqrt(pars.phi)*eye(size(D,2)); sqrt(pars.lambda)*eye(size(D,2))];
   	Ys = learn_coefficients_noise(D1,X,pars.gammay);
    %% 更新D
    D = SAE(Xs, Ys, pars.lambda);
    loss(Xs, As, Ys, Hs, D, W, U, pars);
    accuracy = evaluate(test_feature,test_labels,testClassLabels,test_class_label,SIM,D,W,U,pars);
    fprintf('iter:%d, update D, test accuracy is %.2f%%\n', i, accuracy*100);
    %% 更新W  
    W = learn_basis(Ys, As, pars.gammaw);
    loss(Xs, As, Ys, Hs, D, W, U, pars);
    accuracy = evaluate(test_feature,test_labels,testClassLabels,test_class_label,SIM,D,W,U,pars);
    fprintf('iter:%d, update W, test accuracy is %.2f%%\n', i, accuracy*100);
    %% 更新U
    U = SAE(Hs, Ys, pars.phi/pars.beta);
    loss(Xs, As, Ys, Hs, D, W, U, pars);
    accuracy = evaluate(test_feature,test_labels,testClassLabels,test_class_label,SIM,D,W,U,pars);
    fprintf('iter:%d, update U, test accuracy is %.2f%%\n', i, accuracy*100);
end
end
