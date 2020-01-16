function [ accuracy ] = evaluate( features, labels, ClassLabels,testClassSemantic, SIM, D, W, U, pars )
image_la = learn_coefficients_noise(D, features, pars.gammay);% ||Xu-DYu||_F^2+gamma||Yu||_2^2
class_la = W * testClassSemantic;
image_sim = U*image_la;
class_sim = SIM(:,ClassLabels);

image = [image_la;image_sim];
class = [class_la;class_sim];

accuracy = cosine_acc(image,class,labels,ClassLabels);
method = 'cosine';
[la_acc, sim_acc, la_sim_acc] = test(image_la ,testClassSemantic, class_la, class_sim, W, U,labels,ClassLabels, method);
fprintf('la,test accuracy is %.2f%%\n', la_acc*100); % 隐藏属性空间
fprintf('sim,test accuracy is %.2f%%\n', sim_acc*100); % 相似性空间
fprintf('la_sim,test accuracy is %.2f%%\n', la_sim_acc*100); % 隐藏属性+相似性空间
end

