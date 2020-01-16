function [acc] = cosine_acc(image,class,labels,ClassLabels)
sim = class'*image;
a = sqrt(sum(class.^2));
b = sqrt(sum(image.^2));
sim = sim./(a'*b);
[~,id] = max(sim);
pre = ClassLabels(id);
acc = sum(pre==labels)/length(labels);
end