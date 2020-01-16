function [ D, W, U ] = initial(pars, name)
% initial the parameters
D = rand(pars.dimFeature,pars.numBaseU)-0.5;
D = D - repmat(mean(D,1), size(D,1),1);
D = D*diag(1./sqrt(sum(D.*D)));

W = rand(pars.numBaseU,pars.dimSemantic)-0.5;
W = W - repmat(mean(W,1), size(W,1),1);
W = W*diag(1./sqrt(sum(W.*W)));

U = rand(pars.numTrainClass,pars.numBaseU)-0.5;
U = U - repmat(mean(U,1), size(U,1),1);
U = U*diag(1./sqrt(sum(U.*U)));

save(name,'D','W','U');

end

