function [la_acc, sim_acc, la_sim_acc ] = test( X,proto_a, proto_la, proto_sim, W, W1, test_labels, testClassLabels, method)
% X_a = W\X;
X_la = X;
X_sim = W1*X;

switch method
    case 'cosine'
%         a_acc = cosine_acc(X_a,proto_a,test_labels,testClassLabels);
        la_acc = cosine_acc(X_la,proto_la,test_labels,testClassLabels);
        sim_acc = cosine_acc(X_sim,proto_sim,test_labels,testClassLabels);
%         a_sim_acc = cosine_acc([X_a;X_sim],[proto_a;proto_sim],test_labels,testClassLabels);
%         la_a_acc = cosine_acc([X_la;X_a],[proto_la;proto_a],test_labels,testClassLabels);
        la_sim_acc = cosine_acc([X_la;X_sim],[proto_la;proto_sim],test_labels,testClassLabels);
%         la_a_sim_acc = cosine_acc([X_la;X_a;X_sim],[proto_la;proto_a;proto_sim],test_labels,testClassLabels);
    case 'euclidean'
        a_acc = euclidean_acc(X_a,proto_a,testClassLabels,test_labels);
        la_acc = euclidean_acc(X_la,proto_la,testClassLabels,test_labels);
        sim_acc = euclidean_acc(X_sim,proto_sim,testClassLabels,test_labels);
        a_sim_acc = euclidean_acc([X_a;X_sim],[proto_a,proto_sim],testClassLabels,test_labels);
        la_a_acc = euclidean_acc([X_la;X_a],[proto_la;proto_a],testClassLabels,test_labels);
        la_sim_acc = euclidean_acc([X_la;X_sim],[proto_la;proto_sim],testClassLabels,test_labels);
        la_a_sim_acc = euclidean_acc([X_la;X_a;X_sim],[proto_la;proto_a;proto_sim],testClassLabels,test_labels);
        
end


end

