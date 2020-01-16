function [loss1, loss2, loss3, loss4, loss5] = loss(Xs, As, Ys, Hs, D, W, U, pars)
loss1 = norm(Xs-D*Ys, 'fro');
loss2 = pars.alpha*norm(Ys-W*As, 'fro');
loss3 = pars.beta*norm(Hs-U*Ys, 'fro');
loss4 = pars.phi*norm(Ys-U'*Hs, 'fro');
loss5 = pars.lambda*norm(Ys-D'*Xs, 'fro');
fprintf('loss:%.2f,%.2f,%.2f,%.2f,%.2f\n', loss1, loss2, loss3, loss4, loss5);
% disp([loss1,',',loss2,',',loss3,',',loss4,',',loss5]);