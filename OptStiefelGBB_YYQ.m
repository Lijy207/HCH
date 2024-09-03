function [U] = OptStiefelGBB_YYQ(U, HA, X, S, AA, II, alpha, beta, lambda)

  options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'Cosine';
options.t = 1;
W = constructW_xf(X,options);
W = max(W,W');
L2 = diag(sum(W,2)) - W; 
clear W;

function [F, G] = fun(U, HA, X, S, AA,II, alpha,beta,lambda)
  G = 2.*HA'*L2*HA*U+lambda.*(2.*U*X'*X-2.*HA'*X)+2.*II*U; 
  F = trace(U'*HA'*L2*HA*U) +lambda*norm(HA-X*U', 'fro')^2 + alpha*sum(sum(S.^2)) + sum(sum(U.^2)) - beta*sum(AA); 
end


opts.record = 0; %
opts.mBitr  = 1000;
opts.Btol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;


tic; [U, out]= OptStiefelGBB(U, @fun, opts, HA,X,S,AA,II,alpha,beta,lambda); tsolve = toc;
out.fval = -2*out.fval;
end
