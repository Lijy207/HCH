function v = eval_spreg(res, K,param)
if isfield(param, 'gamma')
    gamma = param.gamma;
else
    gamma = 0.5;
end

if isfield(param, 'type')
    type = param.type;
else
    type = 'hard';
end

[n] = size(res,1);
v = zeros(n,1);

switch type
    case 'hard'        
        v(res<K) = 1;
    case 'linear'
        ind = res<K;
        v(ind) = 1 - res(ind);
    case 'log'        
        ind = res<K;
        v(ind) = 1/log(gamma).*log(gamma-1/K*res(ind));
    case 'mix'
        ind = res>1/(K+1/gamma) & res<1/K;
        v(ind) = gamma./res(ind) - gamma*K;
        v(res<=1/(K+1/gamma)) = 1;
    case 'mix_var'
        ind   = res>1/sqrt(K+1/gamma) & res<1/sqrt(K);
        v(ind) = gamma./sqrt(res(ind)) - gamma*K;    
        v(res<=1/sqrt(K+1/gamma)) = 1;   
end
end