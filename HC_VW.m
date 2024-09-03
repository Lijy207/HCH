function [IDX,W,v] = HC_VW(X,opt)
%% Ouput
% input:   X:    ins * fea
%          Y:    ins * class
%  opt.alpha:    is a control parameter
% output:  W:    fea * class
%% Hashing Learning With Hyper-Class Representation
%  by Jiaye Li


alpha=opt.alpha;
[n,d] =  size(X);
Cluster_number = opt.Cluster_number;

for i =1:d
    XX = X;
    XX(:,i) = [];
    X_NO_i{i} = XX;
    clear XX;
end

k_noi = [];
for i = 1:d
    kx = X_NO_i{i};
    for j =1:n
        k_no_i(j) = kx(j,:)*kx(j,:)';
    end
    k_noi = [k_noi;k_no_i];
end

v0    = zeros(d,1);
idx   = randperm(d);
v0(idx(1:ceil(d/2)))=1;
Q = v0.*X';
G = v0.*k_noi;
W = rand(n,n);
for i =1:n
    dn(i) = sqrt(sum((sum(W.*W,2)+eps)))./sum(W(i,:));
end
F = diag(dn);
res = abs(sum(X' - k_noi * W,2).^2);

L_med    = median(res);

param.type = 'linear';
type=param.type;
switch param.type
    case 'hard'
        K = L_med;
    case 'linear'
        K = L_med;
    case 'log'
        K = L_med;
    case 'mix'
        param.gamma = 2*L_med;
        K           = 1/param.gamma;
    case 'mix_var'
        param.gamma = 2*sqrt(L_med);
        K           = 1/param.gamma;
end

t    = 1;
obji = 1;
while 1
    v = eval_spreg(res, K, param);
    
    Q = v0.*X';
    G = v0.*k_noi;
    W = (G'*G + alpha.*F)\(G'*Q);
    for i =1:n
        dn(i) = sqrt(sum((sum(W.*W,2)+eps)))./sum(W(i,:));
    end
    F = diag(dn);
    
    res = abs(sum(X' - k_noi * W,2).^2);
    if strcmp(type,'hard')||strcmp(type,'linear')||strcmp(type,'log')
        K       =  K/0.65;
    else
        K       =  K/1.1;
    end
    
    t         = t+1;
    if  t == 3,    break,     end
end

dfeature = find(v == max(v));
IDX = kmeans(X(:,dfeature(1)),Cluster_number);


end

