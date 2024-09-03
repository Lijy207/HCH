function [ Y_train,Y_test,finH,U,S ] = HC_Hash( X,testdata,IDX,opt )
%% Hashing Learning With Hyper-Class Representation
% by Jiaye Li


bit = opt.bit;%number of bits
Cluster_number = opt.Cluster_number;
alpha = opt.alpha;
beta = opt.beta;
lambda = opt.lambda;
it = opt.it;
islocal = 0;
[n,d] =  size(X);
U = rand(bit,d);
II = ones(bit,bit);
VV = [];
V1 = [];
for i =1:Cluster_number   
    idx1 = find(IDX==i);
    XI{i} = X(idx1,:);
    idx2 = find(IDX~=i);
    XNOI{i} = X(idx2,:);
    [row(i),col(i)] = size(XI{i});
    I{i} = ones(row(i),1);
    e{i} = ones(row(i),1);
    E{i} = diag(e{i});
    u{i} = mean(XI{i}*U');
    V1 = u{i};
    VV = [VV;V1];
    uu{i} = mean(XNOI{i}*U');
    H{i} = zeros(row(i),bit);
    clear idx u{i};
end
%%
flagL = 'g';
S = zeros(n,n);

%%
LL = [];
t = 1;
INO = ones(Cluster_number-1,1);
while 1
    HA = [];
    
    %% optimize H
    for i =1:Cluster_number
        HH =  H{i};
        xi = XI{i};
        xnoi = XNOI{i};
        AA{i} = 0;
        switch flagL
            case 'g'
                options = [];
                options.NeighborMode = 'KNN';
                options.k = 5;
                options.WeightMode = 'HeatKernel';
                options.t = 1;
                W = constructW_xf(XI{i}*U',options);
                W = max(W,W');
                L{i} = diag(sum(W,2)) - W;
            case 'h'
                Weight = ones(n,1);
                options = [];
                options.NeighborMode = 'KNN';
                options.k = 5;
                options.WeightMode = 'HeatKernel';
                options.t = 1;
                options.bSelfConnected = 1;
                W = constructW_xf(X',options);
                Dv = diag(sum(W)');
                De = diag(sum(W,2));
                invDe = inv(De);
                DV2 = full(Dv)^(-0.5);
                L = eye(n) - DV2 * W * diag(Weight) * invDe * W' *DV2;     
        end
        V = VV([1:i-1,i+1:Cluster_number],:);
        L1 = L{i};
        I1 = I{i};
        u1 = uu{i};
        EE = E{i};
        for j=1:row(i)
            XU = xi*U';
            LBDA = pinv(L1(j,:)*EE(j,:)'- beta + lambda);
            HH(j,:) = 0;
            LEH = L1(j,:)*HH;
            HH(j,:) = LBDA*(-sum(LEH,1) +lambda.*XU(j,:) - beta.*INO'*V);
            yiu{j} = HH(j,:) -uu{i};
            AA{i} = AA{i} + sum(yiu{j}.*yiu{j});
            A1(i) = AA{i};
        end
        HA = [HA;HH];
        clear L{i};
        clear I{i};

        clear HH;
    end

    %%
    [U] = OptStiefelGBB_YYQ(U, HA, X, S, A1, II, alpha,beta,lambda);
    
    %%
    HU = HA*U;
    distx = L2_distance_1(HU',HU');
    
    for i=1:n
        if islocal == 1
            idxa0 = idx(i,2:k+1);
        else
            idxa0 = 1:n;
        end;
        dxi = distx(i,idxa0);
        ad = -(dxi)/(2*alpha);
        T(i,idxa0) = EProjSimplex_new(ad);
    end;
    
    S = (T+T')/2;
    L1 = diag(sum(S)) - S;
    L1 = (L1 + L1') / 2;
    distxi = (S - T).^2;
    
    finH = abs(HA);
    u5 = mean(finH,1);
    clear HA;
    
    t= t+1;
    if  t == 3,    break,     end
    
end
clear i j;

Y_train = sign(X*U');
Y_train(Y_train==1) = 0;
Y_train(Y_train==-1) = 1;
%%%
%% Obtain the binary codes for the test data using the optimal 
%U obtained from the training data.
Z2 = testdata*U';

%%
Y_test = sign(Z2);
Y_test(Y_test == 1) = 0;
Y_test(Y_test == -1) = 1;
clear i j n
end

