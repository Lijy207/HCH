%% We use the Reuters dataset to test the proposed HCH method

clear;clc;
load('reuters.mat');
traindata=NormalizeFea(traindata,1);
testdata=NormalizeFea(testdata,1);
sampleMean1 = mean(traindata,1);
sampleMean2 = mean(testdata,1);
traindata = (traindata - repmat(sampleMean1,size(traindata,1),1));
testdata = (testdata - repmat(sampleMean2,size(testdata,1),1));
tn = size(testdata,1);
opt.Cluster_number = 5;
opt.bit = 16;
opt.alpha = 1;
opt.beta =  1;
opt.lambda = 1;
opt.it = 1;
Xcov = traindata'*traindata;
Xcov = (Xcov + Xcov')/(2*size(traindata, 2));
[U,S,~] = svd(Xcov);
X_train = traindata*U(:,1:opt.bit);
X_test = testdata*U(:,1:opt.bit);
[IDX,W,v] = HC_VW(X_train,opt);
[Y_train,Y_test, finH,U,S ] = HC_Hash( X_train,X_test,IDX,opt );
H = Y_train;
tH = Y_test;
B1 = compactbit(H); %train
B2= compactbit(tH); %test
Dhamm = hammingDist(B1, B2);
MAP = evaluate(B1, B2, traingnd, testgnd);
Results.MAP = MAP.AP;
Results.ham2 = MAP.PH2;
fprintf('The MAP is %8.5f\n',Results.MAP)
fprintf('The HAM2 is %8.5f\n',Results.ham2)

