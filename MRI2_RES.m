% This code is designed to collect the results from MRW2_MWPCR.m
% After all the paralleled jobs from MRW2_MWPCR.m, please submit this code
% to the server.
% This program will analyze the results from the 10 fold cross validation
% and save the final testing accuracy to the file MRI2_MWPCR.mat

testacc = zeros(11,11,9,10);
tuneacc = zeros(11,11,9,10);
tuneauc = zeros(11,11,9,10);
[a,b]=meshgrid(1:10,1:9);
a1=[a(:),b(:)];
a=kron(a1,ones(11^2,1));
b11=(1:11)';
b1=kron(b11,ones(11,1));
b=repmat(b1,90,1);
c1=(1:11)';
c=repmat(c1,11*90,1);
para_pool=[a,b,c];

for ii = 1 : 10890
    
    test_n = para_pool(ii,1);
    tune_nn = para_pool(ii,2);
    lambda1_n = para_pool(ii,3);
    lambda2_n = para_pool(ii,4);
    train_n = setdiff(1:10,test_n);
    tune_n = train_n(tune_nn);
    
    filename = sprintf('/RealData/Res/MRI_%.2d_%.2d_%.2d_%.2d.mat', ...
        test_n,tune_n,lambda1_n,lambda2_n);
    load (filename)
    testacc(lambda1_n,lambda2_n,tune_nn,test_n) = MWPCR_test;
    tuneacc(lambda1_n,lambda2_n,tune_nn,test_n) = MWPCR_tune;
    tuneauc(lambda1_n,lambda2_n,tune_nn,test_n) = MWPCR_tune_AUC;
end

Tune_ACC = zeros(11,11);
Tune_AUC = zeros(11,11);
Test_ACC = zeros(11,11);

Final_ACC = zeros(10,1);

for test_i = 1:10
    Tune_ACC = mean(tuneacc(:,:,:,test_i),3);
    Tune_ACC = Tune_ACC(:);
    [tune_index,~] = find(Tune_ACC==max(Tune_ACC));
    if length(tune_index)>1
        Tune_AUC = mean(tuneauc(:,:,:,test_i),3);
        Tune_AUC = Tune_AUC(:);
        [tune_index_AUC,~] = find(Tune_AUC(tune_index)==max(Tune_AUC(tune_index)));
        tune_index = tune_index(tune_index_AUC);
    end
    Test_ACC = mean(testacc(:,:,:,test_i),3);
    Test_ACC = Test_ACC(:);
    Final_ACC(test_i) = Test_ACC(tune_index);
end

filename = '/RealData/Res/MRI2_MWPCR.mat';

save (filename, 'testacc', 'tuneacc', 'tuneauc','Final_ACC')
