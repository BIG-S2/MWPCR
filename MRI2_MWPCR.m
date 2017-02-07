function  MRI2_MWPCR (LSB_JOBINDEX)
%%%%%%%%%%%%% Analysis of MRI data (AD v.s. NC) %%%%%%%%%%%%%
% This is an example for large data set, using 10 fold cross validation,
% with an outter 10 fold for testing evaluation and an inner 9 fold for
% parameter selection. Submit the job with LSB_JOBINDEX = 1~10890 to the
% server parallely.
% This code is designed for large scale data analysis, please make sure you
% assign enough memory limit for this computation.

[a,b]=meshgrid(1:10,1:9);
a1=[a(:),b(:)];
a=kron(a1,ones(11^2,1));
b11=(1:11)';
b1=kron(b11,ones(11,1));
b=repmat(b1,90,1);
c1=(1:11)';
c=repmat(c1,11*90,1);
para_pool=[a,b,c];

test_n = para_pool(LSB_JOBINDEX,1);
tune_n = para_pool(LSB_JOBINDEX,2);
lambda1_n = para_pool(LSB_JOBINDEX,3);
lambda2_n = para_pool(LSB_JOBINDEX,4);

train_n = setdiff(1:10,test_n);
tune_n = train_n(tune_n);
train_n = setdiff(train_n,tune_n);

%% Please specify the data file accordingly.
load /RealData/ADNI_MRI_rm.mat
% ten fold cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1=sum(MRI.label==1);% # of NC
% n2=sum(MRI.label==2);% # of MCI
n3=sum(MRI.label==3);% # of AD


rawlabel = MRI.label([find(MRI.label ==1);find(MRI.label ==3)]);
rawdata = MRI.data([find(MRI.label ==1);find(MRI.label ==3)],:);
data.index = MRI.index;

seed=100;
rng(seed)
Indices1 = crossvalind('Kfold', n1, 10);
% rng(seed)
% Indices2 = crossvalind('Kfold', n2, 10);
rng(seed)
Indices3 = crossvalind('Kfold', n3, 10);

test_index = [Indices1==test_n;Indices3==test_n];
tune_index = [Indices1==tune_n;Indices3==tune_n];
train_index = logical(1-test_index-tune_index);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data.train_x = rawdata(train_index,:);
data.tune_x = rawdata(tune_index,:);
data.test_x = rawdata(test_index,:);

data.train_y = (rawlabel(train_index)-1)/2;
data.tune_y = (rawlabel(tune_index)-1)/2;
data.test_y = (rawlabel(test_index)-1)/2;

thres_pool = linspace(0.9,0.99,11);
pc_num_pool = 1:11;

thres = thres_pool(lambda1_n);
pc_num = pc_num_pool(lambda2_n);


mask = MRI.mask;
imagesize = size(mask);
A = MRI.aug;
clearvars MRI

% Assigning the training covariates. You need modify this accordingly.
X_tr = data.train_x;
% Assigning the training response. You need modify this accordingly.
Y_tr = data.train_y;
% Compute the threshold for classification;
cla_thres = sum(Y_tr)/length(Y_tr);
[n,~] = size(X_tr);


%% This command will genenrate the head file and Nbr_Dist_2 for futher analysis.
% Please choose the right parameters according to your image types.

h = 1.22; % 1.5 for 1D; 1.28 for 2D; 1.2 for 3D;
S = 3; % 3 for 1D; 4 for 2D; 5 for 3D;

%======================Do Not Edit======================%
% The head file contains all the neighborhood information.
[Mask_Idx, ~,Nbr_Dist_2] = Head_File_For_Mask_MWPCR(mask,h^S);

%% Compute the multi-scale adpative coefficients.
[beta,beta_cov,WE]  = MARM (Mask_Idx,Nbr_Dist_2,X_tr,Y_tr,h,S);

%% Compute the projection matrix Q in the paper.
[Q,~] = MSW (WE,beta,beta_cov,thres);

%% Project the data using the weight matrix Q.
X_t = (X_tr-repmat(mean(X_tr,1),n,1))*Q; % X_t is X_tilde in the paper

%% Further analysis on the projected data, this is an example of PCA
[U,D,V] =   svd(X_t, 'econ');
% E2 = diag(D); % Eigen Values;
PC_score = U*D; % PC scores;
PC_loading = V; % PC loadings;

%% Further analysis using PC scores, this is a classification analysis
% Specify the number of PC components

% Create the design matrix with the PC scores;
X_mwpcr = [ones(n,1),PC_score(:,1:pc_num)];
% Calculate the regression coefficients;
Beta_cla = (X_mwpcr'*X_mwpcr)^-1*X_mwpcr'*Y_tr;

% Load the tune data;
X_tune = data.tune;
% Center the tune data;
X_tune = X_tune-repmat(mean(X_tune,1),size(X_tune,1),1);
% Project the tune using the weight matrix and the PC direction
X_mwpcr_tune = X_tune*Q*PC_loading;
% Create the design matrix for tuneing data;
X_mwpcr_tune = [ones(size(X_tune,1),1),X_mwpcr_tune(:,1:pc_num)];
% Compute the classification scores for the tune data;
Y_hat_tune = X_mwpcr_tune*Beta_cla;
% Threshold the scores to get the class labels;
Y_tune = (Y_hat_tune>cla_thres);
Y_tune_true = data.tune_y;
% Compute the classification accuracy;
Tune_ACC = sum(Y_tune==Y_tune_true)/length(Y_tune_true);
[~,~,~,Tune_AUC] = perfcurve(Y_tune_true, Y_hat_tune,1);



% Load the test data;
X_test = data.test_x;
% Center the test data;
X_test = X_test-repmat(mean(X_test,1),size(X_test,1),1);
% Project the test using the weight matrix and the PC direction
X_mwpcr_test = X_test*Q*PC_loading;
% Create the design matrix for testing data;
X_mwpcr_test = [ones(size(X_test,1),1),X_mwpcr_test(:,1:pc_num)];
% Compute the classification scores for the test data;
Y_hat_test = X_mwpcr_test*Beta_cla;
% Threshold the scores to get the class labels;
Y_test = (Y_hat_test>cla_thres);
Y_test_true = data.test_y;
% Compute the classification accuracy;
Test_ACC = sum(Y_test==Y_test_true)/length(Y_test_true);

MWPCR_coef_img = reshape(A*Q*PC_loading(:,1:tune_index)*Beta_cla(2:end),imagesize);
MWPCR_tune = Tune_ACC;
MWPCR_test = Test_ACC;
[~,~,~,MWPCR_tune_AUC] = perfcurve(Y_tune_true, Y_hat_tune,1);

%% Please specify the filename according your need.
filename = sprintf('/RealData/Res/MRI_%.2d_%.2d_%.2d_%.2d.mat',test_n,tune_n,lambda1_n,lambda2_n);
 save (filename, 'MWPCR_tune','MWPCR_test','MWPCR_tune_AUC')

