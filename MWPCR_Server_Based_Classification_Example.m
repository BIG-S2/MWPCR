%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%================MWPCR_Server_Based_Classification_Example================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars;
close all;
clc;
addpath NIfTI


%% Loading the data
% Please modify the data file path and name accordingly for your own job.
load ('SIMdata.mat') % This is the simulation data from the paper;

% Please use the mask file according your own data.
mask = ones(20,20,10);
imagesize = size(mask); % the image size must match the mask size.


%% Load the data, this is an example for 3-way split data cross validation
data = SIM1;
% load the training data;
X_tr = data.train;
Y_tr = [ones(60,1);zeros(40,1)];
% Load the tuning data;
X_tune = data.tune;
% Center the tuning data;
X_tune = X_tune-repmat(mean(X_tune,1),size(X_tune,1),1);
Y_tune = [ones(60,1);zeros(40,1)];
% Load the testing data;
X_test = data.test;
% Center the testing data;
X_test = X_test-repmat(mean(X_test,1),size(X_test,1),1);
Y_test = [ones(60,1);zeros(40,1)];

[n,p] = size(X_tr);
% Compute the threshold for classification
prob_thres = sum(Y_tr)/n;

%% 2D penal cross validation on a classification analysis
% In this example, we tune the model by selecting different sparsity level
% and the number of PC components.

Sparsity_tune_n = 21; % number of sparsity level being tuned.
PC_num_tune_n = 21; % number of PC's being tuned.

Thres_LB = 0.9; % the lower bound of sparsity level, between 0 and 1;
Thres_UB = 0.99; % the upper bound of sparsity level, between 0 and 1;
thres_pool = linspace(Thres_LB,Thres_UB,Sparsity_tune_n);


%======================Do Not Edit======================%
% Assign the parameters according to your image types.
% h is the base searching radius, we recommend h = 1.5 for 1D; 1.28 for 2D; 1.2 for 3D;
% S is the max power of radius, we recommend S = 3 for 1D; 4 for 2D; 5 for 3D;

dim = length(imagesize);
switch dim
    case 1
        h = 1.5; S = 3;
    case 2
        h = 1.28; S = 4;
    case 3
        h = 1.2; S = 5;
end
%% Compute the head file, it contains all the neighborhood information.
[Mask_Idx, Mask_Loc,Nbr_Dist_2] = Head_File_For_Mask_MWPCR(mask,h^S);

%% Compute the multi-scale adpative coefficients.
[beta,beta_cov,WE]  = MARM (Mask_Idx,Nbr_Dist_2,X_tr,Y_tr,h,S);

%% Begin cross validation analysis
fprintf('+++++++Cross validation procedure started!!+++++++\n');
Tune_ACC = zeros(Sparsity_tune_n,PC_num_tune_n);
Tune_AUC = zeros(Sparsity_tune_n,PC_num_tune_n);

% outter loop is for tuning the sparsity level
for j =1:Sparsity_tune_n
    thres = thres_pool(j); % this is a threshold parameter, comtrol the sparseness of the estimation.
    [Q,~] = MSW (WE,beta,beta_cov,thres);
    %% Project the data using the weight matrix Q.
    X_t = (X_tr-repmat(mean(X_tr,1),n,1))*Q; % X_t is X_tilde in the paper
    
    %% PCA analysis on the projected data
    [U,D,V] =   svd(X_t, 'econ');
    PC_score = U*D; % PC scores;
    PC_loading = V; % PC loadings;
    % inner loop is for tuning the number of pricinple components
    for i = 1:PC_num_tune_n
        PC_num = i;
        % Create the design matrix with the PC scores;
        X_mwpcr = [ones(n,1),PC_score(:,1:PC_num)];
        % Calculate the regression coefficients;
        Beta_cla = (X_mwpcr'*X_mwpcr)^-1*X_mwpcr'*Y_tr;
        % Project the tuning data using the weight matrix and the PC direction
        X_mwpcr_tune = X_tune*Q*PC_loading;
        % Create the design matrix for tuning data;
        X_mwpcr_tune = [ones(size(X_tune,1),1),X_mwpcr_tune(:,1:PC_num)];
        % Compute the classification scores for the tuning data;
        Y_hat_tune = X_mwpcr_tune*Beta_cla;
        % Threshold the scores to get the class labels;
        Y_tune_predict = (Y_hat_tune>prob_thres);
        % Compute the classification accuracy;
        Tune_ACC(i,j) = sum(Y_tune_predict==Y_tune)/length(Y_tune);
        [~,~,~,Tune_AUC(i,j)] = perfcurve(Y_tune, Y_hat_tune,1);
        fprintf('+++++++%2.2f%% of the job is done!!+++++++\n', ((j-1)*Sparsity_tune_n+i)/(Sparsity_tune_n*PC_num_tune_n)*100);
    end
end

%% Tuning parameter selection procedure
% First criteria: based on tuning accuracy
max_index = find(Tune_ACC==max(Tune_ACC(:)));
% If tied, use second criteria: based on tuning AUC
if length(max_index)>1
    max_index_index = find(Tune_AUC(max_index)==max(Tune_AUC(max_index)));
    max_index = max_index(max_index_index);
end
[PC_num_max, thres_max] = ind2sub([Sparsity_tune_n,PC_num_tune_n],max_index);
% if tied, use third criteria: fewer principle components and higher
% sparsity level
PC_num_index = find(PC_num_max == min(PC_num_max));
PC_num_max = min(PC_num_max);
thres_max = max(thres_max(PC_num_index));


%% Compute the testing results, based on the selected model
thres = thres_pool(thres_max);
PC_num = PC_num_max;
% Retrain the model using the slected parameter
[Q,Wg] = MSW (WE,beta,beta_cov,thres);
X_t = (X_tr-repmat(mean(X_tr,1),n,1))*Q;
[U,D,V] =   svd(X_t, 'econ');
PC_score = U*D;
PC_loading = V;
X_mwpcr = [ones(n,1),PC_score(:,1:PC_num)];
Beta_cla = (X_mwpcr'*X_mwpcr)^-1*X_mwpcr'*Y_tr;
% Final model is retrained and ready for prediction

% Project the test using the weight matrix and the PC direction
X_mwpcr_test = X_test*Q*PC_loading;
% Create the design matrix for testing data;
X_mwpcr_test = [ones(size(X_test,1),1),X_mwpcr_test(:,1:PC_num)];
% Compute the classification scores for the testing data;
Y_hat_test = X_mwpcr_test*Beta_cla;
% Threshold the scores to get the class labels;
Y_test_predicted = (Y_hat_test>prob_thres);
% Compute the classification accuracy;
Test_ACC = sum(Y_test_predicted==Y_test)/length(Y_test);

MWPCR_coef = reshape(Q*PC_loading(:,1:PC_num_max)*Beta_cla(2:end),imagesize);
MWPCR_tune = Tune_ACC;
MWPCR_test = Test_ACC;

fprintf('+++++++Cross validation procedure is finished!!+++++++\n');
%====================================================%

%===================Display the results=====================%
%% Display the final testing accuracy
fprintf('+++++++The final testing accuracy is %.2f+++++++\n', Test_ACC)
%%  Log_10 P-value plot
% For 1 D images: use plot(Wg);
% For 2D/3D images: reshape Wg first:
Wg = reshape(Wg,imagesize);
view_nii(make_nii(Wg)); % This is a 3-D view of the important score matrix.
%% ROC curves plot
MWPCR_AUC = ROC_analysis(Y_test, Y_hat_test,1,'MWPCR'); % display the ROC curves
%% Display the estimated coefficient image
nii_view(MWPCR_coef)
%====================================================%

