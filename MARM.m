function [beta,beta_cov,WE] = MARM (Mask_Idx,Nbr_Dist_2,imgs,labels,h,S)
% INPUT:
% mask is a 3D 0/1 array.
% imgs is a n*p masked image data.
% h is the bandwidth, 1<h.
% S is the largest power of h, S<=5.
% OUTPUT:
% beta_opt is the optimal beta estimation from MARM.
% beta is all the estimation in different level.
% beta_opt_cov is the variance of beta_opt.
% beta_cov is the variance estimation of beta.
% S_opt is the optimal scale size (power).
% WES is the optimal beta estimation from MARM.

%% Prepare for MARM
[n, q]  = size(imgs);
gamma = log(n)*chi2inv(.95,1);

%% Initial estimation without neighborhood infomation
X = [ones(n,1),labels];
XTXI = (X'*X)^-1;
XTX = X'*X;
Y = imgs;
beta = XTXI*X'*Y;

var_0 = sum((Y - X*beta).^2,1)./(n-2);
beta_cov = cell(q,1);
beta_cov_inv = cell(q,1);
for idx = 1:q
    beta_cov{idx} = var_0(idx).*XTXI;
    beta_cov_inv{idx} = XTX./var_0(idx);
end

%% MARM
for s = 1:S
    %% Voxel-wise MARM
    [Y,W,WE,Mask_Idx_hs] = MARM_gen (Mask_Idx,Nbr_Dist_2,imgs,beta,...
        beta_cov_inv,var_0,h,s,gamma);
    [beta, beta_cov,beta_cov_inv] = MARM_update (Y,X,XTXI,W,Mask_Idx_hs);
end
