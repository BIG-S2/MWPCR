function [beta, beta_cov,beta_cov_inv] = MARM_update (Y,X,XTXI,W,Mask_Idx_hs)

[q,hs_1,n] = size(Y);
W_S = sum(W,2);
W_SI = W_S'.^-1;
Wn = repmat(W,[1,1,n]);
W_Y = squeeze(sum(Wn.*Y,2))'; % n by q matrix
XTXIXTW_Y = XTXI*X'*W_Y;
beta = repmat(W_SI,2,1).*XTXIXTW_Y;
% Estimate the Covariance part
Beta0_0 = [beta(1,:),0];
Beta1_0 = [beta(2,:),0];
Beta0_Nbr = Beta0_0(Mask_Idx_hs);
Beta1_Nbr = Beta1_0(Mask_Idx_hs);
Y_0 = repmat(Beta0_Nbr,[1,1,n]);
X_1 = X(:,2);
X_1 = repmat(X_1,[1,hs_1,q]);
X_1 = permute(X_1,[3,2,1]);
Beta1_Nbr = repmat(Beta1_Nbr,[1,1,n]);
Y_1 = X_1.*Beta1_Nbr;
Err = Y-Y_0-Y_1;
Err = squeeze(sum(Err,2))';
Err_2 = Err.^2;
beta_cov = cell(q,1);
beta_cov_inv = cell(q,1);
for i = 1:q
    cov_temp = W_SI(i)^2*XTXI*X'*diag(Err_2(:,i))*X*XTXI;
    beta_cov{i} = cov_temp;
    beta_cov_inv{i} = cov_temp^-1;
end