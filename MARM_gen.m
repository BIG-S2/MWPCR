function [Y,W,WE,Mask_Idx_hs] = MARM_gen (Mask_Idx,Nbr_Dist_2,imgs,beta_all,...
    beta_cov_inv,var_0,h,s,gamma)
% Generate the input for WLSE in MARM;
% Input:
% mask_h: head file based on the mask matrix,
% i.e. [mask_h] = head_mask_SWPCA(mask,h);
% dictionary: identify the colomns of mask_h
% imgs: n by p imaging data (masked)
% beta_all: 2 by p matrix, each column is the estimation for idx p
% beta_cov_inv: 1 by p cell, with element of 2 by 2, inverse of the beta_cov.
% var_0: 1 by p vector, sigma2 from SLR at each idx.
% h: neighborhood radial base.
% s: neighborhood radial power.
% gamma: parameter for K2. gamma=log(n)*chi2inv(0.95,2);
% Output:
% Y: q by hs+1 by n array with the 1st column being the index point and
% rest bing the neighborhood. Each layer being a subject.
% W: q by hs+1 matrix, weight/var_0, at idx and nbr of idx;
% WE: q by q, local weight matrix.

[n,q] = size(imgs);
imgs_0 = [imgs,zeros(n,1)];

nbridxcol = find(Nbr_Dist_2<=h^(2*s))+1; % columns of nbr index(masked) in head file
hs = length(nbridxcol);
Mask_Idx_hs = Mask_Idx(:,[1,nbridxcol]);
Index_0 = Mask_Idx_hs==0;
Mask_Idx_hs(Index_0)=q+1;
Mask_Idx_hs(:,1)=1:q;

Y =  imgs_0(:,Mask_Idx_hs);
Y = reshape(Y',q,hs+1,n);


d1_sqr =  Nbr_Dist_2(nbridxcol-1) ; % Distance^2 between point and nbrs.
k1 = 1-d1_sqr.^.5/h^s;

Beta0_0 = [beta_all(1,:),0];
Beta1_0 = [beta_all(2,:),0];
Beta0_Nbr = Beta0_0(Mask_Idx_hs(:,2:end));
Beta1_Nbr = Beta1_0(Mask_Idx_hs(:,2:end));
Beta0_diff = Beta0_Nbr - repmat(beta_all(1,:)',1,hs);
Beta1_diff = Beta1_Nbr - repmat(beta_all(2,:)',1,hs);
k2 = zeros(q,hs);
for i = 1:q
    for j = 1:hs
        beta_dist = [Beta0_diff(i,j);Beta1_diff(i,j)];
        d2 = beta_dist'*beta_cov_inv{i}*beta_dist;
        k2(i,j) = exp(-d2/gamma);
    end
end



var_0_0 = [var_0,0];
var_0_nbr = var_0_0(Mask_Idx_hs);
w = repmat(k1,q,1).*k2;
w = [ones(q,1),w];
w(Index_0)=0;
W = w./var_0_nbr;
W(Index_0)=0;

Index_x = repmat([1:q]',hs+1,1);
Index_y = Mask_Idx_hs(:);
values = w(:);
WE = sparse(Index_x, Index_y, values, q, q +1);
WE(:,q+1) = [];