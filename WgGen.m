function Wg = WgGen(beta,beta_cov)
% Compute the global weight matrix.
% Input: 
% beta: estimation, a p by 1 vector.
% beta_cov: variance of beta, a p by 1 vector. 
% Output:
% Wg: diagonal of the global weight matrix, a p by 1 vector. 
pValue = 1-chi2cdf(beta(2).^2./beta_cov(2,2),1);
Wg = -log10(pValue+10^-100);