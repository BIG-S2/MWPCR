function [Q,Wg] = MSW_v1 (WE,beta,beta_cov,thres)
%% Global Weight Matrix
q = size(beta,2);
Wg = zeros(q,1);
for idx = 1:q
    Wg(idx) = WgGen(beta(:,idx),beta_cov{idx});
end
Wg_thres = quantile(Wg,thres);
Wg(Wg<Wg_thres) = 0;
WG = sparse(1:q,1:q,Wg);
Q = WG*WE;