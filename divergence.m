function [divergence_value, divergence_grad] = divergence(y, t, gamma)
% Computes the sdtw divergence value and gradient wrt to y
% The cost is the squared euclidean

%%%%%%%%%%% INPUTS %%%%%%%%%%%%%
% y -> time_length_y x n_dim
% t -> time_length_t x n_dim
% gamma -> regularization parameter

%%%%%%%%%%% OUTPUTS %%%%%%%%%%%%
% divergence_value 
% divergence_grad
 
[R_yt,P_yt] = sdtw_D(y, t, gamma);
[R_yy,P_yy] = sdtw_D(y, y, gamma);
[R_tt,P_tt] = sdtw_D(t, t, gamma);

divergence_value = R_yt(length(R_yt),height(R_yt))-0.5*R_yy(length(R_yy),height(R_yy))-0.5*R_tt(length(R_tt),height(R_tt));

[~,grad_yt] = sdtw_grad_D(P_yt,y,t);
[~,grad_yy] = sdtw_grad_D(P_yy,y,y);
[~,grad_tt] = sdtw_grad_D(P_tt,t,t);

divergence_grad = grad_yt - grad_yy;

end

