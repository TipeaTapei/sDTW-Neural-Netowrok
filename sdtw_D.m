function [R,P] = sdtw_D(y, t, gamma)
% Computes soft dtw from time series y and t
% The cost is the squared euclidean one
 
y = y';
t = t';

%%%%%%%%%%% INPUTS %%%%%%%%%%%%%
% y -> time_length_y x n_dim (actually, our input is of n_dim x time_length, 
%                             thus I made a transposition)
% t -> time_length_t x n_dim
% gamma -> regularization strength

%%%%%%%%%%% OUTPUTS %%%%%%%%%%%%
% R -> cost accumulation matrix
% P -> transition probability matrix

n = height(y);   % time_length_y
m = height(t);   % time_length_t

% D is the distance matrix (cost matrix) -> n x m
D = pdist2(y,t,'squaredeuclidean');

% Handle regularization parameter gamma 
D = D/gamma;

% R -> cost accumulation matrix
R = zeros(n+1,m+1);

R(1,:)=1e10;    % To handle edge cases
R(:,1)=1e10;    % To handle edge cases
R(1,1)=0;

% P -> tensor containing transition probabilities
P = zeros(n+2,m+2,3);

for in=2:n+1
    for im=2:m+1
        [smin, P(in,im,1), P(in,im,2), P(in,im,3)] = soft_min_argmin(R(in,im-1),R(in-1,im-1),R(in-1,im));
        
        R(in,im)=D(in-1,im-1)+smin;
    end
end

 R = gamma*R;
 
end

