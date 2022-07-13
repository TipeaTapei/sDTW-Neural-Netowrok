function [val,exp_a,exp_b,exp_c] = soft_min_argmin(a,b,c)
% Computes the soft min and softargmin of (a, b, c) 

%%%%%%%%%%% INPUTS %%%%%%%%%%%%%
% a,b,c -> scalars

%%%%%%%%%%% OUTPUTS %%%%%%%%%%%%
% val -> softmin value
% transition probabilities

% It uses the log-sum-exp stabilization trick

min_abc = min([a,b,c]);

exp_a = exp(min_abc-a);
exp_b = exp(min_abc-b);
exp_c = exp(min_abc-c);

s = exp_a + exp_b + exp_c;

exp_a = exp_a/s;
exp_b = exp_b/s;
exp_c = exp_c/s;

val = min_abc - log(s);
end

