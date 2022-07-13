function [E,grad] = sdtw_grad_D(P,y,t)
% Computes the gradient of sdtw wrt y
% The cost is the squared euclidean

y = y';
t = t';

%%%%%%%%%%% INPUTS %%%%%%%%%%%%%
% y -> time_length_y x n_dim
% t -> time_length_t x n_dim
% P -> transition probability matrix

%%%%%%%%%%% OUTPUTS %%%%%%%%%%%%
% E -> Expected alignment matrix under Gibbs distribution - equal to the
%      gradient wrt D
% grad -> gradient wrt y

E = zeros(height(P),length(P));

E(height(P),:)=0;
E(:,length(P))=0;
E(height(P),length(P))=1;

P(height(P),length(P),:)=1;

for im=length(P)-1:-1:2
    for in=height(P)-1:-1:2
        E(in,im)=P(in,im+1,1)*E(in,im+1)+P(in+1,im+1,2)*E(in+1,im+1)+P(in+1,im,3)*E(in+1,im);
    end
end

E=E(2:height(P)-1,2:length(P)-1);

e=sum(E,2);
grad=y.*e;
grad=grad-E*t;

end

