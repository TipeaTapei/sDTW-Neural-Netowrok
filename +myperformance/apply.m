function perfs = apply(t,y,e,param)
% Calculates the performances for each target 
  gamma = 0.1;

  [perfs, ~] = divergence(y, t, gamma);
end
