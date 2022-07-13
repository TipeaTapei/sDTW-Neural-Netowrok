function dy = backprop(t,y,e,param)
% Backpropagates derivatives of performance
% Return dperf/dy 

% gamma can vary in ]0,1]
  gamma = 0.1;

  [~, dy] = divergence(y, t, gamma);
  dy = dy';
end
