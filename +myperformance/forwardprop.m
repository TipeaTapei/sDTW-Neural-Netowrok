function dperf = forwardprop(dy,t,y,e,param)
% Forward propagate derivatives to performance

  gamma = 0.1;

  [~, d] = divergence(y, t, gamma);
  d = d';
  
  dperf = bsxfun(@times,dy,d);
end
