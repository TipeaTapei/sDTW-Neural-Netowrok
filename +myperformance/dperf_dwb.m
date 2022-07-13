function dperf = dperf_dwb(wb,param)
% Computes derivative of regularization performance. In case of no
% regularization, the function returns zero.

dperf = zeros(size(wb));
