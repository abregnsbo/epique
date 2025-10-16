function h = triangular(t)
% Piecewise-linear unit-area-like triangular impulse shape used as a target.
% Rise 0..1 over [0,1], fall to 0 over (1,1.5], then 0 afterwards.
% Usage: h = triangular(t_vector)

  h = zeros(size(t));
  idx1 = t<=1;
  h(idx1) = t(idx1);
  idx2 = t>1 & t<=1.5;
  h(idx2) = 3 - 2*t(idx2);
  % t>1.5 remains 0
end

