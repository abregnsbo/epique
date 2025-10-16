function h = soft_exp_decay(t, a, decayDur)
% soft_exp_decay  Exponential rise (0..1) then smooth polynomial decay.
%   This preserves the original core shape as soft_exp_decay_core, then
%   computes its maximum (x_max, y_max) and applies linear input/output
%   scaling so that soft_exp_decay(1.0) = 1.0.
%
%   h = soft_exp_decay(t) uses default a=2 and decayDur=0.8.
%   h = soft_exp_decay(t, a) sets rise steepness a>0 and decayDur=0.8.
%   h = soft_exp_decay(t, a, decayDur) sets decay duration D>0 so the
%   core falls from 1 at t=1 to 0 at t=1+D (before scaling).

  if nargin < 2 || isempty(a), a = 2; end
  if nargin < 3 || isempty(decayDur), decayDur = 0.8; end
  D = max(decayDur, eps);

  % Columnize input for processing
  tshape = size(t);
  t = t(:);

  % Core function handle (vectorized)
  core = @(x) soft_exp_decay_core(x, a, D);

  % Find maximum of the core on [0, 1+D]
  xlo = 0; xhi = 1 + D;
  x_max = NaN; y_max = NaN;
  try
    % Use fminbnd on negative core to locate maximum
    fneg = @(x) -soft_exp_decay_core_scalar(x, a, D);
    x_max = fminbnd(fneg, xlo, xhi);
    y_max = soft_exp_decay_core_scalar(x_max, a, D);
  catch
    % Fallback: dense sampling
    xs = linspace(xlo, xhi, 2001);
    ys = core(xs);
    [y_max, idx] = max(ys);
    x_max = xs(idx);
  end
  if ~(isfinite(x_max) && isfinite(y_max) && y_max > 0)
    % Robust fallback if something went wrong
    x_max = 1; y_max = max(1, soft_exp_decay_core_scalar(1, a, D));
  end

  % Linear input/output scaling: map core(x_max)=y_max to soft(1)=1
  ts = x_max * t;               % input scaling so t=1 -> x_max
  h = (1 / y_max) * core(ts);   % output scaling so value at t=1 is 1

  % Restore original shape
  h = reshape(h, tshape);
end

% ---- Core definition preserved (no scaling). Vectorized in x. ----
function h = soft_exp_decay_core(x, a, D)
  x = x(:);
  h = zeros(size(x));

  m0 = (x >= 0 & x <= 1);
  m1 = (x > 1 & x <= 1 + D);

  % Rise: exponential normalized to reach 1 at x=1
  if any(m0)
    xm = x(m0);
    denom = exp(a) - 1;
    if abs(denom) < eps
      hr = xm; % linear fallback
    else
      hr = (exp(a*xm) - 1) / denom;
    end
    h(m0) = hr;
  end

  % Decay: quintic polynomial matching value/slope/curvature at x=1
  if any(m1)
    denom = exp(a) - 1;
    if abs(denom) < eps
      s1 = 1; s2 = 0;
    else
      s1 = a*exp(a) / denom;      % f'(1)
      s2 = a^2*exp(a) / denom;    % f''(1)
    end
    % Polynomial P(u) on u in [0,1], u=(x-1)/D
    c0 = 1;
    c1 = s1 * D;
    c2 = (s2 * D^2) / 2; % since P''(0)=2*c2
    r1 = -(c0 + c1 + c2);
    r2 = -(c1 + 2*c2);
    r3 = -(2*c2);
    M = [1 1 1; 3 4 5; 6 12 20];
    c345 = M \ [r1; r2; r3];
    c3 = c345(1); c4 = c345(2); c5 = c345(3);

    xm = x(m1);
    u = (xm - 1) / D; u = max(0, min(1, u));
    h(m1) = polyval([c5 c4 c3 c2 c1 c0], u);
  end
end

% Scalar wrapper for fminbnd
function y = soft_exp_decay_core_scalar(x, a, D)
  y = soft_exp_decay_core(x, a, D);
  if numel(y) ~= 1
    y = y(1);
  end
end
