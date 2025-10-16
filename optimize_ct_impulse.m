function optimize_ct_impulse(hfun)
% Optimize continuous-time poles/zeros (order 10) to match a triangular impulse response
% Does not work on Octave.
%
% Target impulse as defined by 'hfun'
% Time horizon T=2.5 captures tail. Fits a stable analog transfer function of order 'na'
% (denominator degree = na). Numerator degree nb < na
%
% Steps:
% 1) Build desired impulse h_d(t)
% 2) Frequency-domain initialization via invfreqs (if available), stabilized
% 3) Time-domain refinement with stability-constrained pole/zero parameterization using lsqnonlin or fminsearch
% 4) Report RMS error and plot

% ---------------------- User-configurable parameters ----------------------
na = 8;              % denominator order (pole count)
nb = na-1;               % numerator order (< na); set 0..na-1
T  = 3.5;             % seconds, horizon that safely captures tail
dt = 1e-3;            % time step for evaluation of objective (seconds)
Nw = 2048;            % frequency samples for initialization
maxIters = 200;       % nonlinear refinement iterations
init_method = 'fixedpole'; % 'fixedpole' or 'invfreqs'
fit_mode = 'varpro';       % 'varpro' (partial fractions, recommended) or 'zpk'
num_complex_pairs = floor(na/2);     % for varpro: number of complex pole pairs (2*nc + nr = na)
% Optional: emphasize early-time samples in objective (set w_time=[]) to disable
w_time = [];          % e.g., w_time = 1./sqrt(1 + ( (0:dt:T).'/0.5 ).^2 );

% Time grid and target impulse
t = (0:dt:T).';
if nargin < 1 || isempty(hfun)
  % Default target: triangular impulse defined in triangular.m
  hfun = @triangular; % function handle
end
h = hfun(t);
if isempty(w_time), w_time = ones(size(t)); end
w_time = w_time(:);

% ---------------------- Initialization ------------------------------------
switch lower(init_method)
  case 'invfreqs'
    fprintf('Initializing with invfreqs (if available)...\n');
    [b0,a0,hadInit] = init_invfreqs(t,h,nb,na,Nw);
    if ~hadInit
      fprintf('invfreqs not available; using fixed-pole LS initialization.\n');
      [b0,a0] = init_fixedpole_ls(t,h,nb,na,Nw);
    end
  otherwise
    fprintf('Initializing with fixed-pole linear LS...\n');
    [b0,a0] = init_fixedpole_ls(t,h,nb,na,Nw);
end

% Stabilize poles (reflect RHP to LHP)
pr = roots(a0);
pr(real(pr)>0) = complex(-abs(real(pr(real(pr)>0))), imag(pr(real(pr)>0)));
a0 = real(poly(pr));

% Evaluate initial RMS in time domain
sys0 = tf(b0,a0);
y0 = impulse_ct(sys0,t);
rms0 = norm((h - y0).*w_time) / sqrt(numel(h));
fprintf('Initial RMS: %.4e\n', rms0);

% ---------------------- Time-domain refinement ----------------------------
fprintf('Refining via time-domain least-squares...\n');
use_lsqnonlin = exist('lsqnonlin','file')==2 || exist('lsqnonlin','builtin')==5; %#ok<*EXIST>

switch lower(fit_mode)
  case 'varpro'
    % Variable-projection on partial fractions: optimize poles, solve residues LS
    [p_opt, c_opt, y] = varpro_fit(t, h, na, num_complex_pairs, w_time, maxIters, use_lsqnonlin);
    % Convert (poles,residues) to polynomial TF
    a = real(poly(p_opt.'));
    b = residues_to_num(c_opt, p_opt, a);
    sys = tf(b, a);
    y = impulse_ct(sys, t); % evaluate with TF for consistency
  otherwise
    % Original zpk parameterization refinement
    [z0,p0,k0] = tf2zpk(b0,a0);
    tpl = build_template(z0,p0,nb,na,true); % enforce minimum-phase zeros (LHP)
    theta0 = pack_params(z0,p0,k0,tpl);
    if use_lsqnonlin
      try
        opts = optimoptions('lsqnonlin','Display','iter','MaxFunEvals',5e4,'MaxIter',maxIters);
      catch
        opts = struct('Display','iter','MaxIter',maxIters);
      end
      obj = @(th) time_residual(th, tpl, t, h, w_time);
      try
        theta = lsqnonlin(obj, theta0, [], [], opts);
      catch
        obj2 = @(th) sum(abs(obj(th)).^2);
        theta = fminsearch(obj2, theta0);
      end
    else
      obj = @(th) time_residual(th, tpl, t, h, w_time);
      obj2 = @(th) sum(abs(obj(th)).^2);
      theta = fminsearch(obj2, theta0);
    end
    [z,p,k] = unpack_params(theta, tpl);
    sys = zpk(z,p,k);
    y  = impulse_ct(sys,t);
    [b,a] = tfdata(sys,'v');
end

rms1 = norm((h - y).*w_time) / sqrt(numel(h));
fprintf('Refined RMS: %.4e\n', rms1);

% ------------------------------ Plots -------------------------------------
figure; 
subplot(2,1,1);
plot(t,h,'k-', 'LineWidth',1.5); hold on;
plot(t,y0,'r--', 'LineWidth',1.0);
plot(t,y,'b-', 'LineWidth',1.2);
grid on; xlabel('t [s]'); ylabel('h(t)');
legend('target','init','refined'); title(sprintf('Impulse fit (na=%d, nb=%d)  RMS: init %.2e  refined %.2e',na,nb,rms0,rms1));

subplot(2,1,2);
[m,pw,w] = bode_mag(sys, 512);
semilogx(w, 20*log10(max(m,1e-12)),'b'); hold on; grid on;
xlabel('\omega [rad/s]'); ylabel('|H(j\omega)| [dB]'); title('Refined magnitude response');

% Print results
fprintf('\nFinal transfer function coefficients (descending powers of s):\n');
disp('Numerator b:'); disp(b);
disp('Denominator a:'); disp(a);

end % function optimize_ct_impulse

% --------------------------- Helper functions -----------------------------
% (desired_impulse moved to triangular.m)

function [b,a,ok] = init_invfreqs(t,h,nb,na,Nw)
  ok = false; b = []; a = [];
  has_invfreqs = exist('invfreqs','file')==2 || exist('invfreqs','builtin')==5;
  if ~has_invfreqs, return; end
  % Frequency grid
  T = t(end); dt = t(2)-t(1);
  wmin = 2*pi/max(T,eps);
  wmax = min(pi/dt*0.8, 200*(2*pi/T));
  w = logspace(log10(max(wmin,1e-1)), log10(max(wmax,1e1)), Nw);
  % Numerical Fourier transform H(jw) = ∫ h(t) e^{-j w t} dt
  E = exp(-1j*(w(:) * t.'));    % [Nw x Nt]
  F = E .* (ones(numel(w),1) * h.');
  H = trapz(t, F, 2);           % [Nw x 1]
  % Weights (emphasize low-mid band)
  w0 = 2*pi/T; w1 = min(10*w0, max(w));
  W = 1./sqrt(1 + (w(:)/w1).^2);
  try
    [b,a] = invfreqs(H, w, nb, na, W, 50);
    ok = true;
  catch
    try
      [b,a] = invfreqs(H, w, nb, na);
      ok = true;
    catch
      ok = false;
    end
  end
end

function [p_opt, c_opt, y] = varpro_fit(t, h, na, num_complex_pairs, w_time, maxIters, use_lsqnonlin)
  % Variable projection over residues; optimize only poles
  if isempty(w_time), w_time = ones(size(t)); end
  w = w_time(:);
  % Pole structure
  nc = min(num_complex_pairs, floor(na/2));
  nr = na - 2*nc;
  % Initial pole params: spread across [2/T, 150/T]
  T = t(end);
  rates = logspace(log10(2/T), log10(150/T), na).';
  alphas = log(rates(1:nc));
  betas  = log(0.7*rates(1:nc));
  r_alphas = log(rates(nc+1:nc+nr));
  theta0 = [alphas; betas; r_alphas];
  % Objective returning weighted residual vector
  function [res, Phi, coeffs] = obj_fun(th)
    [p, Phi] = poles_and_basis(th, nc, nr, t);
    % Weighted LS for coefficients
    WPhi = bsxfun(@times, sqrt(w), Phi);
    Wh  = sqrt(w) .* h;
    coeffs = WPhi \ Wh;
    yhat = Phi * coeffs;
    res = sqrt(w) .* (yhat - h);
  end
  % Optimize
  if use_lsqnonlin
    try
      opts = optimoptions('lsqnonlin','Display','iter','MaxIter',maxIters,'MaxFunEvals',5e4);
    catch
      opts = struct('Display','iter','MaxIter',maxIters);
    end
    wrapper = @(th) obj_fun(th);
    theta = lsqnonlin(wrapper, theta0, [], [], opts);
  else
    % fminsearch on sum of squares
    f2 = @(th) sum(abs(obj_fun(th)).^2);
    theta = fminsearch(f2, theta0);
  end
  % Final coefficients with optimized poles
  [p_opt, Phi] = poles_and_basis(theta, nc, nr, t);
  WPhi = bsxfun(@times, sqrt(w), Phi);
  Wh  = sqrt(w) .* h;
  c_all = WPhi \ Wh;
  % Map coefficients to residues
  c_opt = coeffs_to_residues(c_all, p_opt, nc, nr);
  y = Phi * c_all;
end

function [p, Phi] = poles_and_basis(th, nc, nr, t)
  % Decode parameters and build real-valued basis Phi
  idx = 1;
  alphas = th(idx:idx+nc-1); idx = idx+nc;
  betas  = th(idx:idx+nc-1); idx = idx+nc;
  r_alphas = th(idx:idx+nr-1);
  % Poles
  sigmas = -exp(alphas);
  omegas = exp(betas);
  r_sigmas = -exp(r_alphas);
  p = [];
  for k=1:nc
    p = [p; sigmas(k)+1j*omegas(k); sigmas(k)-1j*omegas(k)]; %#ok<AGROW>
  end
  p = [p; r_sigmas(:)];
  % Basis: for each complex pair -> e^{σt}cos(ωt), e^{σt}sin(ωt);
  % for each real pole -> e^{σt}
  N = numel(t);
  Phi = zeros(N, 2*nc + nr);
  col = 1;
  for k=1:nc
    et = exp(sigmas(k)*t);
    Phi(:,col)   = et .* cos(omegas(k)*t); col = col+1;
    Phi(:,col)   = et .* sin(omegas(k)*t); col = col+1;
  end
  for k=1:nr
    Phi(:,col) = exp(r_sigmas(k)*t); col = col+1;
  end
end

function r = coeffs_to_residues(c_all, p, nc, nr)
  % Map real-valued coefficients back to complex residues matching poles p
  % c_all layout: [2*nc columns for cos/sin, then nr real poles]
  r = zeros(size(p));
  col = 1; idx = 1;
  for k=1:nc
    ccos = c_all(col); csin = c_all(col+1); col = col+2;
    a = ccos/2; b = -csin/2; % from 2*Re{r e^{(σ+jω)t}}
    r(idx) = a + 1j*b; r(idx+1) = a - 1j*b; idx = idx+2;
  end
  for k=1:nr
    r(idx) = c_all(col); col = col+1; idx = idx+1;
  end
end

function b = residues_to_num(r, p, a)
  % Given residues r, poles p, and denominator a (poly with roots p), build numerator
  na = numel(a)-1; % degree
  b = zeros(1, na);
  for i=1:numel(p)
    Ai = deconv(a, [1, -p(i)]); % degree na-1
    b = b + real(r(i)) * real(Ai) - imag(r(i)) * imag(Ai); % keep real; deconv returns complex if p complex
  end
end

function [b,a] = init_fixedpole_ls(t,h,nb,na,Nw)
  % Robust initializer: choose stable poles, then solve linear LS for numerator
  T = t(end); dt = t(2)-t(1);
  wmin = 2*pi/max(T,eps);
  wmax = min(pi/dt*0.8, 300*(2*pi/T));
  w = logspace(log10(max(wmin,1e-1)), log10(max(wmax,1e1)), Nw).';
  % Numerical Fourier transform H(jw) = ∫ h(t) e^{-j w t} dt
  E = exp(-1j*(w * t.'));
  H = trapz(t, E .* (ones(numel(w),1) * h.'), 2);
  % Select pole set: mix of real poles and complex pairs over [2/T, 150/T]
  rates = logspace(log10(2/T), log10(150/T), na).';
  p = zeros(na,1);
  m = floor(na/2);
  % m complex pairs + remaining real poles
  for i=1:m
    sigma = -rates(i);
    omega = 0.7*rates(i);
    p(2*i-1:2*i) = [sigma+1j*omega; sigma-1j*omega];
  end
  for i=2*m+1:na
    p(i) = -rates(i);
  end
  a = real(poly(p.'));
  % Build linear system for numerator coefficients b (degree nb)
  s = 1j*w;
  Aeval = polyval(a, s);
  y = H .* Aeval; % desired: B(s) ≈ H(s)*A(s)
  X = zeros(numel(w), nb+1);
  for k=0:nb
    X(:,k+1) = s.^(nb-k);
  end
  % Frequency weights to balance band
  wc = 10*(2*pi/T);
  W = 1./sqrt(1 + (w/wc).^2);
  Xw = bsxfun(@times, W, X);
  yw = W .* y;
  % Solve complex LS; enforce real coefficients
  b = Xw\yw;
  b = real(b(:)).';
  % Optional: rescale to avoid large coefficients
  b = b / max(1, norm(b,2));
end

function tpl = build_template(z,p,nb,na,enforceMinPhaseZeros)
  % Fix the structure (#real vs complex pairs) from the initializer
  z = z(:); p = p(:);
  % Trim/expand zeros to requested order (keep leading entries)
  if numel(z) > nb, z = z(1:nb); end
  if numel(z) < nb
    z = [z; -ones(nb-numel(z),1)];
  end
  % Ensure exactly na poles (truncate if needed)
  if numel(p) > na, p = p(1:na); end
  % If fewer, pad with stable real poles
  if numel(p) < na
    p = [p; -linspace(2,20,na-numel(p)).'];
  end
  % Separate real and complex (use only one of each pair with +imag)
  tol = 1e-12;
  prR = p(abs(imag(p))<=tol);
  prCfull = p(abs(imag(p))>tol);
  prC = prCfull(imag(prCfull)>0);
  zrR = z(abs(imag(z))<=tol);
  zrCfull = z(abs(imag(z))>tol);
  zrC = zrCfull(imag(zrCfull)>0);
  tpl.na = na; tpl.nb = nb;
  tpl.prR = prR(:).'; tpl.prC = prC(:).';
  tpl.zrR = zrR(:).'; tpl.zrC = zrC(:).';
  tpl.enforceMinPhaseZeros = logical(enforceMinPhaseZeros);
end

function th = pack_params(z,p,k,tpl)
  tol = 1e-12; th = [];
  % Real poles: p = -exp(alpha)
  for x = tpl.prR
    th(end+1) = log(max(tol, -real(x)));
  end
  % Complex poles: p = -exp(alpha) ± j exp(beta)
  for x = tpl.prC
    th(end+1) = log(max(tol, -real(x)));
    th(end+1) = log(max(tol, abs(imag(x))));
  end
  % Real zeros (min-phase): z = -exp(alpha)
  for x = tpl.zrR
    th(end+1) = log(max(tol, -real(x)));
  end
  % Complex zero pairs: z = -exp(alpha) ± j exp(beta)
  for x = tpl.zrC
    th(end+1) = log(max(tol, -real(x)));
    th(end+1) = log(max(tol, abs(imag(x))));
  end
  % Gain (free real scalar)
  if nargin<3 || isempty(k), k = 1; end
  th(end+1) = real(k);
  th = th(:);
end

function [z,p,k] = unpack_params(th, tpl)
  idx = 1; tol = 1e-12; p = [];
  % Real poles
  for i=1:numel(tpl.prR)
    alpha = th(idx); idx=idx+1;
    p(end+1,1) = -exp(alpha);
  end
  % Complex poles
  for i=1:numel(tpl.prC)
    alpha = th(idx); beta = th(idx+1); idx=idx+2;
    sigma = -exp(alpha); omega = exp(beta);
    p = [p; sigma+1j*omega; sigma-1j*omega]; %#ok<AGROW>
  end
  % Zeros
  z = [];
  if tpl.enforceMinPhaseZeros
    for i=1:numel(tpl.zrR)
      alpha = th(idx); idx=idx+1; z(end+1,1) = -exp(alpha);
    end
    for i=1:numel(tpl.zrC)
      alpha = th(idx); beta = th(idx+1); idx=idx+2;
      sigma = -exp(alpha); omega = exp(beta);
      z = [z; sigma+1j*omega; sigma-1j*omega]; %#ok<AGROW>
    end
  else
    % If not enforcing, you could use free real parts; omitted for brevity
    for i=1:numel(tpl.zrR)
      alpha = th(idx); idx=idx+1; z(end+1,1) = -exp(alpha);
    end
    for i=1:numel(tpl.zrC)
      alpha = th(idx); beta = th(idx+1); idx=idx+2;
      sigma = -exp(alpha); omega = exp(beta);
      z = [z; sigma+1j*omega; sigma-1j*omega]; %#ok<AGROW>
    end
  end
  % Gain
  if idx <= numel(th)
    k = th(idx);
  else
    k = 1.0;
  end
  % Ensure counts match template lengths
  if numel(p) > tpl.na, p = p(1:tpl.na); end
  if numel(z) > tpl.nb, z = z(1:tpl.nb); end
  % Force real coefficients by conjugate pairs already handled
  % Small numerical jitter fix
  p(abs(imag(p))<tol) = real(p(abs(imag(p))<tol));
  z(abs(imag(z))<tol) = real(z(abs(imag(z))<tol));
end

function e = time_residual(th, tpl, t, h, w_time)
  [z,p,k] = unpack_params(th, tpl);
  sys = zpk(z,p,k);
  y = impulse_ct(sys, t);
  e = (y - h) .* w_time;
end

function y = impulse_ct(sys, t)
  % Robust impulse response for continuous-time TF in MATLAB/Octave.
  % Prefer impulse(); fallback to derivative of step().
  y = [];
  try
    y = impulse(sys, t);
    if isstruct(y) % older Octave may return struct
      y = y.y;
    end
    y = y(:);
    if numel(y)==numel(t), return; end
  catch
  end
  % Fallback using numerical derivative of step response
  try
    s = step(sys, t);
    if isstruct(s), s = s.y; end
    s = s(:);
    % Use gradient to keep same length
    y = gradient(s, t);
  catch
    % Last resort: simulate with a narrow pulse
    dt = t(2)-t(1);
    u = zeros(size(t)); u(1) = 1/dt; % area=1
    y = lsim(sys, u, t);
    if isstruct(y), y = y.y; end
    y = y(:);
  end
end

function [mag,phs,w] = bode_mag(sys, N)
  % Lightweight analog magnitude plot helper
  if nargin<2, N=512; end
  % Build frequency grid based on pole/zero spread
  [b,a] = tfdata(sys,'v');
  p = roots(a); z = roots(b);
  reals = [-real(p(:)); -real(z(:))]; reals = reals(reals>0);
  if isempty(reals), w0 = 1; else, w0 = quantile(reals,[0.2 0.8]); end
  if numel(w0)==2
    wmin = max(1e-1, 0.1*w0(1)); wmax = 10*w0(2);
  else
    wmin = 1e-1; wmax = 1e3;
  end
  w = logspace(log10(wmin), log10(wmax), N);
  s = 1j*w;
  % Evaluate H(s) = B(s)/A(s)
  num = polyval(b, s);
  den = polyval(a, s);
  H = num ./ den;
  mag = abs(H(:));
  phs = angle(H(:)); %#ok<NASGU>
end
