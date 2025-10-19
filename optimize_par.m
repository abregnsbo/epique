function sys = optimize_par(hfun, opts)
% Fit two continuous-time LHP, minimum-phase rational models in parallel
% with an adjustable ideal delay on the second branch:
%   H(s) = H1(s) + e^{-Td s} H2(s)
% Both H1 and H2 have order na=6 (poles) and nb=5 (zeros), with all poles
% and zeros constrained to the left-half plane. Td >= 0 is optimized.
%
% Usage:
%   sys = optimize_par(@target_fun)
%   sys = optimize_par(@target_fun, opts)
%     where target_fun is a function handle: h = target_fun(t)
%
% Options (all optional; pass in struct 'opts'):
%   T                 [3.5]   time horizon (seconds)
%   dt                [1e-3]  time step (seconds)
%   num_starts        [12]    multistart count (global search)
%   local_maxiter     [400]   fminsearch iterations per start
%   rand_seed         [42]    RNG seed base
%   effort            []      single knob to scale runtime (multiplies starts and iters)
%   plots             [true]  show result plots
%
% Returns a SISO tf with InputDelay used to model the ideal delay on H2.

if nargin < 1 || isempty(hfun), hfun = @triangular; end
if nargin < 2, opts = struct(); end

% Problem sizes (fixed as requested)
na = 6; nb = 5;     % for each branch

% Time grid
T  = getOpt(opts,'T',10);
dt = getOpt(opts,'dt',1e-2);
t = (0:dt:T).'; Nt = numel(t);
h = hfun(t);
w = ones(Nt,1);

% Global search settings
num_starts    = getOpt(opts,'num_starts',12);
local_maxiter = getOpt(opts,'local_maxiter',400);
rand_seed     = getOpt(opts,'rand_seed',42);

effort = getOpt(opts,'effort',[]);
if ~isempty(effort)
  e = max(0.25, effort);
  num_starts    = max(4, round(num_starts * e));
  local_maxiter = max(200, round(local_maxiter * e));
end

show_plots = getOpt(opts,'plots',true);

% Structure (real vs complex)
nc_p = floor(na/2); nr_p = na - 2*nc_p;    % poles per branch
nc_z = floor(nb/2); nr_z = nb - 2*nc_z;    % zeros per branch

rng(rand_seed);

% Initial theta (branch1 + branch2 + delay)
theta0 = [ default_init(T, na, nb, nc_p, nr_p, nc_z, nr_z); ...
           default_init(T, na, nb, nc_p, nr_p, nc_z, nr_z); ...
           log(0.1 + 1e-6) ];   % delay Td ≈ 0.1 s initial

obj_vec = @(th) residual_parallel(th, t, h, w, nc_p, nr_p, nc_z, nr_z);
obj_scl = @(th) sum(abs(obj_vec(th)).^2);

best_f = inf; best_th = theta0;
% Live best plotting setup (every ~30s)
plot_interval = 30; t0 = tic; next_plot = plot_interval; figBest = [];

% Multistart local search
for s = 1:num_starts
  if s==1
    th = theta0;
  else
    th = [ random_init(T, na, nb, nc_p, nr_p, nc_z, nr_z); ...
           random_init(T, na, nb, nc_p, nr_p, nc_z, nr_z); ...
           log(max(1e-6, 0.01 + 0.5*rand)) ];
  end
  f0 = obj_scl(th);
  optsNM = optimset('Display','off','MaxIter',local_maxiter,'MaxFunEvals',5*local_maxiter);
  try
    th_opt = fminsearch(@fwrap_track, th, optsNM);
  catch
    th_opt = th;
  end
  f1 = obj_scl(th_opt);
  if f1 < best_f
    best_f = f1; best_th = th_opt;
  end
end

% Build final systems
[z1,p1,k1,z2,p2,k2,Td] = unpack_par(best_th, nc_p, nr_p, nc_z, nr_z);
sys1 = zpk(z1,p1,k1);
sys2 = zpk(z2,p2,k2); sys2.InputDelay = Td;
sys  = sys1 + sys2;

% Final report and plots
if show_plots
  y1 = impulse_ct(sys1, t);
  y2 = impulse_ct(sys2, t);
  y  = impulse_ct(sys, t);
  figure;
  subplot(2,1,1);
  plot(t,h,'k-','LineWidth',1.4); hold on;
  plot(t,y,'b-','LineWidth',1.1);
  plot(t,y1,'r--');
  plot(t,y - y1,'g--');
  grid on; xlabel('t [s]'); ylabel('h(t)');
  legend('target','sum','branch1','branch2');
  title(sprintf('Parallel fit: RMS %.3e', sqrt(best_f/numel(h))));

  subplot(2,1,2);
  [m,~,wgrid] = bode_mag(sys, 512, T, dt);
  semilogx(wgrid, 20*log10(max(m,1e-12)),'b'); grid on;
  xlabel('\omega [rad/s]'); ylabel('|H(j\omega)| [dB]'); title('Magnitude');
end


  function f = fwrap_track(x)
    x2 = clamp_params_par(x);
    f = obj_scl(x2);
    if f < best_f
      best_f = f; best_th = x2;
    end
    if show_plots
      tnow = toc(t0);
      if tnow >= next_plot
        try
          [z1_,p1_,k1_,z2_,p2_,k2_,Td_] = unpack_par(best_th, nc_p, nr_p, nc_z, nr_z);
          sys1_ = zpk(z1_,p1_,k1_);
          sys2_ = zpk(z2_,p2_,k2_); sys2_.InputDelay = Td_;
          ybest = impulse_ct(sys1_, t) + impulse_ct(sys2_, t);
          if isempty(figBest) || ~isgraphics(figBest)
            figBest = figure('Name','optimize_par: current best','NumberTitle','off');
          else
            set(0,'CurrentFigure',figBest);
          end
          plot(t, h, 'k-', 'LineWidth',1.4); hold on;
          plot(t, ybest, 'b-', 'LineWidth',1.1); hold off; grid on;
          xlabel('t [s]'); ylabel('h(t)');
          title(sprintf('Current best (RMS %.3e)', sqrt(best_f/numel(h))));
          legend('target','best');
          drawnow limitrate nocallbacks;
        catch
        end
        next_plot = tnow + plot_interval;
      end
    end
  end % function fwrap_track

% --------------------------- Helper functions -----------------------------

function e = residual_parallel(th, t, h, w, nc_p, nr_p, nc_z, nr_z)
  try
    [z1,p1,k1,z2,p2,k2,Td] = unpack_par(th, nc_p, nr_p, nc_z, nr_z);
    % Build branches with consistent delay on H2 and evaluate directly
    sys1 = zpk(z1,p1,k1);
    sys2 = zpk(z2,p2,k2); sys2.InputDelay = Td;
    y1 = impulse_ct(sys1, t);
    y2 = impulse_ct(sys2, t);
    y  = y1 + y2;
    e = (y - h) .* w;
    if ~all(isfinite(e)), e = 1e6*ones(size(h)); end
  catch
    e = 1e6*ones(size(h));
  end
end

function ydel = delay_by_shift(y, t, Td)
  % Shift signal by Td >= 0 on grid t, zero-fill before 0
  if Td <= 0
    ydel = y; return;
  end
  ydel = interp1(t, y, t - Td, 'linear', 0);
  ydel = ydel(:);
end

function th = default_init(T, na, nb, nc_p, nr_p, nc_z, nr_z)
  % Log-parameterization: p = -exp(alpha) +/- j exp(beta)
  rates = logspace(log10(2/T), log10(150/T), max(na,nb)).';
  al_p = log(rates(1:max(nc_p,nr_p)+nc_p));
  al_z = log(rates(1:max(nc_z,nr_z)+nc_z));
  th = [];
  for i=1:nc_p, th(end+1) = al_p(i); th(end+1) = log(0.7*exp(al_p(i))); end
  for i=1:nr_p, th(end+1) = al_p(nc_p+i); end
  for i=1:nc_z, th(end+1) = al_z(i); th(end+1) = log(0.7*exp(al_z(i))); end
  for i=1:nr_z, th(end+1) = al_z(nc_z+i); end
  th(end+1) = 1.0; % gain
  th = th(:);
end

function th = random_init(T, na, nb, nc_p, nr_p, nc_z, nr_z)
  th = [];
  sampleA = @() log( exp(log(2/T)) * (exp(log(150/T))-exp(log(2/T))) * rand + exp(log(2/T)) );
  sampleB = @() log( exp(log(0.5/T)) * (exp(log(200/T))-exp(log(0.5/T))) * rand + exp(log(0.5/T)) );
  for i=1:nc_p
    a = sampleA(); b = log(0.5*exp(a) + 0.5*exp(sampleB()));
    th(end+1) = a; th(end+1) = b;
  end
  for i=1:nr_p, th(end+1) = sampleA(); end
  for i=1:nc_z
    a = sampleA(); b = log(0.5*exp(a) + 0.5*exp(sampleB()));
    th(end+1) = a; th(end+1) = b;
  end
  for i=1:nr_z, th(end+1) = sampleA(); end
  th(end+1) = 1.0*(2*rand-1);
  th = th(:);
end

function th2 = clamp_params_par(th)
  th2 = th;
  % keep logs in a sane band
  L = log(1e-2); U = log(1e3);
  th2(1:end-1) = min(U, max(L, th2(1:end-1)));
  % delay at end? here delay is last element of full theta, unknown here; clamp in unpack
end

function [z1,p1,k1,z2,p2,k2,Td] = unpack_par(th, nc_p, nr_p, nc_z, nr_z)
  % Decode two branches + delay (last parameter)
  % Branch layout = [polesC (2/log), polesR (1/log), zerosC (2/log), zerosR (1/log), gain]
  idx = 1;
  [z1,p1,k1,idx] = unpack_one(th, idx, nc_p, nr_p, nc_z, nr_z);
  [z2,p2,k2,idx] = unpack_one(th, idx, nc_p, nr_p, nc_z, nr_z);
  tau_log = th(idx); Td = max(0, exp(tau_log) - 1e-6);
end

function [z,p,k,idx] = unpack_one(th, idx, nc_p, nr_p, nc_z, nr_z)
  p = [];
  for i=1:nc_p
    a = th(idx); b = th(idx+1); idx=idx+2;
    s = -exp(a); w = exp(b);
    p = [p; s+1j*w; s-1j*w]; %#ok<AGROW>
  end
  for i=1:nr_p
    a = th(idx); idx=idx+1; p = [p; -exp(a)]; %#ok<AGROW>
  end
  z = [];
  for i=1:nc_z
    a = th(idx); b = th(idx+1); idx=idx+2;
    s = -exp(a); w = exp(b);
    z = [z; s+1j*w; s-1j*w]; %#ok<AGROW>
  end
  for i=1:nr_z
    a = th(idx); idx=idx+1; z = [z; -exp(a)]; %#ok<AGROW>
  end
  k = th(idx); idx=idx+1;
end

function y = impulse_ct(sys, t)
  % Robust impulse for continuous-time TF
  y = [];
  try
    y = impulse(sys, t);
    if isstruct(y), y = y.y; end
    y = y(:);
    if numel(y)==numel(t), return; end
  catch
  end
  try
    s = step(sys, t);
    if isstruct(s), s = s.y; end
    s = s(:);
    y = gradient(s, t);
  catch
    dt = t(2)-t(1);
    u = zeros(size(t)); u(1) = 1/dt; % area=1
    y = lsim(sys, u, t);
    if isstruct(y), y = y.y; end
    y = y(:);
  end
end

function [mag,phs,w] = bode_mag(sys, N, T, dt)
  if nargin<2, N=512; end
  if nargin<3 || isempty(T), T = 3.5; end
  if nargin<4 || isempty(dt), dt = 1e-3; end
  % Frequency band from time grid (works for systems with delays)
  wmin = max(1e-1, 2*pi/max(T,eps));
  wmax = min(pi/dt*0.8, 1e4);
  w = logspace(log10(wmin), log10(wmax), N);
  H = squeeze(freqresp(sys, w));
  mag = abs(H(:));
  phs = angle(H(:)); %#ok<NASGU>
end

function v = getOpt(s, name, default)
  if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
    v = s.(name);
  else
    v = default;
  end
end

end
