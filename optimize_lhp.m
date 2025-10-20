function sys = optimize_lhp(hfun, opts)
% Optimize continuous-time poles and zeros directly in LHP to match a target
% impulse response. Uses a slow, robust simulated annealing + multistart
% scheme with local Nelder-Mead refinements. Does not rely on other files.
%
% Usage:
%   sys = optimize_lhp(@target_fun)
%   sys = optimize_lhp([])           % defaults to triangular target
%
% Both poles and zeros are constrained to the left-half plane (minimum
% phase). The fit is time-domain least squares on [0, T].

% ---------------------- User-configurable parameters ----------------------
if nargin < 1 || isempty(hfun), hfun = @triangular; end
if nargin < 2, opts = struct(); end

% Keep order 8 by default (user preference)
na = getOpt(opts,'na',8);           % denominator order (pole count)
nb = getOpt(opts,'nb',na-1);        % numerator order (< na)

% Time grid options
T  = getOpt(opts,'T',10);          % seconds, horizon capturing the impulse tail
dt = getOpt(opts,'dt',1e-2);        % time step for the objective

% Global search options (longer, slower, more starts)
maxSAIters = getOpt(opts,'maxSAIters',50000);      % SA iterations per start
num_starts = getOpt(opts,'num_starts',100);        % multistart annealing runs
local_refine_every = getOpt(opts,'local_refine_every',600); % steps between local refinements
local_maxiter = getOpt(opts,'local_maxiter',800);  % fminsearch iterations per refinement
anneal_alpha = getOpt(opts,'anneal_alpha',0.998);  % temperature decay per iteration
step_scale = getOpt(opts,'step_scale',0.35);       % base proposal step (log-params)
step_decay = getOpt(opts,'step_decay',0.999);      % step-size decay per iter
rand_seed = getOpt(opts,'rand_seed',42);           % base seed for reproducibility

% Time budget: run until this wall-clock seconds is reached (2.5h default)
time_limit_sec = getOpt(opts,'time_limit_sec',9000);

% Adaptive step-size to target acceptance ratio
adapt_steps     = getOpt(opts,'adapt_steps',true);
target_accept   = getOpt(opts,'target_accept',0.30);
adapt_gain      = getOpt(opts,'adapt_gain',0.15);  % multiplicative adjustment strength

% Reheating when stagnant
reheat_enable   = getOpt(opts,'reheat_enable',true);
reheat_interval = getOpt(opts,'reheat_interval',300); % seconds without improvement
reheat_factor   = getOpt(opts,'reheat_factor',0.3);   % Temp = max(Temp, reheat_factor*T0)

% Time-domain sample weights: keep uniform by default (no early emphasis)
w_time = getOpt(opts,'w_time',[]);

% Live plotting intervals (seconds)
stats_interval   = getOpt(opts,'plot_stats_interval',5);   % every few seconds
impulse_interval = getOpt(opts,'plot_impulse_interval',60);% every minute

% ---------------------- Target impulse ------------------------------------
t = (0:dt:T).';
if nargin < 1 || isempty(hfun)
  hfun = @triangular; % default target: see triangular.m in this repo
end
h = hfun(t);
if isempty(w_time), w_time = ones(size(t)); end
w_time = w_time(:);

% ---------------------- Structure (real vs complex) -----------------------
% Choose number of complex conjugate pairs for poles and zeros
nc_p = floor(na/2); nr_p = na - 2*nc_p;
nc_z = floor(nb/2); nr_z = nb - 2*nc_z;

% ---------------------- Initialization ------------------------------------
rng(rand_seed);
theta0 = default_init(T, na, nb, nc_p, nr_p, nc_z, nr_z);

% Objective handles
obj_vec = @(th) time_residual(th, t, h, w_time, nc_p, nr_p, nc_z, nr_z);
obj_scl = @(th) sum(abs(obj_vec(th)).^2);

% ---------------------- Global annealing + multistart ---------------------
best_th = [];
best_f  = inf;
% Live plotting/state
t0 = tic; next_stats = stats_interval; next_imp = impulse_interval; next_save = impulse_interval;
time_hist = []; best_hist = []; temp_hist = []; acc_hist = [];
acc_accept = 0; acc_total = 0; do_plot = true;
figStats = []; figImp = [];
last_improve_t = 0;  % time since t0 of last best improvement
start_idx = 0;
while true
  if toc(t0) >= time_limit_sec, break; end
  start_idx = start_idx + 1;
  if start_idx > num_starts, start_idx = 1; end
  rng(rand_seed + start_idx); % distinct seed per start
  if start_idx==1
    th = theta0;
  else
    th = random_init(T, na, nb, nc_p, nr_p, nc_z, nr_z);
  end
  f = obj_scl(th);
  T0 = max(1e-6, 0.1*f);   % initial temperature
  Temp = T0;
  step = proposal_scale(th, step_scale);
  for k=1:maxSAIters
    if toc(t0) >= time_limit_sec, break; end
    th_prop = propose(th, step);
    f_prop = obj_scl(th_prop);
    acc_total = acc_total + 1;
    if (f_prop < f) || (rand < exp(-(f_prop - f)/max(Temp,1e-12)))
      th = th_prop; f = f_prop;
      acc_accept = acc_accept + 1;
    end
    % Track best
    if f < best_f
      best_f = f; best_th = th;
      last_improve_t = toc(t0);
    end
    % Local refinement periodically
    if mod(k, local_refine_every)==0
      th = local_refine(th, obj_scl, local_maxiter);
      f = obj_scl(th);
      if f < best_f
        best_f = f; best_th = th;
        last_improve_t = toc(t0);
      end
    end
    % Cool down and slightly shrink step size
    Temp = Temp * anneal_alpha;
    step = step_decay * step;

    % Reheat on stagnation
    if reheat_enable && (toc(t0) - last_improve_t > reheat_interval)
      Temp = max(Temp, reheat_factor*T0);
      step = step * (1 + adapt_gain);
      last_improve_t = toc(t0); % avoid repeated reheat immediately
    end

    % Periodic plotting
    tnow = toc(t0);
    if do_plot && tnow >= next_stats
      % Record stats snapshot
      time_hist(end+1,1) = tnow; %#ok<AGROW>
      best_hist(end+1,1) = best_f; %#ok<AGROW>
      temp_hist(end+1,1) = Temp; %#ok<AGROW>
      if acc_total>0
        acc_hist(end+1,1) = acc_accept/acc_total; %#ok<AGROW>
      else
        acc_hist(end+1,1) = NaN; %#ok<AGROW>
      end
      % Reset acceptance counters for next window
      acc_accept = 0; acc_total = 0;
      % Adaptive step-size to steer acceptance toward target
      if adapt_steps && ~isnan(acc_hist(end))
        ratio = acc_hist(end);
        adj = exp(adapt_gain * (ratio - target_accept));
        step = step * adj;
      end
      % Update stats figure
      try
        if isempty(figStats) || ~isgraphics(figStats)
          figStats = figure('Name','optimize_lhp: stats','NumberTitle','off');
        else
          figure(figStats);
        end
        tiledlayout(2,1);
        nexttile(1);
        semilogy(time_hist, best_hist, 'b-'); grid on; xlabel('time [s]'); ylabel('best f');
        title('Best objective over time');
        nexttile(2);
        plot(time_hist, acc_hist, 'm-'); hold on; plot(time_hist, temp_hist, 'r-'); hold off; grid on;
        xlabel('time [s]'); ylabel('acc. ratio / Temp'); legend('accept','Temp');
        drawnow limitrate nocallbacks;
      catch
        do_plot = false; % disable plotting if headless
      end
      next_stats = tnow + max(1, stats_interval);
    end

    % Independent autosave every minute
    if tnow >= next_save
      try
        [zBest,pBest,kBest] = unpack_params(best_th, nc_p, nr_p, nc_z, nr_z);
        sysBest = zpk(zBest,pBest,kBest);
        save('optimize_lhp.mat','sysBest','time_hist','best_hist','temp_hist','acc_hist','t','h');
      catch
      end
      next_save = tnow + max(10, impulse_interval);
    end

    if do_plot && tnow >= next_imp
      try
        [zBest,pBest,kBest] = unpack_params(best_th, nc_p, nr_p, nc_z, nr_z);
        sysBest = zpk(zBest,pBest,kBest);
        yBest = impulse_ct(sysBest, t);
        if isempty(figImp) || ~isgraphics(figImp)
          figImp = figure('Name','optimize_lhp: best impulse','NumberTitle','off');
        else
          figure(figImp);
        end
        plot(t, h, 'k-', 'LineWidth',1.5); hold on;
        plot(t, yBest, 'b-', 'LineWidth',1.2); hold off; grid on;
        xlabel('t [s]'); ylabel('h(t)');
        title(sprintf('Current best impulse (RMS ~ %.3e)', sqrt(best_f/numel(h)) ));
        legend('target','best');
        drawnow limitrate nocallbacks;
      catch
        do_plot = false;
      end
      next_imp = tnow + max(10, impulse_interval);
    end
  end
end

% Final local polish on best
best_th = local_refine(best_th, obj_scl, 2*local_maxiter);

% Build system and report
[z,p,k] = unpack_params(best_th, nc_p, nr_p, nc_z, nr_z);
sys = zpk(z,p,k);
y  = impulse_ct(sys, t);
rms = norm((y - h).*w_time) / sqrt(numel(h));
fprintf('Final RMS (time-domain): %.4e\n', rms);

% ------------------------------ Plots -------------------------------------
figure;
subplot(2,1,1);
plot(t,h,'k-','LineWidth',1.5); hold on;
plot(t,y,'b-','LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('h(t)');
legend('target','fit'); title(sprintf('LHP impulse fit (na=%d, nb=%d), RMS=%.2e',na,nb,rms));

subplot(2,1,2);
[m,~,w] = bode_mag(sys, 512);
semilogx(w, 20*log10(max(m,1e-12)),'b'); grid on;
xlabel('\omega [rad/s]'); ylabel('|H(j\omega)| [dB]'); title('Magnitude response');

end % function optimize_lhp

% --------------------------- Helper functions -----------------------------

function th = default_init(T, na, nb, nc_p, nr_p, nc_z, nr_z)
  % Log-parameterization: real parts via -exp(alpha), imaginary via exp(beta)
  rates = logspace(log10(2/T), log10(150/T), max(na,nb)).';
  al_p = log(rates(1:max(nc_p,nr_p)+nc_p));
  al_z = log(rates(1:max(nc_z,nr_z)+nc_z));
  % Poles: nc pairs then nr real
  th = [];
  for i=1:nc_p
    th(end+1) = al_p(i);          % alpha (real part magnitude)
    th(end+1) = log(0.7*exp(al_p(i))); % beta (imag part magnitude)
  end
  for i=1:nr_p
    th(end+1) = al_p(nc_p+i);     % real poles
  end
  % Zeros: nc pairs then nr real
  for i=1:nc_z
    th(end+1) = al_z(i);
    th(end+1) = log(0.7*exp(al_z(i)));
  end
  for i=1:nr_z
    th(end+1) = al_z(nc_z+i);
  end
  % Gain (free real scalar)
  th(end+1) = 1.0;  % k
  th = th(:);
end

function v = getOpt(s, name, default)
  if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
    v = s.(name);
  else
    v = default;
  end
end

function th = random_init(T, na, nb, nc_p, nr_p, nc_z, nr_z)
  % Random but reasonable initial logs of rates
  th = [];
  % helper to sample logs within a band tied to T
  sampleA = @() log( exp(log(2/T)) * (exp(log(150/T))-exp(log(2/T))) * rand + exp(log(2/T)) );
  sampleB = @() log( exp(log(0.5/T)) * (exp(log(200/T))-exp(log(0.5/T))) * rand + exp(log(0.5/T)) );
  for i=1:nc_p
    th(end+1) = sampleA();
    th(end+1) = log(0.5*exp(th(end)) + 0.5*exp(sampleB()));
  end
  for i=1:nr_p, th(end+1) = sampleA(); end
  for i=1:nc_z
    th(end+1) = sampleA();
    th(end+1) = log(0.5*exp(th(end)) + 0.5*exp(sampleB()));
  end
  for i=1:nr_z, th(end+1) = sampleA(); end
  th(end+1) = 1.0*(2*rand-1); % gain
  th = th(:);
end

function step = proposal_scale(th, base)
  % Per-parameter proposal step sizes
  step = base * ones(size(th));
  step(end) = max(0.1, 0.1*abs(th(end))); % gain step
end

function th2 = propose(th, step)
  % Random Gaussian perturbation with clamping of logs to reasonable range
  th2 = th + step .* randn(size(th));
  th2 = clamp_params(th2);
end

function th = clamp_params(th)
  % Keep logs within a numerically sane band
  % alpha,beta in [log(1e-2), log(1e3)] scaled by 1/T implicitly via init
  L = log(1e-2); U = log(1e3);
  th(1:end-1) = min(U, max(L, th(1:end-1)));
  % gain unconstrained
end

function th_opt = local_refine(th0, obj_scl, maxIter)
  opts = optimset('Display','off','MaxIter',maxIter,'MaxFunEvals',5*maxIter);
  fwrap = @(x) obj_scl(clamp_params(x));
  try
    th_opt = fminsearch(fwrap, th0, opts);
  catch
    th_opt = th0;
  end
  th_opt = clamp_params(th_opt);
end

function [z,p,k] = unpack_params(th, nc_p, nr_p, nc_z, nr_z)
  % Build LHP poles/zeros from log-parameters
  idx = 1; p = [];
  for i=1:nc_p
    alpha = th(idx);  beta = th(idx+1); idx=idx+2;
    sigma = -exp(alpha); omega = exp(beta);
    p = [p; sigma+1j*omega; sigma-1j*omega]; %#ok<AGROW>
  end
  for i=1:nr_p
    alpha = th(idx); idx=idx+1; p = [p; -exp(alpha)]; %#ok<AGROW>
  end
  z = [];
  for i=1:nc_z
    alpha = th(idx);  beta = th(idx+1); idx=idx+2;
    sigma = -exp(alpha); omega = exp(beta);
    z = [z; sigma+1j*omega; sigma-1j*omega]; %#ok<AGROW>
  end
  for i=1:nr_z
    alpha = th(idx); idx=idx+1; z = [z; -exp(alpha)]; %#ok<AGROW>
  end
  k = th(idx);
end

function e = time_residual(th, t, h, w_time, nc_p, nr_p, nc_z, nr_z)
  [z,p,k] = unpack_params(th, nc_p, nr_p, nc_z, nr_z);
  sys = zpk(z,p,k);
  y = impulse_ct(sys, t);
  e = (y - h) .* w_time;
end

function y = impulse_ct(sys, t)
  % Robust impulse for continuous-time TF. Prefer impulse(); fallback to d/dt step().
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
    u = zeros(size(t)); u(1) = 1/dt;
    y = lsim(sys, u, t);
    if isstruct(y), y = y.y; end
    y = y(:);
  end
end

function [mag,phs,w] = bode_mag(sys, N)
  if nargin<2, N=512; end
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
  num = polyval(b, s);
  den = polyval(a, s);
  H = num ./ den;
  mag = abs(H(:));
  phs = angle(H(:)); %#ok<NASGU>
end

