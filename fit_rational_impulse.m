function [num, den, Hs] = fit_rational_impulse(h, dt, denOrder, varargin)
%FIT_RATIONAL_IMPULSE Fit a continuous-time rational transfer to a sampled impulse.
%   [NUM, DEN, HS] = FIT_RATIONAL_IMPULSE(H, DT, DENORDER) approximates the
%   impulse response sequence H (sampled every DT time units) with a rational
%   transfer function whose denominator degree is DENORDER. The numerator degree
%   defaults to DENORDER-1 but can be overridden. The fit is performed in the
%   frequency domain via complex least squares with column scaling, optional
%   Tikhonov regularization, and SVD-based pseudo-inversion for robustness.
%
%   Optional name-value pairs:
%       ''NumeratorOrder''   - desired numerator order (>=0, default denOrder-1)
%       ''FrequencySamples'' - number of frequency points used (default 20*(den+num+1))
%       ''FreqRange''        - [wmin wmax] rad/s range for fitting (default
%                              [2*pi/(dt*(10*length(h))), min(2*pi/dt, pi/dt)])
%       ''MatchTimeDomain''  - logical flag to rescale numerator for best
%                              impulse-domain least squares match (default true)
%       ''Regularization''   - Tikhonov regularization weight (default 1e-1)
%
%   Returns numerator and denominator coefficient vectors (descending powers of s)
%   plus a tf object HS built from those coefficients.

if nargin < 3
    error('fit_rational_impulse:NotEnoughInputs', ...
        'Provide impulse samples h, sample spacing dt, and denominator order.');
end

validateattributes(h, {'double'}, {'nonempty'}, mfilename, 'h', 1);
validateattributes(dt, {'double'}, {'scalar','real','positive'}, mfilename, 'dt', 2);
validateattributes(denOrder, {'double'}, {'scalar','integer','positive'}, mfilename, 'denOrder', 3);

p = inputParser;
defaultNumOrder = max(denOrder - 1, 0);
addParameter(p, 'NumeratorOrder', defaultNumOrder, ...
    @(x) validateattributes(x, {'double'}, {'scalar','integer','>=',0}));
addParameter(p, 'FrequencySamples', [], ...
    @(x) validateattributes(x, {'double'}, {'scalar','integer','>=',denOrder+1}));
addParameter(p, 'FreqRange', [], ...
    @(x) validateattributes(x, {'double'}, {'vector','numel',2,'positive'}));
addParameter(p, 'MatchTimeDomain', true, ...
    @(x) validateattributes(x, {'logical','numeric'}, {'scalar'}));
addParameter(p, 'Regularization', 1e-1, ...
    @(x) validateattributes(x, {'double'}, {'scalar','real','>=',0}));
parse(p, varargin{:});
numOrder = p.Results.NumeratorOrder;
freqSamples = p.Results.FrequencySamples;
freqRange = p.Results.FreqRange;
matchTD = logical(p.Results.MatchTimeDomain);
lambda = p.Results.Regularization;

h = h(:);                               % Column vector
sampleCount = numel(h);
if sampleCount <= denOrder + numOrder
    error('fit_rational_impulse:InsufficientSamples', ...
        'Need more samples than total model order (got %d, need > %d).', ...
        sampleCount, denOrder + numOrder);
end

t = (0:sampleCount-1).' * dt;           % Sample times

% Frequency grid selection
if isempty(freqSamples)
    freqSamples = max(200, 20 * (denOrder + numOrder + 1));
end
if isempty(freqRange)
    wmin = 2 * pi / (dt * max(10 * sampleCount, 1));
    wmax = min(2 * pi / dt, pi / dt);    % Limit emphasis on very high w
else
    wmin = freqRange(1);
    wmax = freqRange(2);
end
wmin = max(wmin, 1e-6);                 % Avoid zero for stability
wmax = max(wmax, wmin * 10);
w = logspace(log10(wmin), log10(wmax), freqSamples).';
s = 1j * w;

% Desired frequency response via Laplace transform approximation
E = exp(-s * t.');                       % freqSamples x sampleCount
Hjw = dt * (E * h);

n = denOrder;
m = numOrder;

% Build complex linear system A*x = b
col_a = zeros(freqSamples, n);
for k = 1:n
    col_a(:, k) = Hjw .* (s .^ (n - k));
end
col_b = zeros(freqSamples, m + 1);
for k = 0:m
    col_b(:, k + 1) = -(s .^ (m - k));
end
A = [col_a, col_b];
b = (s .^ n) .* Hjw;

% Stack real/imag components
A_ls = [real(A); imag(A)];
b_ls = [real(b); imag(b)];

% Column scaling for conditioning
colNorm = sqrt(sum(A_ls.^2, 1));
colNorm(colNorm == 0) = 1;
A_scaled = A_ls ./ colNorm;

% Optional Tikhonov regularization
if lambda > 0
    regMatrix = sqrt(lambda) * eye(size(A_scaled, 2));
    A_aug = [A_scaled; regMatrix];
    b_aug = [b_ls; zeros(size(regMatrix, 1), 1)];
else
    A_aug = A_scaled;
    b_aug = b_ls;
end

% Solve via SVD-based pseudo inverse for robustness
[U, S, V] = svd(A_aug, 'econ');
sigma = diag(S);
tol = max(size(A_aug)) * eps(max(sigma));
sigma_inv = zeros(size(sigma));
valid = sigma > tol;
sigma_inv(valid) = 1 ./ sigma(valid);
x_scaled = V * (sigma_inv .* (U' * b_aug));

x = (x_scaled.' ./ colNorm).';

a = x(1:n).';
bc = x(n+1:end).';

den = [1, a];
num = bc;

% Enforce real coefficients for real impulse response
if isreal(h)
    den = real(den);
    num = real(num);
end

Hs = tf(num, den);

% Optional least-squares amplitude correction in time domain
if matchTD
    h_est = impulse(Hs, t);
    den_norm = h_est' * h_est;
    if den_norm > eps * sampleCount
        gain = (h' * h_est) / den_norm;
        num = gain * num;
        Hs = tf(num, den);
    end
end
end
