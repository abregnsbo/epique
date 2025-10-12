function G = smooth_triang_resp(Tfall, order, beta, lpOrder, lpType, varargin)
%SMOOTH_TRIANG_RESP Triangular impulse smoothed by an analog low-pass
%   G = smooth_triang_resp(Tfall, order, beta, lpOrder, lpType) returns a
%   transfer function whose impulse equals the triangular response convolved
%   with an Nth-order low-pass. Delays inside the triangle use Pade(order).
%
%   Smoothing low-pass options (analog prototypes):
%     - lpType: 'bessel' (default), 'butter', 'cheby1', 'cheby2', 'ellip'
%     - beta  : cutoff rate (rad/s). Larger = lighter smoothing. Default 10.
%     - lpOrder: integer >= 1 (default 1)
%     - Additional name-value for ripple where applicable:
%         'Rp' (dB) for cheby1/ellip (default 1), 'Rs' (dB) for cheby2/ellip (default 40)
%
%   Inputs:
%     - Tfall : nonnegative scalar fall time (seconds)
%     - order : integer Pade order (1..20)
%     - beta  : positive scalar low-pass rate (default 10)
%
%   Notes:
%     - Larger beta -> lighter smoothing (closer to triangle).
%     - Smaller beta -> heavier smoothing (more rounded corners).
%
%   Examples:
%     impulse(smooth_triang_resp(0.5, 12, 10), 3)
%     impulse(smooth_triang_resp(0.1,  6,  8, 2), 3.5)
%     pzmap(smooth_triang_resp(0.5, 12, 10))
%
%   See also: tf, pade, minreal, triang_resp

    if nargin < 1
        error('pade:NotEnoughInputs', 'Provide Tfall (seconds).');
    end
    if ~isscalar(Tfall) || ~isreal(Tfall) || isnan(Tfall) || Tfall < 0
        error('pade:InvalidDelay', 'Tfall must be a real, nonnegative scalar.');
    end
    if nargin < 2 || isempty(order)
        order = 2;
    end
    if ~isscalar(order) || order < 1 || order > 20
        error('pade:InvalidOrder', 'ORDER must be in the range 1-20.');
    end
    if nargin < 3 || isempty(beta)
        beta = 10; % rad/s, light smoothing by default
    end
    if ~isscalar(beta) || ~isreal(beta) || isnan(beta) || beta <= 0
        error('pade:InvalidBeta', 'beta must be a real, positive scalar.');
    end
    if nargin < 4 || isempty(lpOrder)
        lpOrder = 1;
    end
    if ~isscalar(lpOrder) || lpOrder < 1 || lpOrder ~= floor(lpOrder)
        error('pade:InvalidLPOrder', 'lpOrder must be a positive integer.');
    end
    if nargin < 5 || isempty(lpType)
        lpType = 'bessel';
    end
    % Octave-compatible: accept only char arrays for lpType
    if ~ischar(lpType)
        error('pade:InvalidLPType', 'lpType must be a char array: bessel/butter/cheby1/cheby2/ellip.');
    end
    lpType = lower(lpType);

    % Optional ripple parameters
    p = inputParser;
    addParameter(p, 'Rp', 1, @(x) isscalar(x) && isreal(x) && x > 0);
    addParameter(p, 'Rs', 40, @(x) isscalar(x) && isreal(x) && x > 0);
    parse(p, varargin{:});
    Rp = p.Results.Rp; % dB (passband ripple)
    Rs = p.Results.Rs; % dB (stopband attenuation)

    s = tf('s');
    Gtri = triang_resp(Tfall, order);

    % Build analog low-pass prototype at cutoff beta (rad/s)
    switch lpType
        case "bessel"
            % besself returns analog lowpass with cutoff beta
            [num, den] = besself(lpOrder, beta);
            Hlp = tf(num, den);
        case "butter"
            [z, p, k] = butter(lpOrder, beta, 's');
            Hlp = zpk(z, p, k);
        case "cheby1"
            [z, p, k] = cheby1(lpOrder, Rp, beta, 's');
            Hlp = zpk(z, p, k);
        case "cheby2"
            [z, p, k] = cheby2(lpOrder, Rs, beta, 's');
            Hlp = zpk(z, p, k);
        case "ellip"
            [z, p, k] = ellip(lpOrder, Rp, Rs, beta, 's');
            Hlp = zpk(z, p, k);
        otherwise
            error('pade:InvalidLPType', 'Unknown lpType: %s', lpType);
    end

    % Normalize to unity DC gain to preserve overall area
    g0 = dcgain(Hlp);
    if isfinite(g0) && g0 ~= 0
        Hlp = Hlp / g0;
    end

    G = minreal(Gtri * Hlp);
end
