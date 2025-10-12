function G = cos_resp(Tfall, order)
%COS_RESP Return a tf object (Laplacian) for a raised-cosine impulse
%   Response rises from 0 to 1 over 1 second via a raised cosine, then
%   falls from 1 to 0 over Tfall seconds via a raised cosine. The exact
%   Laplace expression contains exponentials, approximated with Pade of
%   order 'order'.
%
%   Examples:
%      impulse(cos_resp(0.5,12), 2.5)
%      pzmap(cos_resp(0.5,12))
%
%   See also: tf, pade, minreal

    if nargin < 1
        error('pade:NotEnoughInputs', 'Provide delay T (seconds).');
    end
    if ~isscalar(Tfall) || ~isreal(Tfall) || isnan(Tfall) || Tfall < 0
        error('pade:InvalidDelay', 'Tfall must be a real, nonnegative scalar.');
    end
    if nargin < 2 || isempty(order)
        order = 2;
    end
    if ~isscalar(order) || order < 1 || order > 20
        error('pade:InvalidOrder', 'ORDER must be in the range 2-20.');
    end

    s = tf('s');
    t1 = 1;
    t2 = t1 + Tfall;

    % Pade approximations of delays
    E1 = pade(t1, order);       % ~ e^{-s t1}

    if Tfall == 0
        % Only the rising raised-cosine segment exists
        w1 = pi / t1;
        I1 = 0.5 * (1 - E1) / s - 0.5 * s * (1 + E1) / (s^2 + w1^2);
        Graw = I1;
    else
        Delta = t2 - t1;
        w1 = pi / t1;
        w2 = pi / Delta;
        EDelta = pade(Delta, order);  % ~ e^{-s * Delta}

        % Integral over [0, t1] of 0.5 - 0.5 cos(pi t / t1)
        I1 = 0.5 * (1 - E1) / s - 0.5 * s * (1 + E1) / (s^2 + w1^2);

        % Integral over [t1, t2] of 0.5 + 0.5 cos(pi (t - t1) / (t2 - t1))
        inner = 0.5 * (1 - EDelta) / s + 0.5 * s * (1 + EDelta) / (s^2 + w2^2);
        I2 = E1 * inner;

        Graw = I1 + I2;
    end

    G = minreal(Graw);
end
