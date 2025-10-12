function G = poly_resp(Tfall, order)
%POLY_RESP Return a tf object (Laplacian) for a quintic (5th-degree) impulse
%   The impulse response rises from 0 to 1 on [0,1] with a 5th-degree
%   polynomial (smooth S-curve), then falls from 1 to 0 on [1, 1+Tfall]
%   with a time-scaled 5th-degree polynomial. Delay exponentials are
%   approximated by Pade of order 'order'.
%
%   The polynomial used is the standard smoothstep quintic:
%       p(x) = 10 x^3 - 15 x^4 + 6 x^5,  x in [0,1]
%   Rise: h1(t) = p(t)
%   Fall: h2(t) = 1 - p((t-1)/Tfall),  t in [1, 1+Tfall]
%
%   Examples:
%      impulse(poly_resp(0.5,12), 2.5)
%      pzmap(poly_resp(0.5,12))
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
    E1 = pade(t1, order);       % ~ e^{-s * 1}

    % Helper: J_n(a) = int_0^a t^n e^{-s t} dt in rational form using Pade for e^{-s a}
    function J = Jn(n, a)
        if a == 0
            J = 0 * s; % zero tf with consistent type
            return;
        end
        if a == 1
            Ea = E1;
        else
            Ea = pade(a, order);
        end
        fac = factorial(n);
        polySum = 0;
        for k = 0:n
            polySum = polySum + (s*a)^k / factorial(k);
        end
        J = fac / s^(n+1) * (1 - Ea * polySum);
    end

    % Rising segment on [0,1]: h1(t) = 10 t^3 - 15 t^4 + 6 t^5
    I1 = 10*Jn(3, 1) - 15*Jn(4, 1) + 6*Jn(5, 1);

    if Tfall == 0
        Graw = I1;
    else
        Delta = t2 - t1;
        % Falling segment on [1,1+Delta]: h2(t) = 1 - 10 (tau/Delta)^3 + 15 (tau/Delta)^4 - 6 (tau/Delta)^5
        % Integral = e^{-s*1} * [ J0(Delta) - 10/Delta^3 J3(Delta) + 15/Delta^4 J4(Delta) - 6/Delta^5 J5(Delta) ]
        K = Jn(0, Delta) - (10/Delta^3)*Jn(3, Delta) + (15/Delta^4)*Jn(4, Delta) - (6/Delta^5)*Jn(5, Delta);
        I2 = E1 * K;
        Graw = I1 + I2;
    end

    G = minreal(Graw);
end

