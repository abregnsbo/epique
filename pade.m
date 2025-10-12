function G = pade(T, order)
%PADE Return [n/n] Pade approximation of a delay as a tf object.
%   G = PADE(T) returns the 2nd-order Pade approximation of the pure delay
%   exp(-s*T) as a continuous-time transfer function model (tf object).
%
%   G = PADE(T, ORDER) returns the [ORDER/ORDER] Pade approximation, where
%   ORDER is an integer 1..20 (default 2). For ORDER=2 or 3 this matches the
%   familiar closed forms.
%
%   Inputs
%     T      - Nonnegative delay time (scalar, in seconds or time units)
%     ORDER  - Integer order in [1, 20] (default 2)
%
%   Output
%     G      - Transfer function tf that approximates exp(-s*T)
%
%   Notes
%   - Requires Control System Toolbox (for tf objects).
%   - Coefficients are normalized so the denominator constant term is 1.
%   - This function name may shadow MATLAB's built-in `pade`. Place this
%     file earlier on your MATLAB path to use it instead.
%
%   Examples
%     % 2nd-order approximation of a 0.5 s delay
%     G2 = pade(0.5);
%
%     % 3rd-, 4th-, and 5th-order approximations
%     G3 = pade(0.5, 3);
%     G4 = pade(0.5, 4);
%     G5 = pade(0.5, 5);
%
%   See also: tf, impulse, bode, pade (MATLAB built-in)

    if nargin < 1
        error('pade:NotEnoughInputs', 'Provide delay T (seconds).');
    end
    if ~isscalar(T) || ~isreal(T) || isnan(T) || T < 0
        error('pade:InvalidDelay', 'T must be a real, nonnegative scalar.');
    end
    if nargin < 2 || isempty(order)
        order = 2;
    end
    if ~isscalar(order) || order ~= fix(order) || order < 1 || order > 20
        error('pade:InvalidOrder', 'ORDER must be an integer in [1, 20].');
    end

    % Shortcut: zero delay is exactly 1
    if T == 0
        G = tf(1, 1);
        return;
    end

    % General [n/n] Pade coefficients for exp(-sT)
    % Build normalized coefficients a_k such that
    %   Q_n(x) = sum_{k=0}^n a_k x^k,  P_n(x) = sum_{k=0}^n (-1)^k a_k x^k,
    % with a_0 = 1 and x = s*T. Recurrence avoids factorial overflow:
    %   a_k = a_{k-1} * (n - k + 1) / ((2n - k + 1) * k)
    n = order;
    a = zeros(1, n + 1);
    a(1) = 1; % a_0
    for k = 1:n
        a(k+1) = a(k) * (n - k + 1) / ((2*n - k + 1) * k);
    end

    % Apply T^k scaling for x = s*T
    Tk = T .^ (0:n);
    den_asc = a .* Tk;                 % positive signs
    num_asc = ((-1).^(0:n)) .* den_asc; % alternating signs

    % Convert to descending powers of s for tf()
    den = fliplr(den_asc);
    num = fliplr(num_asc);

    G = tf(num, den);
end
