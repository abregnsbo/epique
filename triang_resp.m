function G = triang_resp(Tfall, order)
%TRIANG_RESP Return a tf object (Laplacian) that represents a triangular
%   impulse response which rises from 0 to 1 in 1 second, and then falls down 
%   zero in Tfall time. The exact Laplacian contains exponentials which are 
%   approxiamted using Pade functions or order 'order'.
%
%   Examples:
%      impulse(triang_resp(0.5,12),2.5)
%      pzmap(triang_resp(0.5,12))
%
%   See also: tf, pade

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

    s = tf('s')
    t1 = 1
    t2 = t1 + Tfall
    G = ((t2-t1) + t1*pade(t2,order) - t2*pade(t1,order)) / (s^2*t1*(t2-t1))
end
