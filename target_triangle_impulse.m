function [n, t, h, dt] = target_triangle_impulse(sample_count, tri_samples)
%TARGET_TRIANGLE_IMPULSE Generate the desired asymmetric triangular impulse.
%   [N, T, H, DT] = TARGET_TRIANGLE_IMPULSE(SAMPLE_COUNT, TRI_SAMPLES) returns
%   the sample indices N, time vector T (in arbitrary units with spacing DT), and
%   the impulse response H that rises linearly to unity over 3/4 of the
%   TRI_SAMPLES window and decays back to zero three times faster. Defaults are
%   SAMPLE_COUNT = 1000 and TRI_SAMPLES = 300.
%
%   The impulse maximum is normalized to 1 and occupies the first TRI_SAMPLES
%   out of SAMPLE_COUNT total samples.

if nargin < 1 || isempty(sample_count)
    sample_count = 1000;
end
if nargin < 2 || isempty(tri_samples)
    tri_samples = 300;
end

validateattributes(sample_count, {'double'}, {'scalar','integer','>=',tri_samples, '>=',1}, mfilename, 'sample_count', 1);
validateattributes(tri_samples, {'double'}, {'scalar','integer','>=',2}, mfilename, 'tri_samples', 2);

rise_len = round(0.75 * tri_samples);
fall_len = tri_samples - rise_len;

n = (0:sample_count-1).';
dt = 1;                                % Unit spacing between samples
t = n * dt;

h = zeros(sample_count, 1);
if rise_len > 1
    rise_mask = n < rise_len;
    h(rise_mask) = n(rise_mask) / (rise_len - 1);
end
if fall_len > 1
    fall_mask = n >= rise_len & n < rise_len + fall_len;
    h(fall_mask) = 1 - (n(fall_mask) - rise_len) / (fall_len - 1);
elseif fall_len == 1
    h(rise_len) = 1;
end

end
