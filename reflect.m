function sys_out = reflect(sys_in)
% reflect  Reflect RHP zeros of a continuous-time SISO tf to the LHP.
%   sys_out = reflect(sys_in) takes a continuous-time SISO transfer
%   function model `sys_in`, reflects all zeros with positive real part
%   across the imaginary axis, and returns a tf model `sys_out` with no
%   RHP zeros. The overall gain is preserved (no gain adjustment).
%
%   Notes
%   - Only continuous-time SISO models are supported.
%   - Zeros on the imaginary axis (within tolerance) are left unchanged.
%   - The function does not modify poles or gain.

  % Basic checks
  if ~isa(sys_in, 'tf') && ~isa(sys_in, 'zpk')
    error('reflect:argtype', 'Input must be a tf or zpk model.');
  end
  if ~isequal(size(sys_in), [1 1])
    error('reflect:siso', 'Only SISO systems are supported.');
  end
  try
    Ts = get(sys_in, 'Ts');
  catch
    Ts = 0;
  end
  if ~isempty(Ts) && Ts ~= 0
    error('reflect:discrete', 'Only continuous-time systems are supported.');
  end

  % Work in ZPK form
  [z,p,k] = zpkdata(sys_in, 'v');
  tol = 1e-12;

  % Reflect RHP zeros across the imaginary axis
  zr = z;
  idx = real(zr) > tol;
  zr(idx) = -conj(zr(idx));

  % Build output (as tf, preserving gain)
  sys_out = tf(zpk(zr, p, k));
end

