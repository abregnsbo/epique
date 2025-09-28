%% Fit and visualize rational approximation of triangular impulse response
% Obtain the target impulse from helper function
[n, t, h_tri, dt] = target_triangle_impulse(1000, 300);

% Rational transfer function fitting parameters
numOrder = 5;                        % Numerator order (default den-1)
denOrder = 6;                        % Denominator order
[num, den, Hs] = fit_rational_impulse(h_tri, dt, denOrder, ...
    'NumeratorOrder', numOrder, 'Regularization', 1e-1);

% Continuous-time impulse response of rational approximation on same grid
h_rat = impulse(Hs, t);

% Plot comparison (sample index axis)
figure;
plot(n, h_tri, 'LineWidth', 1.8, 'DisplayName', 'Target triangular impulse');
hold on;
plot(n, h_rat, '--', 'LineWidth', 1.8, 'DisplayName', 'Rational H(s) impulse');
hold off;
xlabel('Sample index');
ylabel('Amplitude');
title('Triangular impulse vs. fitted 6th-order rational approximation');
legend('Location', 'best');
grid on;

% Display transfer function in the Command Window
fprintf('Rational approximation H(s):\n');
display(Hs);

% Frequency response (Bode) in arbitrary frequency units
f_Hz = logspace(-3, 1, 600);          % With dt = 1, covers sub-Hz to 10 Hz
w = 2 * pi * f_Hz;
[mag, phase] = bode(Hs, w);
mag = squeeze(mag);
phase = squeeze(phase);

figure;
subplot(2,1,1);
semilogx(f_Hz, 20*log10(mag), 'LineWidth', 1.4);
ylabel('Magnitude (dB)');
title('Bode magnitude and phase of fitted H(s)');
grid on;

subplot(2,1,2);
semilogx(f_Hz, phase, 'LineWidth', 1.4);
xlabel('Frequency (1/time units)');
ylabel('Phase (degrees)');
grid on;
