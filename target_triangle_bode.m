%% Bode plot of the desired triangular impulse response
[sample_count, tri_samples] = deal(1000, 300);
[n, t, h_tri, dt] = target_triangle_impulse(sample_count, tri_samples);

% Frequency axis for Bode-style evaluation (column vectors)
f_Hz = logspace(-3, 1, 600).';
w = 2 * pi * f_Hz;

% Continuous-time frequency response via discrete Laplace approximation
E = exp(-1j * (w * t.'));  % Each row corresponds to a frequency
H_target = dt * (E * h_tri);
mag_target = abs(H_target);
phase_target = unwrap(angle(H_target)) * 180/pi;

% Plot impulse shape and Bode magnitude/phase
figure;
subplot(3,1,1);
plot(n, h_tri, 'LineWidth', 1.5);
xlabel('Sample index');
ylabel('Amplitude');
title('Desired triangular impulse (300-sample window)');
grid on;

subplot(3,1,2);
semilogx(f_Hz, 20*log10(mag_target), 'LineWidth', 1.4);
ylabel('Magnitude (dB)');
title('Bode magnitude of desired triangular impulse');
grid on;

subplot(3,1,3);
semilogx(f_Hz, phase_target, 'LineWidth', 1.4);
xlabel('Frequency (1/time units)');
ylabel('Phase (degrees)');
title('Bode phase of desired triangular impulse');
grid on;
