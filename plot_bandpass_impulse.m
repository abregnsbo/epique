%% Plot impulse response of a designed bandpass filter
fs = 1000;                    % Sampling frequency in Hz
fpass = [100 200];            % Passband edge frequencies in Hz
filterOrder = 50;             % Filter order (even number for symmetric FIR)

% Design an FIR bandpass filter using a Kaiser window for decent sidelobe control
beta = 3;                     % Kaiser window beta sets sidelobe attenuation
b = fir1(filterOrder, fpass/(fs/2), 'bandpass', kaiser(filterOrder + 1, beta));
a = 1;                        % FIR denominator coefficients

% Compute impulse response samples long enough to expose decay
impulseSamples = 256;
[h, n] = impz(b, a, impulseSamples);

% Plot the impulse response in discrete-time and time axes
stem(n/fs, h, 'filled');
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Impulse Response of FIR Bandpass Filter');
grid on;

% Mark the constant group delay of this linear-phase FIR filter
hold on;
groupDelay = filterOrder / 2;           % samples
xline(groupDelay / fs, '--r', 'Group Delay');
hold off;
