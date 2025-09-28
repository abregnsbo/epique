%% Custom FIR filter with asymmetric triangular impulse response
riseSamples = 120;                        % Number of samples for the linear rise
fallSamples = ceil(riseSamples / 3);      % Drop back to zero 3x faster

% Construct the piecewise-linear impulse response
rise = linspace(0, 1, riseSamples + 1);   % Includes both 0 and the peak
fall = linspace(1, 0, fallSamples + 1);   % Includes the peak and trailing zero
h = [rise(1:end-1), fall];                % Concatenate without repeating the peak

% Normalize so the filter has unity DC gain (optional but often desirable)
h = h / sum(h);

% Visualize the impulse response
n = 0:numel(h)-1;
stem(n, h, 'filled');
xlabel('Sample');
ylabel('Amplitude');
title('Asymmetric Triangular Impulse Response');
grid on;
