
%%-------------------------------------------------------------------------------------------%%
% Generation of 90112 .wav files with added mixtures of harmonics and white noise distortion  %
%%-------------------------------------------------------------------------------------------%%

%freq = 25.9565436
freq = 440
%C8 bzw n = 88 = 4186.01Hz
freq_max = freq;
i = 0;

% sample rate
fs = 2^14;
NFFT = 2^14;
Ts = 1/fs;    
fNy = fs / 2;
duration = 1.0;
t = 0 : Ts : duration-Ts;
numSamples = length(t);

%while freq <= freq_max

    root(:,1) = root_note(440, 10, 2^14, 1.0);
    root(:,1) = normalize(root(:,1), 'range', [-1 1]);
    
    % generate normalized white noise
    noise = wgn(NFFT,1,1);
    noise = noise / max(noise);

    decay_types = ["linear", "exponential", "hyperbolic", "random", "lin_reciprocal", "exp_reciprocal"];

    figure('NumberTitle', 'off', 'Name', 'sinus_10harmonics_root_damp0.9')
    plot_index = 1;
  
    for inc = 1:6

        harmonic_tones = harmonics(decay_types(inc), 10, freq, numSamples, t, 0);
        output_sum = root(:,1) + sum(harmonic_tones, 2);
        output_sum(:,1) = normalize(output_sum(:,1), 'range', [-1 1]);

        % Windowing
        h = tukeywin(NFFT, 0.01);
        output_sum = output_sum(:,1) .* h;

        filename = sprintf('sinus_10harmonics_%s_decay_root_damp0.9.wav', decay_types(inc));
        audiowrite(filename, output_sum(:,1), fs);

        subplot(3,4, plot_index)
        plot(t, output_sum, 'b')
        xlim([2*1/freq,12*1/freq])
        title('time domain')
        xlabel('time in s')
        ylabel('level')
        plot_index = plot_index + 1;

        max_ref = max(abs(fft(root(:,1))));
        
        subplot(3,4, plot_index)
        x_magnitude = abs(fft(root(:,1)));
        x_magnitude  = x_magnitude(1:end/2+1,:)';
        x_magnitude = x_magnitude / max_ref;
        plot(x_magnitude,'b')
        hold on;
        x_magnitude = abs(fft(sum(harmonic_tones, 2)));
        x_magnitude  = x_magnitude(1:end/2+1,:)';
        x_magnitude = x_magnitude / max_ref;
        plot(x_magnitude,'r')
        hold off
        legend('root', 'harmonics')
        title('frequency domain')
        xlabel('frequency in Hz')
        ylabel('level')
        %ylim([0,1])
        plot_index = plot_index + 1;
    end

    % % Addition der Obertöne
    % for m = 1:10
    %     combinations = combnk(2:11,m);
    % 
    %     for n = 1:size(combinations,1)
    %         y_oberton_summe = zeros(NFFT, 1);
    %         y_grund_oberton = zeros(NFFT, 1);
    %         
    %         for e = 1:m
    %             y_grund_oberton = zeros(NFFT, 1);
    %           
    %             for u = 1:m
    %                 y_oberton_summe(:,1) = y_oberton_summe(:,1) + y(:,(combinations(n,1+m-u)));
    %             end
    %             
    %             y_grund_oberton(:,1) = y(:,1) + 0.5 * y_oberton_summe(:,1);
    %             y_grund_oberton(:,1) = y_grund_oberton(:,1) / max(y_grund_oberton(:,1));
    %             y_grund_oberton(:,1) = 0.5 * y_grund_oberton(:,1) + ((rand * 0.3 + 0.033) * 1 .* noise);
    % 
    %             % Windowing
    %             h = tukeywin(NFFT, 0.1);
    %             y_sum = y_grund_oberton(:,1) .* h;
    %             filename = sprintf('sinus%d_comb%d_oberton%d.wav', i, m, n);
    %             audiowrite(filename, y_grund_oberton(:,1), fs);
    %             
    %             %z = z + 1;
    %             %plotGraphs(y_grund_oberton, t, f, NFFT, z);
    %         end
    %     end
    % end

    %filename = sprintf('sinus_%d.wav', i);
    %audiowrite(filename, output_sum, fs);

    %i = i+1;

%end

% TODO:
% addition of non-harmonics (-> residuals)

