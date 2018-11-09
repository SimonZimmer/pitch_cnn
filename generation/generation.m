

%% Generation of root + harmonic mixtures with different decay types, added white noise, harmonic_distortion and residual damping

%B1 bzw n = 15
freq = 123.47;
%C6 bzw n = 64
freq_max = 1046.50;
i = 0;

fs = 2^14;
Ts = 1/fs;    
fNy = fs / 2;
duration = 1.0;
t = 0 : Ts : duration-Ts;
numSamples = length(t);

% range = 16-64
start_note = 16;
end_note = 64;
while freq <= freq_max

    freq = freq * 2^(1/12)

    % generate root_note
    root(:,1) = root_note(freq, fs, duration);
    root(:,1) = normalize(root(:,1), 'range', [-1 1]);
    
    % generate normalized white noise
    noise = wgn(fs,1,1);
    noise = noise / max(noise);

    % "linear", "exponential", "hyperbolic", "random", "lin_reciprocal", "exp_reciprocal"
    decay_types = [1, 2, 3, 4, 5, 6];

    % Windowing
    window = tukeywin(fs, 0.01);

    % Addition der Obertöne
    for m = 1:10
        combinations = combnk(1:10,m);
     
        for n = 1:size(combinations,1)
            output_sum = zeros(fs, 1);
            
            for e = 1:m

                for u = 1:m
                    root_damping = rand * 0.8;
                    freq_noise = rand * 0.01;
                    decay_index = floor(5*rand+1);
                    % generate harmonics
                    harmonic_tones = harmonics(decay_types(decay_index), 10, freq, freq_noise, fs, duration);
                    output_sum(:,1) = (root_damping*root(:,1)) + harmonic_tones(:,(combinations(n,m-u+1)));
                    output_sum(:,1) = output_sum(:,1) + ((rand * 0.3) .* noise);
                    output_sum(:,1) = normalize(output_sum(:,1), 'range', [-1 1]);
                end

                % Windowing
                output_sum = output_sum(:,1) .* window;
                filename = sprintf('/Users/simonzimmermann/dev/pitch_cnn/main_training_set/sinus%d_comb%d_oberton%d.wav', i+start_note, m, n);
                audiowrite(filename, output_sum(:,1), fs);
            end
        end
    end

    filename = sprintf('sinus_%d.wav', i);
    audiowrite(filename, root(:,1), fs);

    i = i+1;
end


