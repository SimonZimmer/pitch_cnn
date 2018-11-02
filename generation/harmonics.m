function [output] = harmonics(type, numHarmonics, root_freq, rand_range, fs, duration)
% HARMONICS: computes harmonic frequencies applying different decay curves
% duration in s

% TODO: implement limit to avoid mirroring
Ts = 1/fs;
t = 0 : Ts : duration-Ts;

output = zeros(duration*fs, numHarmonics);

    for k = 1 : numHarmonics
        if strcmpi("exponential",type)
            factor = exp(-((k)));
        end
        if strcmpi("linear",type) 
            factor = 1 - (k / 11);
        end
        if strcmpi("hyperbolic",type)
            factor = 1 / (k+1);
        end
        if strcmpi("random",type)
            factor = rand;
        end
        if strcmpi("lin_reciprocal",type)
            factor = ((k) / 10);
        end
        if strcmpi("exp_reciprocal",type)
            factor = exp(k) / exp(numHarmonics);
        end
        new_freq = root_freq * (k+1);
        if new_freq <= fs / 2
            noise = new_freq * (rand_range * rand);
            output(:,k) = sin(2 * pi * (new_freq + noise) * t);
            output(:,k) = factor * normalize(output(:,k), 'range', [-1 1]);
        else
            output(:,k) = 0;
        end
    end
end