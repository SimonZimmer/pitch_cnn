function [output] = harmonics(type, numHarmonics, root_freq, rand_range, fs, duration)
% HARMONICS: computes harmonic frequencies applying different decay curves
% duration in s

% TODO: implement limit to avoid mirroring
Ts = 1/fs;
nyquist = fs / 2;
t = 0 : Ts : duration-Ts;

output = zeros(duration*fs, numHarmonics);

    for k = 1 : numHarmonics
        if strcmpi("exponential",type) || type==1
            factor = exp(-((k)));
        end
        if strcmpi("linear",type) || type==2
            factor = 1 - (k / 11);
        end
        if strcmpi("hyperbolic",type) || type==3
            factor = 1 / (k+1);
        end
        if strcmpi("random",type) || type==4
            factor = rand;
        end
        if strcmpi("lin_reciprocal",type) || type==5
            factor = ((k) / 10);
        end
        if strcmpi("exp_reciprocal",type) || type==6
            factor = exp(k) / exp(numHarmonics);
        end
        new_freq = root_freq * (k+1);
        % prevent generation of harmonics with freq > niquist
        if new_freq <= nyquist
            noise = new_freq * (rand_range * rand);
            output(:,k) = sin(2 * pi * (new_freq + noise) * t);
            output(:,k) = factor * normalize(output(:,k), 'range', [-1 1]);
        else
            output(:,k) = 0;
        end
    end
end