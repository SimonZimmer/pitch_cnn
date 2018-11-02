function [output] = harmonics(type, numHarmonics, root_freq, rand_range, fs, duration)
% HARMONICS: computes harmonic frequencies applying different decay curves
% duration in s

% TODO: implement limit to avoid mirroring
Ts = 1/fs;
t = 0 : Ts : duration-Ts;

    output = zeros(duration*fs, numHarmonics);

    if strcmpi("exponential",type)
        for k = 1 : numHarmonics
            factor = exp(-((k)));
            new_freq = root_freq * (k+1);
            noise = new_freq * (rand_range * rand);
            output(:,k) = sin(2 * pi * (new_freq + noise) * t);
            output(:,k) = factor * normalize(output(:,k), 'range', [-1 1]);
        end
    end

    if strcmpi("linear",type)
        for k = 1 : numHarmonics
            factor = 1 - (k / 11);
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

    if strcmpi("hyperbolic",type)
        for k = 1 : numHarmonics
            factor = 1 / (k+1);
            new_freq = root_freq * (k+1);
            noise = new_freq * (rand_range * rand);
            output(:,k) =  factor * sin(2 * pi * (new_freq + noise) * t);
        end
    end

    %random decay harmonics
    if strcmpi("random",type)
        for k = 1 : numHarmonics
            factor = rand;
            new_freq = root_freq * (k+1);
            noise = new_freq * (rand_range * rand);
            output(:,k) =  factor * sin(2 * pi * (new_freq + noise) * t);
        end
    end

    %linear reciprocal decay harmonics
    if strcmpi("lin_reciprocal",type)
        for k = 1 : numHarmonics
            factor = ((k) / 10);
            new_freq = root_freq * (k+1);
            noise = new_freq * (rand_range * rand);
            output(:,k) =  factor * sin(2 * pi * (new_freq + noise) * t);
        end
    end

    %exponential reciprocal decay harmonics
    if strcmpi("exp_reciprocal",type)
        for k = 1 : numHarmonics
            factor = exp(k) / exp(numHarmonics);
            new_freq = root_freq * (k+1);
            noise = new_freq * (rand_range * rand);
            output(:,k) =  factor * sin(2 * pi * (new_freq + noise) * t);
        end
    end

end