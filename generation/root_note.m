function[output] = root_note(freq, fs, duration)
% ROOT_NOTE: computes harmonic frequencies applying different decay curves
% duration in s

    Ts = 1/fs;    
    t = 0 : Ts : duration-Ts;

    % phase
    phi = (rand * 0.1);

    % base frequency
    output(:,1) = sin(2 * pi * freq * t);
end