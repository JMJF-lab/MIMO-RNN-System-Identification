load('net_esn.mat'); % Debe contener Wout (y opcionalmente bias)

total_params = numel(Wout);
if exist('bias_out', 'var')
    total_params = total_params + numel(bias_out);
end

fprintf('>> Parámetros exactos (ESN - entrenados): %d\n', total_params);
