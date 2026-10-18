load('net.mat'); % O tu objeto de red entrenado
total_params = 0;

if isprop(net, 'Learnables') || isfield(net, 'Learnables')
    for i = 1:height(net.Learnables)
        total_params = total_params + numel(net.Learnables.Value{i});
    end
end

fprintf('>> Parámetros exactos (GRU): %d\n', total_params);
