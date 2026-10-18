load('net_elman.mat');
total_params = 0;

for r = 1:size(net.IW, 1)
    for c = 1:size(net.IW, 2)
        if ~isempty(net.IW{r,c})
            total_params = total_params + numel(net.IW{r,c});
        end
    end
end

for r = 1:size(net.LW, 1)
    for c = 1:size(net.LW, 2)
        if ~isempty(net.LW{r,c})
            total_params = total_params + numel(net.LW{r,c});
        end
    end
end

for r = 1:numel(net.b)
    if ~isempty(net.b{r})
        total_params = total_params + numel(net.b{r});
    end
end

fprintf('>> Parámetros exactos (Elman): %d\n', total_params);
