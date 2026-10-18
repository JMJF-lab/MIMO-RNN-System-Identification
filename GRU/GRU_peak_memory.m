% 1. Arrancar el monitor de Unix limpio para GRU
[~, pid_str] = system('echo $$'); pid = strtrim(pid_str);
system(sprintf('while true; do ps -p %s -o rss=; sleep 0.5; done 2>/dev/null > /tmp/gru_memory_log.txt &', pid));
system('echo $! > /tmp/gru_memory_pid');
disp('>> Monitor iniciado. Entrenando GRU...');

try
    % 2. Ejecutar tu script de entrenamiento de GRU (sin el .m)
    AA_entrenar_GRU; 
    disp('>> ¡Entrenamiento de GRU finalizado con éxito!');
catch ME
    disp(['>> Error: ', ME.message]);
end

% 3. Detener el monitor y extraer el número máximo con seguridad
system('kill $(cat /tmp/gru_memory_pid) 2>/dev/null; rm -f /tmp/gru_memory_pid');

% Leer el archivo de log línea por línea en MATLAB
fid = fopen('/tmp/gru_memory_log.txt', 'r');
if fid ~= -1
    raw_data = textscan(fid, '%f');
    fclose(fid);
    system('rm -f /tmp/gru_memory_log.txt');
    
    if ~isempty(raw_data{1})
        max_kb = max(raw_data{1});
        peak_mb = max_kb / 1024;
        fprintf('\n========================================\n');
        fprintf('>> GRU Peak RSS (MATLAB) (MB): %.2f MB\n', peak_mb);
        fprintf('========================================\n');
    else
        % Fallback de seguridad midiendo el proceso actual
        pid = feature('getpid');
        [~, cmdout] = system(sprintf('ps -o rss= -p %d', pid));
        peak_mb = str2double(strtrim(cmdout)) / 1024;
        fprintf('>> GRU Current Process Memory (RSS): %.2f MB\n', peak_mb);
    end
else
    disp('>> No se pudo abrir el archivo de log.');
end
