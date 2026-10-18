% 1. Arrancar el monitor de Unix limpio para NARX
[~, pid_str] = system('echo $$'); pid = strtrim(pid_str);
system(sprintf('while true; do ps -p %s -o rss=; sleep 0.5; done 2>/dev/null > /tmp/narx_memory_log.txt &', pid));
system('echo $! > /tmp/narx_memory_pid');
disp('>> Monitor iniciado. Entrenando NARX...');

tic; % Iniciar cronómetro de entrenamiento
try
    % 2. Ejecutar tu script de entrenamiento NARX (sin el .m)
    aa_entrenar_NARX; 
    trainingTime = toc;
    disp(['>> ¡Entrenamiento de NARX finalizado con éxito! Tiempo: ', num2str(trainingTime), ' s']);
catch ME
    trainingTime = toc;
    disp(['>> Error: ', ME.message]);
end

% 3. Detener el monitor y extraer el valor máximo con seguridad
system('kill $(cat /tmp/narx_memory_pid) 2>/dev/null; rm -f /tmp/narx_memory_pid');

% Leer el archivo de log línea por línea en MATLAB
fid = fopen('/tmp/narx_memory_log.txt', 'r');
if fid ~= -1
    raw_data = textscan(fid, '%f');
    fclose(fid);
    system('rm -f /tmp/narx_memory_log.txt');
    
    if ~isempty(raw_data{1})
        max_kb = max(raw_data{1});
        peak_mb = max_kb / 1024;
        fprintf('\n========================================\n');
        fprintf('>> NARX Peak RSS (MATLAB) (MB): %.2f MB\n', peak_mb);
        fprintf('========================================\n');
    else
        pid = feature('getpid');
        [~, cmdout] = system(sprintf('ps -o rss= -p %d', pid));
        peak_mb = str2double(strtrim(cmdout)) / 1024;
        fprintf('>> NARX Current Process Memory (RSS): %.2f MB\n', peak_mb);
    end
else
    disp('>> No se pudo abrir el archivo de log.');
end

% 4. Obtener el tamaño del modelo en disco usando d.bytes
if exist('narx_best_model.mat', 'file')
    d = dir('narx_best_model.mat');
    narxSizeMB = d.bytes / (1024 * 1024);
    fprintf('>> NARX Model Size (disco): %.5f MB\n', narxSizeMB);
end
