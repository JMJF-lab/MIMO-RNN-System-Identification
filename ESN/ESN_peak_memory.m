% 1. Arrancar el monitor de Unix limpio para ESN
[~, pid_str] = system('echo $$'); pid = strtrim(pid_str);
system(sprintf('while true; do ps -p %s -o rss=; sleep 0.5; done 2>/dev/null > /tmp/esn_memory_log.txt &', pid));
system('echo $! > /tmp/esn_memory_pid');
disp('>> Monitor iniciado. Entrenando ESN...');

tic; % Iniciar cronómetro
try
    % 2. Ejecutar el script de entrenamiento de la ESN
    aaa_entrenarESN_mejor_param; 
    trainingTime = toc;
    disp(['>> ¡Entrenamiento de ESN finalizado con éxito! Tiempo: ', num2str(trainingTime), ' s']);
catch ME
    trainingTime = toc;
    disp(['>> Error: ', ME.message]);
end

% 3. Detener el monitor y extraer el valor máximo con seguridad
system('kill $(cat /tmp/esn_memory_pid) 2>/dev/null; rm -f /tmp/esn_memory_pid');

% Leer el archivo de log línea por línea en MATLAB
fid = fopen('/tmp/esn_memory_log.txt', 'r');
if fid ~= -1
    raw_data = textscan(fid, '%f');
    fclose(fid);
    system('rm -f /tmp/esn_memory_log.txt');
    
    if ~isempty(raw_data{1})
        max_kb = max(raw_data{1});
        peak_mb = max_kb / 1024;
        fprintf('\n========================================\n');
        fprintf('>> ESN Peak RSS (MATLAB) (MB): %.2f MB\n', peak_mb);
        fprintf('========================================\n');
    else
        pid = feature('getpid');
        [~, cmdout] = system(sprintf('ps -o rss= -p %d', pid));
        peak_mb = str2double(strtrim(cmdout)) / 1024;
        fprintf('>> ESN Current Process Memory (RSS): %.2f MB\n', peak_mb);
    end
else
    disp('>> No se pudo abrir el archivo de log.');
end

% 4. Obtener el tamaño en disco de la ESN con su nombre correcto
if exist('best_esn_model.mat', 'file')
    d = dir('best_esn_model.mat');
    esnSizeMB = d.bytes / (1024 * 1024);
    fprintf('>> ESN Model Size (disco): %.5f MB\n', esnSizeMB);
end
