% 1. Arrancar el monitor de Unix limpio para la LSTM
[~, pid_str] = system('echo $$'); pid = strtrim(pid_str);
system(sprintf('while true; do ps -p %s -o rss=; sleep 0.5; done 2>/dev/null > /tmp/lstm_memory_log.txt &', pid));
system('echo $! > /tmp/lstm_memory_pid');
disp('>> Monitor iniciado. Entrenando LSTM (aprox. 20 min)...');

try
    % 2. Ejecutar tu script de entrenamiento de la LSTM
    aa_LSTM_training; 
    disp('>> ¡Entrenamiento finalizado con éxito!');
catch ME
    disp(['>> Error: ', ME.message]);
end

% 3. Detener el monitor y extraer el número máximo con seguridad
system('kill $(cat /tmp/lstm_memory_pid) 2>/dev/null; rm -f /tmp/lstm_memory_pid');

% Leer el archivo de log línea por línea en MATLAB
fid = fopen('/tmp/lstm_memory_log.txt', 'r');
if fid ~= -1
    raw_data = textscan(fid, '%f');
    fclose(fid);
    system('rm -f /tmp/lstm_memory_log.txt');
    
    if ~isempty(raw_data{1})
        max_kb = max(raw_data{1});
        peak_mb = max_kb / 1024;
        fprintf('\n========================================\n');
        fprintf('>> LSTM Peak Training Memory (Unix RSS): %.2f MB\n', peak_mb);
        fprintf('========================================\n');
    else
        disp('>> El log está vacío.');
    end
else
    disp('>> No se pudo abrir el archivo de log.');
end
