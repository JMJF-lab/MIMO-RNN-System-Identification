% 1. Start a clean Unix monitor for the LSTM.
[~, pid_str] = system('echo $$'); pid = strtrim(pid_str);
system(sprintf('while true; do ps -p %s -o rss=; sleep 0.5; done 2>/dev/null > /tmp/lstm_memory_log.txt &', pid));
system('echo $! > /tmp/lstm_memory_pid');
disp('>> Monitor started. Training LSTM. (aprox. 20 min)...');

try
    % 2. Run the LSTM training script.
    LSTM_training; 
    disp('>> Training completed successfully!');
catch ME
    disp(['>> Error: ', ME.message]);
end

% 3. Stop the monitor and safely extract the maximum value.
system('kill $(cat /tmp/lstm_memory_pid) 2>/dev/null; rm -f /tmp/lstm_memory_pid');

% Read the log file line by line in MATLAB.
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
        disp('>> The log file is empty.');
    end
else
    disp('>> Unable to open the log file.');
end
