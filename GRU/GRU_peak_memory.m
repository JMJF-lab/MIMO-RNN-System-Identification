% 1. Start a clean Unix monitor for GRU.
[~, pid_str] = system('echo $$'); pid = strtrim(pid_str);
system(sprintf('while true; do ps -p %s -o rss=; sleep 0.5; done 2>/dev/null > /tmp/gru_memory_log.txt &', pid));
system('echo $! > /tmp/gru_memory_pid');
disp('>> Monitor started. Training GRU...');

try
    % 2. Run the GRU training script.
    GRU_training; 
    disp('>> GRU training completed successfully!');
catch ME
    disp(['>> Error: ', ME.message]);
end

% 3. Stop the monitor and safely extract the maximum value.
system('kill $(cat /tmp/gru_memory_pid) 2>/dev/null; rm -f /tmp/gru_memory_pid');

% Read the log file line by line in MATLAB.
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
        % Safety fallback: measure the current process.
        pid = feature('getpid');
        [~, cmdout] = system(sprintf('ps -o rss= -p %d', pid));
        peak_mb = str2double(strtrim(cmdout)) / 1024;
        fprintf('>> GRU Current Process Memory (RSS): %.2f MB\n', peak_mb);
    end
else
    disp('>> Unable to open the log file.');
end
