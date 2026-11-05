% 1. Start the Unix monitor with a clean state
[~, pid_str] = system('echo $$'); pid = strtrim(pid_str);
system(sprintf('while true; do ps -p %s -o rss=; sleep 0.5; done 2>/dev/null > /tmp/esn_memory_log.txt &', pid));
system('echo $! > /tmp/esn_memory_pid');
disp('>> Monitor started. Training ESN...');

tic; % Start timer
try
    % 2. Run the ESN training script
    ESN_training; 
    trainingTime = toc;
    disp(['>> ESN training completed successfully! Time:', num2str(trainingTime), ' s']);
catch ME
    trainingTime = toc;
    disp(['>> Error: ', ME.message]);
end

% 3. Stop the monitor and safely extract the maximum value
system('kill $(cat /tmp/esn_memory_pid) 2>/dev/null; rm -f /tmp/esn_memory_pid');

% Read the log file line by line in MATLAB
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
    disp('>> Unable to open the log file.');
end

% 4. Obtain the disk size of the ESN
if exist('best_esn_model.mat', 'file')
    d = dir('best_esn_model.mat');
    esnSizeMB = d.bytes / (1024 * 1024);
    fprintf('>> ESN Model Size (disk): %.5f MB\n', esnSizeMB);
end
