%% Read Surface Spectra Data
% Reads Surface_Spectra.csv and stores Chl and surface reflectance spectra
% into separate variables.
%
% File structure:
%   Column 1  - Sample number (e.g. #1, #2, ...)
%   Column 2  - Chl concentration
%   Columns 3-10 - Surface reflectance at wavelengths: 412, 442, 490, 510,
%                  560, 620, 665, 681 nm

%% --- Read the file ---
filename = 'Surface_Spectra.csv';
opts = detectImportOptions(filename);

% Column 1 (Sample#) contains strings like "#1", "#2" — keep as text
opts = setvartype(opts, opts.VariableNames{1}, 'char');

% All remaining columns are numeric
opts = setvartype(opts, opts.VariableNames(2:end), 'double');

T = readtable(filename, opts);

%% --- Extract variables ---

% Chl concentration (column 2), as a numeric column vector [N x 1]
Chl = T{:, 2};

% Wavelengths (nm) parsed from column headers 3-10
wavelength_strs = T.Properties.VariableNames(3:end);   % cell of strings
wavelengths = cellfun(@(s) str2double(s(3:end)), wavelength_strs);    % [1 x 8] numeric array

% Surface reflectance spectra: rows = samples, columns = wavelengths [N x 8]
Rrs = T{:, 3:end};

%% --- Display summary ---
fprintf('File loaded: %s\n', filename);
fprintf('Number of samples : %d\n', size(Rrs, 1));
fprintf('Number of bands   : %d\n', numel(wavelengths));
fprintf('Wavelengths (nm)  : %s\n', num2str(wavelengths));
fprintf('Chl range         : %.4f – %.4f\n', min(Chl), max(Chl));

%% --- Quick visualisation (optional) ---
% figure('Position', [10 300 700 550]);
% 
% subplot(1, 2, 1);
% plot(wavelengths, Rrs', 'LineWidth', 0.8);
% xlabel('Wavelength (nm)');
% ylabel('Surface Reflectance');
% title('Reflectance Spectra');
% grid on;
% 
% subplot(1, 2, 2);
% histogram(Chl, 15);
% xlabel('Chl');
% ylabel('Count');
% title('Chl Distribution');
% grid on;