function [features, feature_names] = Feature_Expansion(Rrs, wavelengths, opts)

% Generate Spectral Band Combinations and Ratios
% Input Arguments:
%   (Required)
% Rrs        : matrix of size [n_samples x n_wavelengths]
% wavelengths: vector of corresponding wavelengths
%
%   (Optional)
%   ShowTable  : Display Table of Features [true/false]
%   SaveTable  : Save features table to excel file [true/false]
%
% Outputs:
%   features      : Values of spectra and ban-ratios 
%   feature_names : Name of spectra and band-ratios
% -------------------------------------------------------------------------
arguments
    Rrs
    wavelengths
    opts.ShowTable logical = false
    opts.SaveTable logical = false
end
% -------------------------------------------------------------------------
n = length(wavelengths);
% -------------------------------------------------------------------------
if opts.ShowTable
    disp('========== Feature Expansion ==========')

    ratio_table = cell(n+1, n+1);
    ratio_table{1,1} = 'Band(nm)';
    for j = 1:n
        ratio_table{1, j+1} = num2str(wavelengths(j));
    end
    for i = 1:n
        ratio_table{i+1, 1} = num2str(wavelengths(i));
        for j = 1:n
            if j > i
                ratio_table{i+1, j+1} = sprintf('%d/%d', wavelengths(i), wavelengths(j));
            else
                ratio_table{i+1, j+1} = '';
            end
        end
    end
    disp(ratio_table)
end
% -------------------------------------------------------------------------
% Save results to Excel file (optional)
if opts.SaveTable
    writecell(ratio_table, 'band_ratio_matrix.xlsx');
end
% -------------------------------------------------------------------------
% Extract features 
n_samples = size(Rrs, 1);
n_bands = length(wavelengths);

% Calculate number of features
n_ratios = n_bands * (n_bands - 1) / 2;
n_features = n_bands + n_ratios;

% Initialize feature matrix
features = zeros(n_samples, n_features);
feature_names = cell(n_features, 1);

% Add individual bands
for i = 1:n_bands
    features(:, i) = Rrs(:, i);
    feature_names{i} = sprintf('Rrs_%d', wavelengths(i));
end

% Add band ratios
idx = n_bands;
for i = 1:n_bands
    for j = i+1:n_bands
        idx = idx + 1;
        % Avoid division by zero
        denominator = Rrs(:, j);
        denominator(denominator == 0) = eps;
        features(:, idx) = Rrs(:, i) ./ denominator;
        feature_names{idx} = sprintf('Ratio_%d_%d', wavelengths(i), wavelengths(j));
    end
end

end
