function [selected_X, selected_features] = Feature_Selection_SA(Rrs, Chl, bands, showResult)

% ==========================================
% GPR + Sensitivity Analysis + Feature Selection
% ==========================================

if nargin < 4
    showResult = true;
end

% ------------------------------------------
% Preprocessing
% ------------------------------------------
y = log10(Chl);

[X, feature_names] = WGAFS04_A_Feature_Expansion(Rrs, bands, true);

% % Standardize
% [X, ~, ~] = zscore(X);

N = size(X,1);
D = size(X,2);

% ==========================================
% 1. TRAIN GPR (ARD kernel)
% ==========================================
gprMdl = fitrgp(X, y, 'KernelFunction','squaredexponential', 'Standardize', true);

Xtrain = gprMdl.X;
ytrain = gprMdl.Y;

% Kernel parameters
params = gprMdl.KernelInformation.KernelParameters;
sigma_f = params(1);
ell = params(2:end);

sigma_n = gprMdl.Sigma;

% Build kernel matrix manually
K = zeros(N, N);

for i = 1:N
    K(:,i) = ard_kernel(Xtrain(i,:), Xtrain, sigma_f, ell);
end

% Add noise variance
K = K + sigma_n^2 * eye(N);

% Inverse
K_inv = inv(K);

% Alpha
alpha = K_inv * ytrain;

% ==========================================
% 2. SENSITIVITY ANALYSIS (MEAN + VAR)
% ==========================================
S_mean = zeros(D,1);
S_var  = zeros(D,1);

for j = 1:D
    
    grad_mu_sq = zeros(N,1);
    grad_var_sq = zeros(N,1);
    
    for i = 1:N
        
        x = Xtrain(i,:);
        
        % Kernel vector k(x, X)
        kx = ard_kernel(x, Xtrain, sigma_f, ell);
        
        % dk/dx_j
        dk = ard_kernel_derivative(x, Xtrain, sigma_f, ell, j);
        
        % ----- Mean derivative -----
        dmu = dk' * alpha;
        grad_mu_sq(i) = dmu^2;
        
        % ----- Variance derivative -----
        v = K_inv * kx;
        ds = -2 * (dk' * v);
        grad_var_sq(i) = ds^2;
        
    end
    
    S_mean(j) = mean(grad_mu_sq);
    S_var(j)  = mean(grad_var_sq);
end

% ==========================================
% 3. FEATURE RANKING
% ==========================================

% Normalize
S_mean_n = S_mean / sum(S_mean);
S_var_n  = S_var / sum(S_var);

% Combined score (paper-consistent idea)
score = S_mean_n ./ S_var_n;

[~, idx] = sort(score, 'descend');

if showResult
    fprintf('\n=== SA Feature Ranking ===\n');
    disp(table(feature_names(idx), S_mean(idx), S_var(idx), score(idx), ...
         'VariableNames', {'Band_nm','S_mean','S_var','Score'}));
end

% ==========================================
% 4. FEATURE SELECTION
% ==========================================

rmse = zeros(D,1);

for k = 1:D
    
    selected = idx(1:k);
    Xk = X(:, selected);
    
    mdl = fitrgp(Xk, y, 'KernelFunction','squaredexponential', 'Standardize', true);
    
    y_pred = predict(mdl, Xk);
    chl_pred = 10.^y_pred;
    
    rmse(k) = sqrt(mean((chl_pred - Chl).^2));
end

% ==========================================
% 5. PLOTS
% ==========================================
if showResult
    % Sensitivity (mean)
    figure;
    bar(feature_names, S_mean);
    xlabel('Wavelength (nm)');
    ylabel('Mean Sensitivity');
    title('Feature Importance (Mean)');
    grid on;

    % Sensitivity (variance)
    figure;
    bar(feature_names, S_var);
    xlabel('Wavelength (nm)');
    ylabel('Variance Sensitivity');
    title('Uncertainty Sensitivity');
    grid on;

    % RMSE vs number of features
    figure;
    plot(1:D, rmse, '-o','LineWidth',1.5);
    xlabel('Number of Features');
    ylabel('RMSE');
    title('Feature Selection Curve');
    grid on;
end
% ==========================================
% 6. FINAL MODEL (optimal features)
% ==========================================

[~, best_k] = min(rmse);
best_features = idx(1:best_k);

if showResult
    fprintf('\nOptimal number of features: %d\n', best_k);
    fprintf('Selected bands:\n');
    disp(feature_names(best_features));
end


selected_features = feature_names(best_features);
selected_X        = X(:,best_features);

end

function k = ard_kernel(x, X, sigma_f, ell)
    % x: 1xD
    % X: NxD
    diff = (X - x) ./ ell;
    sqdist = sum(diff.^2, 2);
    k = sigma_f^2 * exp(-0.5 * sqdist);
end

function dk = ard_kernel_derivative(x, X, sigma_f, ell, j)
    % derivative wrt x_j
    
    diff = (X - x) ./ ell;
    sqdist = sum(diff.^2, 2);
    
    k = sigma_f^2 * exp(-0.5 * sqdist);
    
    dk = k .* ( (X(:,j) - x(j)) / (ell.^2) );
end
