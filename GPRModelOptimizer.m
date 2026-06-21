function gprMdl = GPRModelOptimizer(X, Y, showResult)

if nargin < 3
    showResult = false;
end

% ========== STEP 1: Data Preprocessing ==========
nP = size(X,1);
[X, Y] = remove_outliers(X, Y);
fprintf('Removed %d outliers\n', nP - size(X,1));

Y = log10(Y + eps);

% ========== STEP 2: Train-Test Split ==========
rng(632)

N = size(X,1);
cv = cvpartition(N,'HoldOut',0.3);

train_idx = training(cv);
test_idx  = test(cv);

X_train = X(train_idx, :);
y_train = Y(train_idx);
X_test  = X(test_idx, :);
y_test  = Y(test_idx);

% ========== STEP 3: GPR Model with Optimized Kernel ==========
fprintf('Training GPR Model\n');
% Train with multiple kernels and select best
gprMdl = train_optimized_gpr(X_train, y_train);

% ========== STEP 4: Prediction and Evaluation ==========
fprintf('Model Evaluation: Calibration\n');
% Predict on train set
[train_y_pred_log, train_y_pred_std, y_int] = predict(gprMdl, X_train);

% Convert back from log scale
train_y_pred   = 10.^train_y_pred_log;
train_y_actual = 10.^y_train;

% Calculate performance metrics
metrics = calculate_metrics(train_y_actual, train_y_pred);
disp(metrics);

% ========== STEP 5: Prediction and Evaluation ==========
fprintf('Model Evaluation: Validation\n');
% Predict on train set
[test_y_pred_log, test_y_pred_std, y_int] = predict(gprMdl, X_test);

% Convert back from log scale
test_y_pred   = 10.^test_y_pred_log;
test_y_actual = 10.^y_test;

% Calculate performance metrics
metrics = calculate_metrics(test_y_actual, test_y_pred);
disp(metrics);

if showResult
    % ========== STEP 6: Visualization =========%
    plot_results(test_y_actual, test_y_pred, train_y_actual, train_y_pred, train_y_pred_std, test_y_pred_std);

    % ========== STEP 7: Cross-Validation ==========
    fprintf('Step 6: Cross-Validation\n');
    cv_metrics = cross_validate_gpr(X, Y, 5);
    disp('Cross-validation results:');
    disp(cv_metrics);
end

end

function [cleaned_Rrs, cleaned_Chla] = remove_outliers(Rrs, Chla)
    % Remove outliers based on Chla using modified Z-score
    z_scores = abs((Chla - median(Chla)) / (1.4826 * mad(Chla)));
    outlier_idx = z_scores > 2.5; % Conservative threshold
    cleaned_Rrs = Rrs(~outlier_idx, :);
    cleaned_Chla = Chla(~outlier_idx);
end

function gprMdl = train_optimized_gpr(X_train, y_train)
    % Train GPR with different kernels and select best
    
    % Define kernels to try
    kernels = {
        'squaredexponential', ...
        'ardsquaredexponential', ...
        'matern32', ...
        'matern52', ...
        'rationalquadratic'
        };
            
    best_loss = Inf;
    best_gprMdl = [];
    best_kernel = '';
    
    for k = 1:length(kernels)
        try
            fprintf('Testing kernel: %s\n', kernels{k});
            
            % Train GPR model with current kernel
            gprMdl = fitrgp(X_train, y_train, ...
                            'KernelFunction', kernels{k}, ...
                            'BasisFunction',  'linear', ...
                            'Standardize', true);
            
            % Calculate cross-validation loss
            cv_loss = calculate_cv_loss(gprMdl, X_train, y_train, 5);
            
            fprintf('  CV Loss: %.4f\n', cv_loss);
            
            if cv_loss < best_loss
                best_loss = cv_loss;
                best_gprMdl = gprMdl;
                best_kernel = kernels{k};
            end
        catch ME
            fprintf('  Kernel %s failed: %s\n', kernels{k}, ME.message);
        end
    end
    
    % If all kernels failed, use basic GPR
    if isempty(best_gprMdl)
        fprintf('All kernels failed, using basic GPR\n');
        best_gprMdl = fitrgp(X_train, y_train, 'Standardize', true);
    else
        fprintf('\nBest kernel selected: %s (CV Loss: %.4f)\n', best_kernel, best_loss);
        
        % Fine-tune the best model with hyperparameter optimization
        fprintf('Fine-tune the %s model with hyperparameter optimization\n', best_kernel);
        best_gprMdl = optimize_hyperparameters(best_gprMdl, X_train, y_train);
    end
    
    gprMdl = best_gprMdl;
end

function cv_loss = calculate_cv_loss(gprMdl, X, y, k_folds)
    % Calculate cross-validation loss for a given model
    cv = cvpartition(length(y), 'KFold', k_folds);
    losses = zeros(k_folds, 1);
    
    for i = 1:k_folds
        train_idx = training(cv, i);
        test_idx = test(cv, i);
        
        try
            % Train model on fold using same kernel
            fold_mdl = fitrgp(X(train_idx,:), y(train_idx), ...
                    'KernelFunction', gprMdl.KernelInformation.Name, ...
                    'Standardize', true);
            
            y_pred = predict(fold_mdl, X(test_idx,:));
            losses(i) = mean((y(test_idx) - y_pred).^2);
        catch
            losses(i) = inf;
        end
    end
    
    cv_loss = mean(losses);
end

function optimized_mdl = optimize_hyperparameters(gprMdl, X, y)
    % Optimize hyperparameters using pattern search or fmincon
    try
        % Get current kernel function
        kernel_func = gprMdl.KernelInformation.Name;
        
        % Optimize using fitrgp with automatic hyperparameter optimization
        optimized_mdl = fitrgp(X, y, ...
                        'KernelFunction', kernel_func, ...
                        'BasisFunction',  'linear', ...
                        'Standardize', true, ...
                        'OptimizeHyperparameters', 'auto', ...
                        'HyperparameterOptimizationOptions', struct(...
                            'AcquisitionFunctionName', 'expected-improvement-plus', ...
                            'MaxObjectiveEvaluations', 100, ...
                            'ShowPlots', true, ...
                            'Verbose', 0));
    catch
        % If optimization fails, return original model
        fprintf('Hyperparameter optimization failed, using original model\n');
        optimized_mdl = gprMdl;
    end
end


function loss = cv_loss(params, X, y)
    % Cross-validation loss for hyperparameter optimization
    sigma = params(1);
    try
        gprMdl = fitrgp(X, y, ...
            'KernelFunction', 'squaredexponential', ...
            'Standardize', true, ...
            'Sigma', sigma);
        loss = crossval_loss(gprMdl, X, y, 5);
    catch
        loss = inf;
    end
end

function cv_loss_val = crossval_loss(gprMdl, X, y, k_folds)
    % Custom cross-validation loss
    cv = cvpartition(length(y), 'KFold', k_folds);
    losses = zeros(k_folds, 1);
    
    for i = 1:k_folds
        train_idx = training(cv, i);
        test_idx = test(cv, i);
        
        % Retrain model on fold
        try
            fold_mdl = fitrgp(X(train_idx,:), y(train_idx), ...
                'KernelFunction', gprMdl.KernelInformation.KernelFunction, ...
                'Standardize', true);
            y_pred = predict(fold_mdl, X(test_idx,:));
            losses(i) = mean((y(test_idx) - y_pred).^2);
        catch
            losses(i) = inf;
        end
    end
    
    cv_loss_val = mean(losses);
end


function plot_results(te_y_actual, te_y_pred, tr_y_actual, tr_y_pred, tr_y_pred_std, te_y_pred_std)
    % Comprehensive visualization
    figure('Position', [100, 100, 1200, 500]);
    
    % Plot 1: Scatter plot with 1:1 line
    subplot(1,3,1);
    hold on;
    scatter(tr_y_actual, tr_y_pred, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(te_y_actual, te_y_pred, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    
    min_val = min([tr_y_actual; te_y_pred]);
    max_val = max([tr_y_actual; te_y_pred]);
    plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);

    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')

    xlabel('Measured Chla');
    ylabel('Predicted Chla');
    title('GPR Prediction Performance');
    grid on;
    box on;
    axis equal;
    
    %----------------------------------------------------------------------
    % Plot 2: Residuals
    subplot(1,3,2);
    hold on;
    residuals = tr_y_actual - tr_y_pred;
    scatter(tr_y_pred, residuals, 50, 'filled', 'MarkerFaceAlpha', 0.6);

    residuals = te_y_actual - te_y_pred;
    scatter(te_y_pred, residuals, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    
    yline(0, 'r-', 'LineWidth', 2);
    hold off
    
    xlabel('Predicted Chla');
    ylabel('Residuals');
    title('Residual Plot');
    grid on;
    box on;

    %----------------------------------------------------------------------
    % Plot 3: Prediction intervals (log scale)
    subplot(1,3,3);
    [~, sort_idx] = sort(tr_y_actual);
    tr_y_test_sorted = tr_y_actual(sort_idx);
    tr_y_pred_sorted = tr_y_pred(sort_idx);
    tr_y_std_sorted  = tr_y_pred_std(sort_idx);

    [~, sort_idx] = sort(te_y_actual);
    te_y_test_sorted = te_y_actual(sort_idx);
    te_y_pred_sorted = te_y_pred(sort_idx);
    te_y_std_sorted  = te_y_pred_std(sort_idx);
    
    hold on;
    fill([tr_y_test_sorted; flip(tr_y_test_sorted)], ...
         [tr_y_pred_sorted - 1.96*tr_y_std_sorted; ...
         flip(tr_y_pred_sorted + 1.96*tr_y_std_sorted)], ...
         [0.8, 0.8, 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

    plot(tr_y_test_sorted, tr_y_pred_sorted, 'b-', 'LineWidth', 2);
    plot(tr_y_test_sorted, tr_y_test_sorted, 'r--', 'LineWidth', 2);


    fill([te_y_test_sorted; flip(te_y_test_sorted)], ...
         [te_y_pred_sorted - 1.96*te_y_std_sorted; ...
         flip(te_y_pred_sorted + 1.96*te_y_std_sorted)], ...
         [0.6, 0.6, 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

    plot(te_y_test_sorted, te_y_pred_sorted, 'g-', 'LineWidth', 2);
    plot(te_y_test_sorted, te_y_test_sorted, 'k--', 'LineWidth', 2);


    hold off;
    xlabel('Measured Chla');
    ylabel('Predicted Chla');
    title('Prediction with 95% Confidence Intervals');
    legend('95% CI', 'Predicted', '1:1 Line', 'Location', 'best');
    grid on;
    box on;

    sgtitle('GPR Model Performance for Chla Prediction');
end

function cv_metrics = cross_validate_gpr(features, targets, k_folds)
    % Perform k-fold cross-validation
    cv = cvpartition(length(targets), 'KFold', k_folds);
    all_pred = zeros(size(targets));
    all_actual = targets;
    
    for i = 1:k_folds
        train_idx = training(cv, i);
        test_idx = test(cv, i);
        
        % Train model
        gprMdl = fitrgp(features(train_idx,:), targets(train_idx), ...
            'KernelFunction', 'ardsquaredexponential', ...
            'Standardize', true);
        
        % Predict
        all_pred(test_idx) = predict(gprMdl, features(test_idx,:));
    end
    
    % Convert back from log if necessary
    if max(targets) < 10 % Assuming log-transformed
        all_actual = 10.^all_actual;
        all_pred = 10.^all_pred;
    end
    
    cv_metrics = calculate_metrics(all_actual, all_pred);
end
