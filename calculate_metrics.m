function metrics = calculate_metrics(y_actual, y_pred)
    % Calculate comprehensive performance metrics
    n = length(y_actual);
    
    % Basic metrics
    metrics.RMSE = sqrt(mean((y_actual - y_pred).^2));
    metrics.MAE = mean(abs(y_actual - y_pred));
    metrics.MAPE = mean(abs((y_actual - y_pred) ./ (y_actual + eps))) * 100;
    metrics.R2 = 1 - sum((y_actual - y_pred).^2) / sum((y_actual - mean(y_actual)).^2);
    
    % Log-scale metrics (often better for Chla)
    log_actual = log10(y_actual + eps);
    log_pred = log10(y_pred + eps);
    metrics.RMSE_log = sqrt(mean((log_actual - log_pred).^2));
    metrics.R2_log = 1 - sum((log_actual - log_pred).^2) / sum((log_actual - mean(log_actual)).^2);
    
    % Bias
    metrics.Bias = mean(y_pred - y_actual);
    metrics.Relative_Bias = metrics.Bias / (mean(y_actual) + eps) * 100;
    
    % fprintf('\n========== Performance Metrics ==========\n');
    % fprintf('RMSE: %.4f\n', metrics.RMSE);
    % fprintf('MAE: %.4f\n', metrics.MAE);
    % fprintf('MAPE: %.2f%%\n', metrics.MAPE);
    % fprintf('R²: %.4f\n', metrics.R2);
    % fprintf('RMSE (log10): %.4f\n', metrics.RMSE_log);
    % fprintf('R² (log10): %.4f\n', metrics.R2_log);
    % fprintf('Bias: %.4f (%.2f%%)\n', metrics.Bias, metrics.Relative_Bias);
end
