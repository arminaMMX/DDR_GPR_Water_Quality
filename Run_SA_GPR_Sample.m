%% Chla prediction using SA-GPR model

clear, clc;

%% Load Data
read_surface_spectra

%% Feature Expansion
[features, featureNames] = Feature_Expansion(Rrs, wavelengths, "ShowTable",true);

%% DDR-based feature selection
[SAFeatureValues, SAFeatureNames] = Feature_Selection_SA(features, Chl, wavelengths);

%% DDR-GPR prediction
[ChlPred_Tran, ChlPred_Test, ChlTran, ChlTest] = Predict_GPR(SAFeatureValues, Chl);

%% Metrics
MetricsTran = calculate_metrics(ChlTran, ChlPred_Tran);
MetricsTest = calculate_metrics(ChlTest, ChlPred_Test);

disp('========== SA-GPR Performance Metrics ==========')
fprintf('RMSE Calibration = %.2f; RMSE Validation = %.2f\n', MetricsTran.RMSE, MetricsTest.RMSE)
fprintf('R2 Calibration = %.2f; R2 Validation = %.2f\n', MetricsTran.R2_log, MetricsTest.R2_log)

%%
figure('Position', [10 300 700 550]);
hold on;
scatter(ChlTran, ChlPred_Tran, 60, 'MarkerFaceColor',[.0 .5 .7], ...
        'MarkerEdgeColor',[0 0 1], 'MarkerFaceAlpha',0.5, ...
        'DisplayName','Calibration');

scatter(ChlTest, ChlPred_Test, 60, 'MarkerFaceColor',[.7 .5 .0], ...
        'MarkerEdgeColor',[1 0 0], 'MarkerFaceAlpha',0.5, ...
        'DisplayName','Validation');


plot([0.4 25], [0.4 25], 'k-','LineWidth',1.5, 'HandleVisibility','off');

hold off

xlim([0.4 25])
ylim([0.4 25])
xlabel('measured Chl{\ita} (mg m^{-3})', 'FontSize',18);
ylabel('predicted Chl{\ita} (mg m^{-3})', 'FontSize', 18);

set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')

legend('FontSize', 18)
grid on;
box on
title('DDR-GPR Prediction', 'FontSize',18)