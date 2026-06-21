function [selectedFeature, selectedFeatureNames] = Feature_Selection_DDR(X, y, featureNames)

% =========================================================
%  GP-ARD based DDR FEATURE SELECTION
%
%  PURPOSE:
%  1- Train Gaussian Process Regression (GPR)
%  2- Compute DDR (Derivative Decomposition Ratio)
%  3- Rank spectral bands
%  4- Select optimal subset using cumulative DDR
%  5- Evaluate model performance iteratively
%
%  INPUT:
%     X  -> NxD matrix of Rrs features
%            Example:
%            [Rrs412 Rrs443 Rrs490 Rrs510 Rrs555 Rrs670]
%
%     y  -> Nx1 Chl-a vector
%
%  OUTPUT:
%     - selectedFeature
%     - selectedFeatureNames
%     - Optional: (may be added by user)
%           - DDR importance
%           - Ranked features
%           - Optimal feature subset
%           - RMSE curves
%           - Cumulative DDR curve
%
% =========================================================
%
%
% =========================================================
% PREPROCESSING
% =========================================================

% Log-transform Chl-a
y = log10(y);

% Remove NaN rows
idxValid = all(~isnan(X),2) & ~isnan(y);

X = X(idxValid,:);
y = y(idxValid);

[N,D] = size(X);

% =========================================================
% TRAIN / TEST SPLIT
% =========================================================
rng(41)
cv = cvpartition(N,'HoldOut',0.3);

idxTrain = training(cv);
idxTest  = test(cv);

Xtrain = X(idxTrain,:);
ytrain = y(idxTrain);

Xtest  = X(idxTest,:);
ytest  = y(idxTest);

% =========================================================
% TRAIN GP MODEL
% =========================================================
gprMdl = fitrgp( ...
    Xtrain,...
    ytrain,...
    'KernelFunction','ardsquaredexponential',...
    'Standardize',true);

% =========================================================
% EXTRACT GP PARAMETERS
% =========================================================
alpha = gprMdl.Alpha;
Xtr   = gprMdl.X;
params = gprMdl.KernelInformation.KernelParameters;
sigmaF = params(1);
% ARD length-scales
L = params(2:end);

% =========================================================
% CALCULATE DDR
% =========================================================
DDR = zeros(D,1);
Ntr = size(Xtr,1);

for n = 1:Ntr
    x = Xtr(n,:);
    grad = zeros(D,1);

    % ---------------------------------------------
    % Compute gradient for each feature
    % ---------------------------------------------
    for i = 1:D
        g = 0;
        for j = 1:Ntr
            xj = Xtr(j,:);
            % Ensure row vectors
            x  = x(:)';
            xj = xj(:)';
            Lr = L(:)';

            % Squared distance
            diff2 = ((x - xj).^2) ./ (Lr.^2);

            % Kernel value (scalar)
            k = sigmaF^2 * exp(-0.5 * sum(diff2));

            % Derivative wrt feature i
            dkdx = k * ((xj(i) - x(i)) / (Lr(i)^2));

            % Accumulate scalar gradient
            g = g + alpha(j) * dkdx;

        end

        grad(i) = g;

    end

    % ---------------------------------------------
    % DDR for sample n
    % ---------------------------------------------
    denom = sum(grad.^2) + eps;
    DDRsample = (grad.^2) ./ denom;
    DDR = DDR + DDRsample;
end

% =========================================================
% AVERAGE DDR
% =========================================================
DDR = DDR / Ntr;

% =========================================================
% NORMALIZE DDR
% =========================================================
DDR = DDR ./ sum(DDR);

% =========================================================
% SORT FEATURES
% =========================================================
[DDRsorted,idxSort] = sort(DDR,'descend');
featureRank = featureNames(idxSort);

LSort       = L(idxSort);
DDRSort     = DDR(idxSort);
% =========================================================
% CUMULATIVE DDR
% =========================================================
cumDDR = cumsum(DDRsorted);

% =========================================================
% ITERATIVE MODEL EVALUATION
% =========================================================
RMSEtrain = zeros(D,1);
RMSEtest  = zeros(D,1);
R2train   = zeros(D,1);
R2test    = zeros(D,1);

for k = 1:D
    feat = idxSort(1:k);

    Xtr_k = Xtrain(:,feat);
    Xte_k = Xtest(:,feat);

    % ---------------------------------------------
    % Train GP model
    % ---------------------------------------------
    mdl = fitrgp( ...
        Xtr_k,...
        ytrain,...
        'KernelFunction','squaredexponential',...
        'Standardize',true);

    % ---------------------------------------------
    % Predictions
    % ---------------------------------------------
    yhatTr = predict(mdl,Xtr_k);
    yhatTe = predict(mdl,Xte_k);

    % ---------------------------------------------
    % RMSE
    % ---------------------------------------------
    RMSEtrain(k) = sqrt(mean((ytrain - yhatTr).^2));
    RMSEtest(k)  = sqrt(mean((ytest  - yhatTe).^2));

    % ---------------------------------------------
    % R2
    % ---------------------------------------------
    SSres = sum((ytrain - yhatTr).^2);
    SStot = sum((ytrain - mean(ytrain)).^2);
    R2train(k) = 1 - SSres/SStot;

    SSres = sum((ytest - yhatTe).^2);
    SStot = sum((ytest - mean(ytest)).^2);
    R2test(k) = 1 - SSres/SStot;

end

% =========================================================
% DISPLAY RESULTS
% =========================================================
fprintf('\n================DDR Length-scales:===============\n');
disp(L);

fprintf('\n=========================================\n');
fprintf('DDR FEATURE IMPORTANCE\n');
fprintf('=========================================\n');

for i = 1:D

    fprintf('%d) Band %s nm --> DDR = %.4f\n',...
        i,...
        featureRank{i},...
        DDRsorted(i));

end

% =========================================================
% SELECT OPTIMAL FEATURES
% =========================================================
tau = 0.98; % User-defined threshold
kopt = find(cumDDR >= tau,1,'first');
selectedIdx = idxSort(1:kopt);
selectedFeatureNames = featureNames(selectedIdx);

fprintf('\n=========================================\n');
fprintf('DDR OPTIMAL FEATURES\n');
fprintf('=========================================\n');
fprintf('Threshold = %.2f\n',tau);
fprintf('Optimal Number of Features = %d\n',kopt);

disp(selectedFeatureNames);

% =========================================================
% SELECT OPTIMAL FEATURES
% =========================================================
colIdx = nan(kopt,1);
for i=1:kopt
    fName = selectedFeatureNames(i);
    colIdx(i) = find(ismember(featureNames, fName));
end

selectedFeature = X(:,colIdx);

% =========================================================
% FIGURE 1: DDR IMPORTANCE
% =========================================================
figure;
bar(DDRsorted);
xticks(1:D);
xticklabels(featureRank);
xlabel('Spectral Bands (nm)');
ylabel('DDR Importance');
title('GP-DDR Feature Importance');
grid on;

% =========================================================
% FIGURE 2: CUMULATIVE DDR
% =========================================================
figure;
plot(1:D,cumDDR,'-o','LineWidth',2);
hold on;
yline(tau,'r--','Threshold');
xlabel('Number of Features');
ylabel('Cumulative DDR');
title('Cumulative DDR for Feature Selection');
grid on;

% =========================================================
% FIGURE 3: PERFORMANCE VS FEATURES
% =========================================================
figure;
yyaxis left
hold on
plot(1:D,RMSEtrain,'-s','LineWidth',2, 'Color','b', 'DisplayName','RMSE Train');
plot(1:D,RMSEtest,'-s','LineWidth',2, 'Color','k', 'DisplayName','RMSE Test');
ylabel('RMSE');

hold off
yyaxis right
plot(1:D,cumDDR,'-o','LineWidth',2, 'DisplayName','DDR');
ylabel('Cumulative DDR');
xlabel('Number of Selected Features');
legend()
title('Iterative Feature Selection');
grid on;

% =========================================================
% FIGURE 4: R2
% =========================================================
figure;
plot(1:D,R2test,'-o','LineWidth',2);
xlabel('Number of Selected Features');
ylabel('R^2');
title('Model Performance');
grid on;

% =========================================================
% SUMMARY
% =========================================================
fprintf('\n=========================================\n');
fprintf('SUMMARY\n');
fprintf('=========================================\n');

fprintf('Best RMSE = %.4f\n',min(RMSEtest));
fprintf('Best R2   = %.4f\n',max(R2test));
fprintf('Selected Bands:\n');
disp(selectedFeatureNames);

end

