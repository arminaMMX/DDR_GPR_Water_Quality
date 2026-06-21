function [YPred_Tran, YPred_Test, YTran, YTest] = Predict_GPR(X, Y)

% ===============================
% 1. Train/Test Split
% ===============================
rng(53)

N = size(X,1);
cv = cvpartition(N,'HoldOut',0.3);

idxTrain = training(cv);
idxTest  = test(cv);

XTran = X(idxTrain,:);
YTran = Y(idxTrain);

XTest  = X(idxTest,:);
YTest  = Y(idxTest);

% ===============================
% 2. Train Optimized GPR Model
% ===============================
gprModel = GPRModelOptimizer(X, Y, true);

% ===============================
% 3. Predict
% ===============================
YPred_Tran_log = predict(gprModel, XTran);
YPred_Test_log = predict(gprModel, XTest);

% Convert back
YPred_Tran = 10.^YPred_Tran_log;
YPred_Test = 10.^YPred_Test_log;


end

