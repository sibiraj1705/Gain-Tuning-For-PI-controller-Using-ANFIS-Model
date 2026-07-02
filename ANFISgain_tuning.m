%% =========================================================================
% ADAPTIVE PI ANFIS TRAINING - FINAL VERSION
%
% Inputs:
%   In1 = Error_norm      [-1 1]
%   In2 = dError_norm     [-1 1]
%   In3 = Disturbance     [0 1]
%
% Outputs:
%   Kp_corr
%   Ki_corr
%
% Final controller:
%   Kp = Kp_base * Kp_corr
%   Ki = Ki_base * Ki_corr
%% =========================================================================

clear;
clc;
close all;

rng(42);

%% =========================================================================
% USER SETTINGS
%% =========================================================================

err_max  = 1000;
derr_max = 250;

N = 5000;

wc = 0.2;

%% =========================================================================
% BASELINE GAINS AT D = 0.5
%% =========================================================================

D_base = 0.5;

num = [0.00477465 0.000477465];

den = [10,...
       10*D_base + 21,...
       1.0190986*D_base + 2,...
       0.0190986*D_base^2 + 0.0381972*D_base];

G_base = tf(num,den);

C_base = pidtune(G_base,'PI',wc);

Kp_base = C_base.Kp;
Ki_base = C_base.Ki;

fprintf('\nBaseline gains:\n');
fprintf('Kp_base = %.4f\n',Kp_base);
fprintf('Ki_base = %.4f\n\n',Ki_base);

%% =========================================================================
% PREALLOCATE
%% =========================================================================

TrainData_Kp = zeros(N,4);
TrainData_Ki = zeros(N,4);

disp('Generating training data ...')

k = 1;

while k <= N

    %% =====================================================================
    % RANDOM OPERATING POINT
    %% =====================================================================

    Err  = -err_max  + 2*err_max*rand;

    dErr = -derr_max + 2*derr_max*rand;

    % Full disturbance range [0,1]
    D = rand;

    %% =====================================================================
    % PLANT MODEL
    %% =====================================================================

    den = [10,...
           10*D + 21,...
           1.0190986*D + 2,...
           0.0190986*D^2 + 0.0381972*D];

    G = tf(num,den);

    %% =====================================================================
    % NOMINAL PI TUNING
    %% =====================================================================

    try

        C = pidtune(G,'PI',wc);

    catch

        continue;

    end

    Kp_nom = C.Kp;
    Ki_nom = C.Ki;

    %% =====================================================================
    % NORMALIZATION
    %% =====================================================================

    Err_norm  = Err/err_max;
    dErr_norm = dErr/derr_max;

    %% =====================================================================
    % BASE CORRECTION FACTORS
    %% =====================================================================

    Kp_corr = Kp_nom/Kp_base;
    Ki_corr = Ki_nom/Ki_base;

    %% =====================================================================
    % SCHEDULING LOGIC
    %% =====================================================================

    e_mag = abs(Err_norm);

    kp_boost = 1 + 1.0*e_mag;

    ki_scale = 0.6 + 0.2*(1 - e_mag);

    if sign(Err) == sign(dErr)

        kp_boost = kp_boost * 1.10;
        ki_scale = ki_scale * 0.80;

    else

        kp_boost = kp_boost * 0.95;
        ki_scale = ki_scale * 1.10;

    end

    disturbance_factor = 1 + 0.30*D;

    Target_Kp = Kp_corr ...
              * kp_boost ...
              * disturbance_factor;

    Target_Ki = Ki_corr ...
              * ki_scale ...
              * disturbance_factor;

    

    %% =====================================================================
    % LIMIT OUTPUTS
    %% =====================================================================

    Target_Kp = min(max(Target_Kp,0.50),2.00);

    Target_Ki = min(max(Target_Ki,0.50),2.00);

    %% =====================================================================
    % STORE TRAINING DATA
    %% =====================================================================

    TrainData_Kp(k,:) = ...
        [Err_norm dErr_norm D Target_Kp];

    TrainData_Ki(k,:) = ...
        [Err_norm dErr_norm D Target_Ki];

    k = k + 1;

end

disp('Training data generation complete.')

%% =========================================================================
% ADD EXPLICIT EDGE SAMPLES (D = 0 AND D = 1)
%% =========================================================================

disp('Adding disturbance edge samples ...')

D_edge = [0 1];

for D = D_edge

    for Err = [-err_max 0 err_max]

        for dErr = [-derr_max 0 derr_max]

            den = [10,...
                   10*D + 21,...
                   1.0190986*D + 2,...
                   0.0190986*D^2 + 0.0381972*D];

            G = tf(num,den);

            try

                C = pidtune(G,'PI',wc);

            catch

                continue;

            end

            Kp_nom = C.Kp;
            Ki_nom = C.Ki;

            Err_norm  = Err/err_max;
            dErr_norm = dErr/derr_max;

            Kp_corr = Kp_nom/Kp_base;
            Ki_corr = Ki_nom/Ki_base;

            e_mag = abs(Err_norm);

            kp_boost = 1 + 1.0*e_mag;

            ki_scale = 0.6 + 0.2*(1 - e_mag);

            if sign(Err) == sign(dErr)

                kp_boost = kp_boost * 1.10;
                ki_scale = ki_scale * 0.80;

            else

                kp_boost = kp_boost * 0.95;
                ki_scale = ki_scale * 1.10;

            end

            disturbance_factor = 1 + 0.30*D;

            Target_Kp = Kp_corr ...
                      * kp_boost ...
                      * disturbance_factor;

            Target_Ki = Ki_corr ...
                      * ki_scale ...
                      * disturbance_factor;

            Target_Kp = min(max(Target_Kp,0.50),2.00);

            Target_Ki = min(max(Target_Ki,0.50),2.00);

            TrainData_Kp = [TrainData_Kp;
                            Err_norm dErr_norm D Target_Kp];

            TrainData_Ki = [TrainData_Ki;
                            Err_norm dErr_norm D Target_Ki];

        end
    end
end

disp('Edge samples added.')

%% =========================================================================
% SPLIT DATA
%% =========================================================================

N_total = size(TrainData_Kp,1);

idx = randperm(N_total);

Ntrain = round(0.8*N_total);

train_idx = idx(1:Ntrain);
check_idx = idx(Ntrain+1:end);

trainKp = TrainData_Kp(train_idx,:);
checkKp = TrainData_Kp(check_idx,:);

trainKi = TrainData_Ki(train_idx,:);
checkKi = TrainData_Ki(check_idx,:);

%% =========================================================================
% INITIAL FIS
%% =========================================================================

opt = genfisOptions('GridPartition');

opt.NumMembershipFunctions = [7 7 5];

opt.InputMembershipFunctionType = 'gaussmf';

fisKp = genfis(trainKp(:,1:3),trainKp(:,4),opt);

fisKi = genfis(trainKi(:,1:3),trainKi(:,4),opt);

%% =========================================================================
% TRAIN Kp
%% =========================================================================

disp('Training Kp ANFIS ...')

[~,trainErrKp,~,bestFisKp,valErrKp] = ...
    anfis(trainKp,...
          fisKp,...
          [50 0 0.01 0.9 1.1],...
          [],...
          checkKp);

%% =========================================================================
% TRAIN Ki
%% =========================================================================

disp('Training Ki ANFIS ...')

[~,trainErrKi,~,bestFisKi,valErrKi] = ...
    anfis(trainKi,...
          fisKi,...
          [50 0 0.01 0.9 1.1],...
          [],...
          checkKi);

%% =========================================================================
% EXPORT FIS FILES
%% =========================================================================

writeFIS(bestFisKp,'Kp_Adaptive_Sugeno');

writeFIS(bestFisKi,'Ki_Adaptive_Sugeno');


assignin('base','Kp_Model03',bestFisKp);
assignin('base','Ki_Model03',bestFisKi);

assignin('base','Kp_base',Kp_base);
assignin('base','Ki_base',Ki_base);

disp('FIS files exported.')

%% =========================================================================
% TRAINING PLOTS
%% =========================================================================

figure;
plot(trainErrKp,'LineWidth',1.5);
hold on;
plot(valErrKp,'LineWidth',1.5);
grid on;
xlabel('Epoch');
ylabel('RMSE');
title('Kp ANFIS Training');
legend('Training','Validation');

figure;
plot(trainErrKi,'LineWidth',1.5);
hold on;
plot(valErrKi,'LineWidth',1.5);
grid on;
xlabel('Epoch');
ylabel('RMSE');
title('Ki ANFIS Training');
legend('Training','Validation');

%% =========================================================================
% DISPLAY IMPLEMENTATION VALUES
%% =========================================================================

fprintf('\nUse these values in Simulink:\n');

fprintf('\nKp = %.4f * Kp_corr\n',Kp_base);
fprintf('Ki = %.4f * Ki_corr\n',Ki_base);

fprintf('\nError normalization gain = 1/%d\n',err_max);
fprintf('dError normalization gain = 1/%d\n',derr_max);

Kb = Ki_base/Kp_base;

fprintf('\nRecommended anti-windup:\n');
fprintf('Kb = %.4f\n',Kb);

disp(' ');
disp('SUCCESS!');
disp('Load Kp_Model02 and Ki_Model02 into Simulink.');