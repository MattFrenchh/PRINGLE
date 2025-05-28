%% Setup: Define File and Sheet Info
Wt_E8_fileName = 'S1_Data.xlsx';
Wt_E8_sheetName = 'Wt_E8.5_data';

%% Data Import
% Read the full table from the specified sheet
Wt_E8_data = readtable(Wt_E8_fileName, 'Sheet', Wt_E8_sheetName);

%% Filtering: NMPROI == 1 and Marker Positivity
% Subset for NMP cells only
Wt_E8_datanmpData = Wt_E8_data(Wt_E8_data.NMPROI == 1, :);

% Subset based on gene marker expression
Sox2posNMP  = Wt_E8_datanmpData(Wt_E8_datanmpData.Sox2pos  == "Sox2+",  :);
TposNMP     = Wt_E8_datanmpData(Wt_E8_datanmpData.Tpos     == "T+",     :);
Tbx6posNMP  = Wt_E8_datanmpData(Wt_E8_datanmpData.Tbx6pos  == "Tbx6+",  :);

%% Initialize Variables
reps = unique(Wt_E8_datanmpData.Embryo);
nrep = numel(reps);
dataSox2 = cell(nrep, 1);
dataT    = cell(nrep, 1);
dataTbx6 = cell(nrep, 1);

%% Extract CV Data by Embryo
for i = 1:nrep
    dataSox2{i} = table2array(Sox2posNMP(Sox2posNMP.Embryo == i, 'CV_SOX2'));
    dataT{i}    = table2array(TposNMP(TposNMP.Embryo == i, 'CV_TBXT'));
    dataTbx6{i} = table2array(Tbx6posNMP(Tbx6posNMP.Embryo == i, 'CV_TBX6'));
end

%% Visualization: Violin Plots
figure(1); clf;

% Sox2 (Position 1)
superviolin(dataSox2, 1, 'Errorbars', 'sem', 'Bandwidth', 0.1, ...
    'Xposition', 1, 'Width', 0.8, 'FaceAlpha', 0.2);

% TBXT (Position 2)
superviolin(dataT, 2, 'Errorbars', 'sem', 'Bandwidth', 0.1, ...
    'Xposition', 2, 'Width', 0.8, 'FaceAlpha', 0.2);

% TBX6 (Position 3)
superviolin(dataTbx6, 3, 'Errorbars', 'sem', 'Bandwidth', 0.1, ...
    'Xposition', 3, 'Width', 0.8, 'FaceAlpha', 0.2);

% Formatting
set(gca, 'FontSize', 20, 'XTick', []);
xlim([0.5, 3.5]);
