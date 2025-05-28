%% Import and Process Data from Excel

% Define the file and sheet for each dataset
Wt_E8_fileName = 'S1_Data.xlsx';
Wt_E8_sheetName = 'Wt_E8.5_data';

Gloid_hNMP_filename = 'S2_Data.xlsx';
Gastruloid_sheetName = 'Gastruloid_D5_Data';
hNMP_sheetName = 'hNMP_D3_Data';

% Read the full table from the specified sheet
Wt_E8_data = readtable(Wt_E8_fileName, 'Sheet', Wt_E8_sheetName);
Gloid_data = readtable(Gloid_hNMP_filename, 'Sheet', Gastruloid_sheetName);
hNMP_data = readtable(Gloid_hNMP_filename, 'Sheet', hNMP_sheetName);

% Filter the data where NMPROI == 1 for Gastruloid and Wt E8.5 data
Wt_E8_datanmpData = data(Wt_E8_data.NMPROI == 1, :);
GastruloidNMPlikedata = Gloid_data(Gloid_data.NMPROI == 1, :);

% Extract unique embryo IDs
reps = unique(Wt_E8_datanmpData.Embryo, 'sorted');
nrep = numel(reps);

% Initialize cell arrays
WtE8_Tdata = cell(nrep, 1);
WtE8_Sox2data = cell(nrep, 1);
WtE8_Tbx6data = cell(nrep, 1);

% Loop through each embryo ID and subset the data
for i = 1:nrep
    embryoID = reps(i);

    % Subset for current embryo
    currRows = nmpData.Embryo == embryoID;

    % Extract each gene CV values
    WtE8_Tdata{i} = nmpData.CV_TBXT(currRows);
    WtE8_Sox2data{i} = nmpData.CV_SOX2(currRows);
    WtE8_Tbx6data{i} = nmpData.CV_TBX6(currRows);
end

% Optional: display a message
disp('Data import and processing complete for S1_Data.xlsx.');
%% Data Extraction and Visualization for Wt E8.5 Data

% Plot data using superviolincvcomp function
figure(1);
clf; % Clear the figure

% Plot data with error bars for Sox2, T, and Tbx6
superviolincvcomp(WtE8_Sox2data(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 1, 'Width', 1, 'Colour', '#43a2ca');
superviolincvcomp(WtE8_Tdata(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 5, 'Width', 1, 'Colour', '#78c679');
superviolincvcomp(WtE8_Tbx6data(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 9, 'Width', 1, 'Colour', '#e7298a');

%% Data Extraction and Visualization for hNMP Processed Data (CHIR = 2 and 3)
% Initialize cell arrays for processed data
hNMPdataT = cell(5, 1);
hNMPdataTbx6 = cell(5, 1);
hNMPdataSox2 = cell(5, 1);

% Loop over each replicate index for CHIR = 2
for i = 1:5
    % Extract and clean data for CHIR = 2
    hNMPdataT{i} = table2array(hNMP_data(hNMP_data.Replicate == i & hNMP_data.CHIR == 2 & hNMP_data.logNormT > 0.3, 'CV_TBXT'));
    hNMPdataTbx6{i} = table2array(hNMP_data(hNMP_data.Replicate == i & hNMP_data.CHIR == 2 & hNMP_data.logNormTbx6 > 0.3, 'CV_TBX6'));
    hNMPdataSox2{i} = table2array(hNMP_data(hNMP_data.Replicate == i & hNMP_data.CHIR == 2 & hNMP_data.logNormSox2 > 0.3, 'CV_SOX2'));
    
    % Remove NaN values from the data
    hNMPdataT{i} = hNMPdataT{i}(~isnan(hNMPdataT{i}));
    hNMPdataTbx6{i} = hNMPdataTbx6{i}(~isnan(hNMPdataTbx6{i}));
    hNMPdataSox2{i} = hNMPdataSox2{i}(~isnan(hNMPdataSox2{i}));
end

% Plot data for CHIR = 2
figure(1);
superviolincvcomp(hNMPdataSox2(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 3, 'Width', 0.6, 'Colour', '#43a2ca');
superviolincvcomp(hNMPdataT(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 7, 'Width', 0.6, 'Colour', '#78c679');
superviolincvcomp(hNMPdataTbx6(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 11, 'Width', 0.6, 'Colour', '#e7298a');

% Reinitialize cell arrays for next CHIR value
hNMPdataT = cell(5, 1);
hNMPdataTbx6 = cell(5, 1);
hNMPdataSox2 = cell(5, 1);

% Loop over each replicate index for CHIR = 3
for i = 2:4
    % Extract and clean data for CHIR = 3
    hNMPdataT{i} = table2array(hNMP_data(hNMP_data.Replicate == i & hNMP_data.CHIR == 3 & hNMP_data.logNormT > 0.3, 'CV_TBXT'));
    hNMPdataTbx6{i} = table2array(hNMP_data(hNMP_data.Replicate == i & hNMP_data.CHIR == 3 & hNMP_data.logNormTbx6 > 0.3, 'CV_TBX6'));
    hNMPdataSox2{i} = table2array(hNMP_data(hNMP_data.Replicate == i & hNMP_data.CHIR == 3 & hNMP_data.logNormSox2 > 0.3, 'CV_SOX2'));
    
    % Remove NaN values from the data
    hNMPdataT{i} = hNMPdataT{i}(~isnan(hNMPdataT{i}));
    hNMPdataTbx6{i} = hNMPdataTbx6{i}(~isnan(hNMPdataTbx6{i}));
    hNMPdataSox2{i} = hNMPdataSox2{i}(~isnan(hNMPdataSox2{i}));
end

% Plot data for CHIR = 3
figure(1);
superviolincvcomp(hNMPdataSox2(2:4), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 4, 'Width', 0.6, 'Colour', '#43a2ca');
superviolincvcomp(hNMPdataT(2:4), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 8, 'Width', 0.6, 'Colour', '#78c679');
superviolincvcomp(hNMPdataTbx6(2:4), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 12, 'Width', 0.6, 'Colour', '#e7298a');

%% Data Extraction for Gastruloid Replicates
% Initialize cell arrays to store data
Gloid_Tdata = {};
Gloid_Tbx6data = {};
Gloid_Sox2data = {};

% Get unique replicate values from GastruloidNMPlikeTdata
uniqueReplicates = unique(GastruloidNMPlikedata.Replicate);

% Counter for indexing
counter = 1;

% Loop over each unique replicate
for i = 1:length(uniqueReplicates)
    % Get the current replicate name
    currentReplicate = uniqueReplicates{i};

    % Extract and clean data for each dataset
    Gloid_Tdata{counter} = table2array(GastruloidNMPlikedata(strcmp(GastruloidNMPlikedata.Replicate, currentReplicate), 'CV_TBXT'));
    Gloid_Tbx6data{counter} = table2array(GastruloidNMPlikedata(strcmp(GastruloidNMPlikedata.Replicate, currentReplicate), 'CV_TBX6'));
    Gloid_Sox2data{counter} = table2array(GastruloidNMPlikedata(strcmp(GastruloidNMPlikedata.Replicate, currentReplicate), 'CV_SOX2'));

    % Remove NaN values from the data
    Gloid_Tdata{counter} = Gloid_Tdata{counter}(~isnan(Gloid_Tdata{counter}));
    Gloid_Tbx6data{counter} = Gloid_Tbx6data{counter}(~isnan(Gloid_Tbx6data{counter}));
    Gloid_Sox2data{counter} = Gloid_Sox2data{counter}(~isnan(Gloid_Sox2data{counter}));

    % Increment the counter
    counter = counter + 1;
end

% Plot data for Gastruloid replicates
figure(1);
superviolincvcomp(Gloid_Sox2data(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 2, 'Width', 0.8, 'Colour', '#43a2ca');
superviolincvcomp(Gloid_Tdata(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 6, 'Width', 0.8, 'Colour', '#78c679');
superviolincvcomp(Gloid_Tbx6data(:), 'Errorbars', 'sem', 'Bandwidth', 0.1, 'Xposition', 10, 'Width', 0.8, 'Colour', '#e7298a');

%% Customize plot appearance
ax = gca; % Get current axes handle
set(ax, 'FontSize', 12); % Set font size for axes labels and ticks
set(ax, 'XTick', []); % Remove x-axis tick labels
ax.LineWidth = 1.5; % Set the thickness of the axis lines (e.g., 2 points)
ylabel('Local heterogeneity (CV)', 'FontSize', 14); % Set y-axis label


%% Save Plot
% Define paper size and position
width = 10; % Width in inches
height = 4; % Height in inches
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]); % Position and size on paper
set(gcf, 'PaperSize', [width, height]); % PDF size

%% Save figure as PDF with 800 DPI resolution
print('NMP_CV_Comparison', '-dpdf', '-r800', '-painters');
