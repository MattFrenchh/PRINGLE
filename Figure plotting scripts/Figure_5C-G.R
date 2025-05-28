library(ggplot2)
library(scico)
library(patchwork)
library(reshape2)
library(dplyr)
library(emdist)
library(readxl)

# Import data
Wt_E8_data <- as.data.frame(read_excel("S1_Data.xlsx", sheet = "Wt_E8.5_data", col_names = TRUE))
NMPoutlines <- as.data.frame(read_excel("S1_Data.xlsx", sheet = "WtE8_NMP_ROI_outlines", col_names = TRUE))

# Figure 5B ---------------------------------------------------------------

# Perform Welch's paired test to compare CV values between TFs in the NMP 
# region

NMPROI_data = Wt_E8_data[Wt_E8_data$NMPROI==1,] %>% #Isolate cells in the bi-fated NMP ROI
  group_by(Embryo) %>%
  summarise(mean_CV_SOX2=mean(CV_SOX2),
            mean_CV_TBX6=mean(CV_TBX6),
            mean_CV_TBXT=mean(CV_TBXT))


# Extract paired vectors
CV_SOX2_PV <- NMPROI_data$mean_CV_SOX2
CV_TBX6_PV <- NMPROI_data$mean_CV_TBX6
CV_TBXT_PV <- NMPROI_data$mean_CV_TBXT

# Paired t-tests with unequal variances - defaults to Welch's
test_1 <- t.test(CV_SOX2_PV, CV_TBX6_PV, paired = TRUE, var.equal = FALSE)
test_2 <- t.test(CV_SOX2_PV, CV_TBXT_PV,    paired = TRUE, var.equal = FALSE)
test_3 <- t.test(CV_TBX6_PV, CV_TBXT_PV,    paired = TRUE, var.equal = FALSE)

# Collect raw p-values
raw_pvals <- c(test_1$p.value, test_2$p.value, test_3$p.value)

# Adjust for multiple testing
adj_pvals <- p.adjust(raw_pvals, method = "BH")

# Organize and print
results <- data.frame(
  Comparison = c("Sox2 vs Tbx6", "Sox2 vs T", "Tbx6 vs T"),
  Raw_P = raw_pvals,
  Adjusted_P = adj_pvals
)

print(results)

#This plotted graph is displayed as violin Superplots in MATLAB script

# Figure 5 C -------------------------------------------------------------


binning <- function(epi_data, yparam, xparam) {
  # Parameters:
  # epi_data: Data frame with 3 columns: x, y, value 
  # yparam: vector of form c(bin_width, min_y, max_y)
  # xparam: vector of form c(bin_width, min_x, max_x)
  
  ybinw <- yparam[1]
  xbinw <- xparam[1]
  yseq <- seq(yparam[2], yparam[3], by = ybinw)
  xseq <- seq(xparam[2], xparam[3], by = xbinw)
  
  # Initialize matrix to store mean values per bin
  Tbins <- matrix(ncol = length(yseq), nrow = length(xseq))
  colnames(Tbins) <- yseq
  rownames(Tbins) <- xseq
  
  # Loop through x and y bin centers
  for (x in seq_along(xseq)) {
    for (y in seq_along(yseq)) {
      
      # Select data points within bin (double bin width buffer)
      sub <- epi_data[
        epi_data[, 2] <= (yseq[y] + (ybinw * 2)) &
          epi_data[, 2] >  (yseq[y] - (ybinw * 2)) &
          epi_data[, 1] <= (xseq[x] + (xbinw * 2)) &
          epi_data[, 1] >  (xseq[x] - (xbinw * 2)),
      ]
      
      # Compute mean value of the third column in the subset
      Tbins[x, y] <- mean(sub[, 3], na.rm = TRUE)
    }
  }
  
  # Return binned data as long-format data frame for ggplot2
  return(as.data.frame(melt(Tbins)))
}
# Helper to normalize a column
normalize <- function(x) {
  rng <- range(x, na.rm = TRUE)
  (x - rng[1]) / diff(rng)
}




# Define SP stages
SPs <- c(4, 6, 8, 10)

# Define embryo outlines
SPmasks <- list()

for (i in seq_along(SPs)) {
  sp <- SPs[i]
  SP_Data <- Wt_E8_data[Wt_E8_data$SP == sp, ]
  
  # Get convex hull and close it
  ch <- chull(SP_Data$Scaled_Rel_LR_Position, SP_Data$Scaled_Rel_AP_Position)
  convex_hull <- SP_Data[ch, c('Scaled_Rel_LR_Position', 'Scaled_Rel_AP_Position')]
  convex_hull$SP <- sp
  convex_hull <- rbind(convex_hull, convex_hull[1, ])  # Close polygon
  
  # Get outer rectangle coordinates and close it
  x_range <- range(SP_Data$Scaled_Rel_LR_Position, na.rm = TRUE) + c(-30, 30)
  y_range <- range(SP_Data$Scaled_Rel_AP_Position, na.rm = TRUE) + c(-30, 30)
  
  outer_rect <- data.frame(
    Scaled_Rel_LR_Position = c(x_range[2], x_range[1], x_range[1], x_range[2], x_range[2]),
    Scaled_Rel_AP_Position = c(y_range[1], y_range[1], y_range[2], y_range[2], y_range[1]),
    SP = sp
  )
  outer_rect <- rbind(outer_rect, outer_rect[1, ])  # Close polygon
  
  # Reverse convex hull to form the hole (and already closed)
  hole <- convex_hull[nrow(convex_hull):1, ]
  
  # Combine mask
  mask_polygon <- rbind(outer_rect, hole)
  mask_polygon$group <- sp  # Optional: use for grouping in plots
  
  SPmasks[[i]] <- mask_polygon
}

SPmasks <- do.call(rbind, SPmasks)


# Define parameters
yparamsnorm <- c(12, -100, 450)
xparamsnorm <- c(12, -200, 200)
TF_names <- c("SOX2", "TBXT", "TBX6")
TFs <- c("CV_SOX2", "CV_TBXT", "CV_TBX6")
# low_perc = quantile(Wt_E8_data$SOX2[Wt_E8_data$SP==10], 0.1)
# high_qannt = quantile(Wt_E8_data$SOX2[Wt_E8_data$SP==10], 0.95)
# 
# Wt_E8_data$SOX2[Wt_E8_data$SP==10]<- (Wt_E8_data$SOX2[Wt_E8_data$SP==10] - low_perc)/(high_qannt - low_perc) * (max(Wt_E8_data$SOX2)*0.9)

# Run binning for each SP and TF
binned_list <- list()

for (i in seq_along(TFs)) {
  tf <- TFs[i]
  tf_name <- TF_names[i]
  
  binned_tf <- lapply(SPs, function(sp) {
    df <- Wt_E8_data[Wt_E8_data$SP == sp, c("Scaled_Rel_LR_Position", "Scaled_Rel_AP_Position", tf)]
    binned <- binning(df, yparamsnorm, xparamsnorm)
    binned$SP <- sp
    return(binned)
  })
  
  combined <- do.call(rbind, binned_tf)
  combined$TF <- tf_name
  
  binned_list[[i]] <- combined
}

# Combine all into one data frame
all <- do.call(rbind, binned_list)

# Clip values to limits of displayed values
# Needs to be done before plotting as the colours of stat_contour_filled() output are set with full range,
# but the values are averaged in bins and colour limits are lower. But changing limits in stat_contour_filled() causes issues
# as it produces a discretised colour scheme
all$value[all$value>0.4] <-0.4

# Plot maps with ROI outlines

ggplot() + 
  # Contour fill and line layers
  stat_contour_filled(
    data = all,
    aes(x = Var1, y = Var2, z = value),
    bins = 12
  ) +
  geom_contour(
    data = all,
    aes(x = Var1, y = Var2, z = value, colour = after_stat(level)),
    bins = 12,
    size = 0.1
  ) +
  
  # Outline segments (e.g., NMP regions)
  geom_segment(
    data = NMPoutlines,
    aes(x = XS, xend = XE, y = YS, yend = YE),
    color = 'white',
    alpha = 0.6,
    size = 0.2,
    linejoin = 'round'
  ) +
  
  # Center point marker of NSB
  geom_point(
    data = data.frame(x = 0, y = 0),
    aes(x, y),
    size = 2,
    col = 'black',
    pch = 21,
    fill = 'transparent'
  ) +
  
  # White masks to cut out the convex hull regions
  geom_polygon(
    data = SPmasks,
    aes(x = Scaled_Rel_LR_Position, y = Scaled_Rel_AP_Position, subgroup = SP),
    fill = 'white'
  ) +
  
  # Themes and axis formatting
  theme_classic() +
  theme(
    text = element_text(size = 8),
    strip.text.x = element_text(size = 12),
    strip.text.y = element_text(size = 12),
    axis.text = element_text(size = 8),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    title = element_text(size = 0),
    legend.position = "none"
  ) +
  labs(x = "", y = "") +
  
  # Color scales
  scale_fill_viridis_d(option = 'magma') +
  scale_color_viridis_c(option = 'magma') +
  
  
  # Faceting by TF and SP
  facet_grid(
    factor(TF, levels = c('SOX2', 'TBXT', 'TBX6')) ~ as.numeric(SP)
  )+
  
  coord_fixed()


# Figure 5D --------------------------------------------------------------

# Manually defined pseudospace limits from fate map
Pseudospace_lims = c(2.4, 4.8)

# Ensure the factor levels for somite pair levels are set correctly
Wt_E8_data$SP_char <- factor(as.character(Wt_E8_data$SP), levels = c("4", "6", "8", "10"))

# Common theme and aesthetics
base_theme <- theme_classic() +
  theme(panel.background = element_rect(fill = 'grey60'),
        axis.line = element_line(size = 1.5),
        axis.ticks = element_line(size = 1.5),
        text = element_text(size = 20))

base_aes <- aes(x = Pseudospace,
                group = SP_char,
                fill = SP_char,
                colour = SP_char)

base_labs_NFI <- labs(x = 'Pseudospace',
                      y = 'NFI (a.u.)',
                      color = 'Somite-\npair\nstage',
                      fill = 'Somite-\npair\nstage')

base_labs_NbrGrad <- labs(x = 'Pseudospace',
                          y = '3D NFI gradient\nsteepness (a.u.)',
                          color = 'Somite-\npair\nstage',
                          fill = 'Somite-\npair\nstage')

vline_layer <- geom_vline(xintercept = Pseudospace_lims,
                          size = 2, linetype = 'dashed')

# Individual plots of NFI
plot_sox2_CV <- ggplot(Wt_E8_data, aes(y = CV_SOX2)) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  coord_cartesian(ylim = c(0, 0.4)) + 
  base_theme + base_labs_NFI + vline_layer

plot_tbxt_CV <- ggplot(Wt_E8_data, aes(y = CV_TBXT)) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  coord_cartesian(ylim = c(0, 0.4)) + 
  base_theme + base_labs_NFI + vline_layer

plot_tbx6_CV <- ggplot(Wt_E8_data, aes(y = CV_TBX6)) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  coord_cartesian(ylim = c(0, 0.4)) + 
  base_theme + base_labs_NFI + vline_layer

# Combine to one plot
plot_sox2_CV / plot_tbxt_CV / plot_tbx6_CV



# Figure 5F, Gi, Gii ---------------------------------------------------------------

#Set positive gates (in logarithms) based on pairwise plots
Tbx6pos = 7.68
Sox2pos = 7.1
Tpos=7.5

# Define parameters
maxpseudo <- max(Wt_E8_data[!is.na(Wt_E8_data$Pseudospace), 'Pseudospace'])
binning_factor <- 2

# Create pseudoblock: along evenly spaced bins along pseudospace
Wt_E8_data$Pseudoblock <- round((Wt_E8_data$Pseudospace * binning_factor) / maxpseudo, digits = 0.5)
Wt_E8_data$Pseudoblock <- (Wt_E8_data$Pseudoblock * maxpseudo) / binning_factor

# Update maxpseudo and filter data
maxpseudo <- max(Wt_E8_data[!is.na(Wt_E8_data$Pseudospace), 'Pseudospace'])
pseudospace_Sub <- Wt_E8_data[!is.na(Wt_E8_data$Pseudospace),]

# Get unique pseudoblocks
pseudoblocks <- unique(pseudospace_Sub$Pseudoblock[!is.na(pseudospace_Sub$Pseudoblock)])
pseudoblocks <- pseudoblocks[order(pseudoblocks)]

# Define the number of bins for histograms
bins <- 200

# Function to calculate normalized histograms
calculate_normalized_histogram <- function(data, bins, range_min, range_max) {
  hist_data <- hist(data, breaks = seq(range_min, range_max, length.out = bins + 1), plot = FALSE)
  as.matrix(hist_data$counts / sum(hist_data$counts))  # Normalize by the total number of data points
}

# Determine global min and max values across relevant columns
global_min <- min(c(
  min(pseudospace_Sub$NbrR_TBX6[is.finite(pseudospace_Sub$NbrR_TBX6)], na.rm = TRUE),
  min(pseudospace_Sub$NbrR_TBX6_Sim[is.finite(pseudospace_Sub$NbrR_TBX6_Sim)], na.rm = TRUE),
  min(pseudospace_Sub$NbrR_TBXT[is.finite(pseudospace_Sub$NbrR_TBXT)], na.rm = TRUE),
  min(pseudospace_Sub$NbrR_TBXT_Sim[is.finite(pseudospace_Sub$NbrR_TBXT_Sim)], na.rm = TRUE),
  min(pseudospace_Sub$NbrR_SOX2[is.finite(pseudospace_Sub$NbrR_SOX2)], na.rm = TRUE),
  min(pseudospace_Sub$NbrR_SOX2_Sim[is.finite(pseudospace_Sub$NbrR_SOX2_Sim)], na.rm = TRUE)
))

global_max <- max(c(
  max(pseudospace_Sub$NbrR_TBX6[is.finite(pseudospace_Sub$NbrR_TBX6)], na.rm = TRUE),
  max(pseudospace_Sub$NbrR_TBX6_Sim[is.finite(pseudospace_Sub$NbrR_TBX6_Sim)], na.rm = TRUE),
  max(pseudospace_Sub$NbrR_TBXT[is.finite(pseudospace_Sub$NbrR_TBXT)], na.rm = TRUE),
  max(pseudospace_Sub$NbrR_TBXT_Sim[is.finite(pseudospace_Sub$NbrR_TBXT_Sim)], na.rm = TRUE),
  max(pseudospace_Sub$NbrR_SOX2[is.finite(pseudospace_Sub$NbrR_SOX2)], na.rm = TRUE),
  max(pseudospace_Sub$NbrR_SOX2_Sim[is.finite(pseudospace_Sub$NbrR_SOX2_Sim)], na.rm = TRUE)
))

# Define transcription factors
TFs <- c("SOX2", "TBXT", "TBX6")

# Initialize the data frame to store Earth Mover's Distance (EMD) results
EMDdiff_alldata <- data.frame(
  pseudoblocks = rep(pseudoblocks, length(unique(pseudospace_Sub$Embryo)) * length(TFs)),
  embryo = rep(rep(unique(pseudospace_Sub$Embryo), each = length(pseudoblocks)), length(TFs)),
  EMD = NaN,
  CondTF = rep(TFs, each = length(pseudoblocks) * length(unique(pseudospace_Sub$Embryo)))
)

# Function to filter out outliers using the IQR method
filter_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  x[x >= (Q1 - 1.5 * IQR) & x <= (Q3 + 1.5 * IQR)]
}

# Define the minimum number of data points required
limit <- 1

# Calculate quantile thresholds for filtering 'high' cells: top 50th percentile
Sox2poshigh <- quantile(Wt_E8_data$log_SOX2[Wt_E8_data$log_SOX2 > Sox2pos], 0.5, na.rm=TRUE)
Tposhigh <- quantile(Wt_E8_data$log_TBXT[Wt_E8_data$log_TBXT > Tpos], 0.5, na.rm=TRUE)
Tbx6poshigh <- quantile(Wt_E8_data$log_TBX6[Wt_E8_data$log_TBX6 > Tbx6pos], 0.5, na.rm=TRUE)

# Loop through each pseudoblock and embryo to compute EMD
for (i in unique(pseudospace_Sub$Pseudoblock)) {
  print(i)
  for (embryo in unique(pseudospace_Sub$Embryo)) {
    
    # Filter data for the current pseudoblock and embryo
    Sox2_raw_data <- pseudospace_Sub$NbrR_SOX2[
      (pseudospace_Sub$Pseudoblock == i) &
        (pseudospace_Sub$Embryo == embryo) &
        (pseudospace_Sub$log_SOX2 > Sox2poshigh)
    ]
    Sox2_sim_data <- pseudospace_Sub$NbrR_SOX2_Sim[
      (pseudospace_Sub$Pseudoblock == i) &
        (pseudospace_Sub$Embryo == embryo) &
        (pseudospace_Sub$log_SOX2_Sim > Sox2poshigh)
    ]
    
    T_raw_data <- pseudospace_Sub$NbrR_TBXT[
      (pseudospace_Sub$Pseudoblock == i) &
        (pseudospace_Sub$Embryo == embryo) &
        (pseudospace_Sub$log_TBXT > Tposhigh)
    ]
    T_sim_data <- pseudospace_Sub$NbrR_TBXT_Sim[
      (pseudospace_Sub$Pseudoblock == i) &
        (pseudospace_Sub$Embryo == embryo) &
        (pseudospace_Sub$log_TBXT_Sim > Tposhigh)
    ]
    
    Tbx6_raw_data <- pseudospace_Sub$NbrR_TBX6[
      (pseudospace_Sub$Pseudoblock == i) &
        (pseudospace_Sub$Embryo == embryo) &
        (pseudospace_Sub$log_TBX6 > Tbx6poshigh)
    ]
    Tbx6_sim_data <- pseudospace_Sub$NbrR_TBX6_Sim[
      (pseudospace_Sub$Pseudoblock == i) &
        (pseudospace_Sub$Embryo == embryo) &
        (pseudospace_Sub$log_TBX6_Sim > Tbx6poshigh)
    ]
    
    # Calculate EMD for Sox2 if sufficient data is available
    if (length(Sox2_raw_data) > limit & length(Sox2_sim_data) > limit) {
      hist_Sox2_raw <- calculate_normalized_histogram(filter_outliers(Sox2_raw_data), bins, global_min, global_max)
      hist_Sox2_sim <- calculate_normalized_histogram(filter_outliers(Sox2_sim_data), bins, global_min, global_max)
      EMDdiff_alldata$EMD[
        EMDdiff_alldata$pseudoblocks == i &
          EMDdiff_alldata$embryo == embryo &
          EMDdiff_alldata$CondTF == 'SOX2'
      ] <- emd2d(hist_Sox2_raw, hist_Sox2_sim)
    }
    
    # Calculate EMD for T if sufficient data is available
    if (length(T_raw_data) > limit & length(T_sim_data) > limit) {
      hist_T_raw <- calculate_normalized_histogram(filter_outliers(T_raw_data), bins, global_min, global_max)
      hist_T_sim <- calculate_normalized_histogram(filter_outliers(T_sim_data), bins, global_min, global_max)
      EMDdiff_alldata$EMD[
        EMDdiff_alldata$pseudoblocks == i &
          EMDdiff_alldata$embryo == embryo &
          EMDdiff_alldata$CondTF == 'TBXT'
      ] <- emd2d(hist_T_raw, hist_T_sim)
    }
    
    # Calculate EMD for Tbx6 if sufficient data is available
    if (length(Tbx6_raw_data) > limit & length(Tbx6_sim_data) > limit) {
      hist_Tbx6_raw <- calculate_normalized_histogram(filter_outliers(Tbx6_raw_data), bins, global_min, global_max)
      hist_Tbx6_sim <- calculate_normalized_histogram(filter_outliers(Tbx6_sim_data), bins, global_min, global_max)
      EMDdiff_alldata$EMD[
        EMDdiff_alldata$pseudoblocks == i &
          EMDdiff_alldata$embryo == embryo &
          EMDdiff_alldata$CondTF == 'TBX6'
      ] <- emd2d(hist_Tbx6_raw, hist_Tbx6_sim)
    }
  }
}

# Summarize EMD results
EMDavsSE <- EMDdiff_alldata %>%
  group_by(pseudoblocks, CondTF) %>%
  summarise(
    N = n(),
    meanEMD = mean(EMD, na.rm = TRUE),
    SE.lowEMDS = meanEMD - abs(qnorm((1 - 0.95) / 2)) * (sd(EMD, na.rm = TRUE) / sqrt(N)),
    SE.highEMD = meanEMD + abs(qnorm((1 - 0.95) / 2)) * (sd(EMD, na.rm = TRUE) / sqrt(N))
  )


# Function to summarize high cell data for a given condition
highTF_summarise <- function(data, cond, gate) {
  condraw <- paste('NbrR', cond, sep = '_')
  condsim <- paste(condraw, 'Sim', sep = '_')
  lognormTF <- paste('log', cond, sep = '_')
  lognormTFsim <- paste(lognormTF, 'Sim', sep = '_')
  
  # Determine the quantile gate
  quant_gate <- quantile(data[, lognormTF][data[, lognormTF] > gate], 0.5, na.rm=TRUE)
  data_mask <- !is.na(data$Pseudospace) & data[, lognormTF] > quant_gate
  data_masksim <- !is.na(data$Pseudospace) & data[, lognormTFsim] > quant_gate
  
  # Create a data frame of high cell data
  simvraw <- data.frame(
    TF = c(as.numeric(data[data_mask, condraw]), as.numeric(data[data_masksim, condsim])),
    Cond = c(rep('Raw', sum(data_mask)), rep('Sim', sum(data_masksim))),
    Pseudoblock = c(as.numeric(data[data_mask, 'Pseudoblock']), as.numeric(data[data_masksim, 'Pseudoblock'])),
    Embryo = c(as.numeric(data[data_mask, 'Embryo']), as.numeric(data[data_masksim, 'Embryo']))
  )
  
  # Summarize NbrR by pseudoblocks and conditions with confidence interval calculation
  simvrawbypsudo <- simvraw %>%
    group_by(Pseudoblock, Cond) %>%
    summarise(N = n(),
              meanNbrR = mean(TF),
              SE.low = meanNbrR - abs(qnorm((1 - 0.95) / 2)) * (sd(TF) / sqrt(N)),
              SE.high = meanNbrR + abs(qnorm((1 - 0.95) / 2)) * (sd(TF) / sqrt(N)))
  
  simvrawbypsudoe <- simvraw %>%
    group_by(Pseudoblock, Cond, Embryo) %>%
    summarise(meanNbrR = mean(TF))
  
  # Handle pseudoblocks with insufficient data
  for (i in unique(simvrawbypsudoe$Pseudoblock)) {
    if (nrow(unique(simvrawbypsudoe[simvrawbypsudoe$Pseudoblock == i & simvrawbypsudoe$Cond == 'Raw', 'Embryo'])) < 3 |
        nrow(unique(simvrawbypsudoe[simvrawbypsudoe$Pseudoblock == i & simvrawbypsudoe$Cond == 'Sim', 'Embryo'])) < 3) {
      simvrawbypsudoe$meanNbrR[simvrawbypsudoe$Pseudoblock == i] <- NA
      simvrawbypsudo$meanNbrR[simvrawbypsudo$Pseudoblock == i] <- NA
      simvrawbypsudo$SE.low[simvrawbypsudo$Pseudoblock == i] <- NA
      simvrawbypsudo$SE.high[simvrawbypsudo$Pseudoblock == i] <- NA
    }
  }
  
  simvrawbypsudo$TF <- cond
  return(list(simvrawbypsudo, simvrawbypsudoe))
}

# Calculate high cell data for different conditions
Sox2_high_NbrRdata <- highTF_summarise(pseudospace_Sub, 'SOX2', Sox2pos)
T_high_NbrRdata <- highTF_summarise(pseudospace_Sub, 'TBXT', Tpos)
Tbx6_high_NbrRdata <- highTF_summarise(pseudospace_Sub, 'TBX6', Tbx6pos)

# Create a data frame for storing p-values
NbrRdiffdata <- data.frame(
  pseudoblocks = rep(pseudoblocks, 3),
  pvalue = NaN,
  CondTF = rep(c("SOX2", "TBXT", "TBX6"), each = length(pseudoblocks))
)

# Calculate p-values for each pseudoblock between Raw vs Simulated values
for (i in pseudoblocks) {
  print(i)
  
  Sox2_raw_data <- Sox2_high_NbrRdata[[2]]$meanNbrR[Sox2_high_NbrRdata[[2]]$Pseudoblock == i & Sox2_high_NbrRdata[[2]]$Cond == "Raw"]
  Sox2_sim_data <- Sox2_high_NbrRdata[[2]]$meanNbrR[Sox2_high_NbrRdata[[2]]$Pseudoblock == i & Sox2_high_NbrRdata[[2]]$Cond == "Sim"]
  
  T_raw_data <- T_high_NbrRdata[[2]]$meanNbrR[T_high_NbrRdata[[2]]$Pseudoblock == i & T_high_NbrRdata[[2]]$Cond == "Raw"]
  T_sim_data <- T_high_NbrRdata[[2]]$meanNbrR[T_high_NbrRdata[[2]]$Pseudoblock == i & T_high_NbrRdata[[2]]$Cond == "Sim"]
  
  Tbx6_raw_data <- Tbx6_high_NbrRdata[[2]]$meanNbrR[Tbx6_high_NbrRdata[[2]]$Pseudoblock == i & Tbx6_high_NbrRdata[[2]]$Cond == "Raw"]
  Tbx6_sim_data <- Tbx6_high_NbrRdata[[2]]$meanNbrR[Tbx6_high_NbrRdata[[2]]$Pseudoblock == i & Tbx6_high_NbrRdata[[2]]$Cond == "Sim"]
  
  # Calculate p-values using t-test for each condition
  if (length(Sox2_raw_data)>3 & length(Sox2_sim_data)>3) {
    pvalue_Sox2 <- t.test(Sox2_raw_data, Sox2_sim_data)$p.value
    NbrRdiffdata$pvalue[NbrRdiffdata$pseudoblocks == i & NbrRdiffdata$CondTF == 'SOX2'] <- pvalue_Sox2
  }
  
  if (length(T_raw_data)>3 & length(T_sim_data)>3) {
    pvalue_T <- t.test(T_raw_data, T_sim_data)$p.value
    NbrRdiffdata$pvalue[NbrRdiffdata$pseudoblocks == i & NbrRdiffdata$CondTF == 'TBXT'] <- pvalue_T
  }
  
  if (length(Tbx6_raw_data)>3 & length(Tbx6_sim_data)>3) {
    pvalue_Tbx6 <- t.test(Tbx6_raw_data, Tbx6_sim_data)$p.value
    NbrRdiffdata$pvalue[NbrRdiffdata$pseudoblocks == i & NbrRdiffdata$CondTF == 'TBX6'] <- pvalue_Tbx6
  }
}

#Bind and set condition
allsimvrawbypsudo<-rbind(T_high_NbrRdata[[1]],Tbx6_high_NbrRdata[[1]],Sox2_high_NbrRdata[[1]])
allsimvrawbypsudo$CondTF<-paste(allsimvrawbypsudo$TF,allsimvrawbypsudo$Cond, sep = '_')


#Convert Conditions to vector for ordered legends
allsimvrawbypsudo$CondTF <- factor(as.character(allsimvrawbypsudo$CondTF), levels = c("SOX2_Raw", "SOX2_Sim", 
                                                                                      "TBXT_Raw", "TBXT_Sim", 
                                                                                      "TBX6_Raw", "TBX6_Sim"))
#Define colours per gene and raw vs simulated
Raw_v_Sim_cols = c('#abd9e9ff','#10275eff',
                   '#c0e43eff', '#006837ff',
                   '#de77aeff','#67001fff')

#Plot
(ggplot()+
    geom_vline(xintercept = c(
      2.4,
      4.4),size=1.5)+
    
    geom_vline(xintercept = 0,size=0.05)+
    
    geom_hline(yintercept = 0,size=0.05)+
    geom_ribbon(data=allsimvrawbypsudo, aes(x=Pseudoblock, ymin=SE.low, ymax=SE.high, fill=CondTF), alpha=0.2, col='transparent')+
    geom_line(data=allsimvrawbypsudo, aes(x=Pseudoblock, y=meanNbrR, col=CondTF),size=1)+
    geom_point(data=allsimvrawbypsudo, aes(x=Pseudoblock, y=meanNbrR, col=CondTF),size=3)+
    
    geom_ribbon(data=allsimvrawbypsudo, aes(x=Pseudoblock, ymin=SE.low, ymax=SE.high, fill=CondTF), alpha=0.2, col='transparent')+
    geom_line(data=allsimvrawbypsudo, aes(x=Pseudoblock, y=meanNbrR, col=CondTF),size=1)+
    geom_point(data=allsimvrawbypsudo, aes(x=Pseudoblock, y=meanNbrR, col=CondTF),size=3)+
    
    geom_ribbon(data=allsimvrawbypsudo, aes(x=Pseudoblock, ymin=SE.low, ymax=SE.high, fill=CondTF), alpha=0.2, col='transparent')+
    geom_line(data=allsimvrawbypsudo, aes(x=Pseudoblock, y=meanNbrR, col=CondTF),size=1)+
    geom_point(data=allsimvrawbypsudo, aes(x=Pseudoblock, y=meanNbrR, col=CondTF),size=3)+
    
    
    
    scale_color_manual(values=Raw_v_Sim_cols)+
    scale_fill_manual(values=Raw_v_Sim_cols)+
    
    
    
    theme_minimal()+
    labs(x='Psuedospace',y ='Neighbour Ratio (NR)', col='', fill='')+
    theme(legend.key.size = unit(1, 'cm'),
          text = element_text(size=25),
          axis.text.x = element_blank(),
          legend.text = element_text(size=26),
          # legend.position = 'none',
          axis.line.x =  element_blank())+
    
    coord_cartesian(xlim=c(0, maxpseudo),
                    ylim=c(-0.2,1.2)))/

(ggplot()+
    geom_vline(xintercept = 0,size=0.05)+
    geom_hline(yintercept = 0,size=0.05)+
    
    theme_minimal()+
    labs(x='Psuedospace',y ='Earth movers distance', col='', fill='')+
    theme(legend.key.size = unit(1, 'cm'),
          text = element_text(size=25),
          axis.text.x = element_blank(),
          legend.text = element_text(size=26),
          # legend.position = 'none',
          axis.line.x =  element_blank())+   geom_ribbon(data = EMDavsSE, 
                                                         aes(x = pseudoblocks, ymax =  SE.highEMD, ymin = SE.lowEMDS, fill = CondTF ),
                                                         col = 'transparent', alpha = 0.2)+
    
    geom_line(data = EMDavsSE, aes(x = pseudoblocks, y = meanEMD, col = CondTF ), size = 2)+
    geom_point(data = EMDavsSE, aes(x = pseudoblocks, y = meanEMD, col = CondTF ), size = 3)+
    
    geom_vline(xintercept = c(2.4,
                              4.4),size=1.5)+
    scale_color_manual(values=c('#235ee8','#2cd443','#c926b9')) +
    scale_fill_manual(values=c('#235ee8','#2cd443','#c926b9')) +
    
    xlim(0, maxpseudo))/

  
  (  ggplot()+
       geom_vline(xintercept = 0,size=0.05)+
       geom_hline(yintercept = 0,size=0.05)+
       geom_hline(yintercept =-log(c(0.05)) , linetype = 'dashed')+
       
       geom_vline(xintercept = c(
         2.4,
         4.4),size=1.5)+
       geom_point(data = NbrRdiffdata, aes(x=pseudoblocks, y =-log(pvalue), col = CondTF  ), size = 3)+
       
       geom_line(data = NbrRdiffdata, aes(x=pseudoblocks, y =-log(pvalue), col = CondTF  ), size = 1.5)+
       scale_color_manual(values=c('#235ee8','#c926b9','#2cd443')) +
       labs(x='Psuedospace',y ='-log(p-value)', col='', fill='')+
       theme_minimal()+
       theme(legend.key.size = unit(1, 'cm'),
             text = element_text(size=25),
             axis.text.x = element_blank(),
             legend.text = element_text(size=26),
             # legend.position = 'none',
             axis.line.x =  element_blank()))


