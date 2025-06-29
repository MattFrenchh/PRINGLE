library(ggplot2)
library(scico)
library(patchwork)
library(readxl)
library(reshape2)
library(readxl)

# Import data
Wt_E8_data <- as.data.frame(read_excel("Data_1.xlsx", sheet = "Wt_E8.5_data", col_names = TRUE))
NMPoutlines <- as.data.frame(read_excel("Data_1.xlsx", sheet = "WtE8_NMP_ROI_outlines", col_names = TRUE))


# Define functions --------------------------------------------------------

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

# Figure 4 Bi -------------------------------------------------------------

# Define parameters
yparamsnorm <- c(12, -100, 450)
xparamsnorm <- c(12, -200, 200)
TF_names <- c("SOX2", "TBXT", "TBX6")
TFs <- c("SOX2", "TBXT", "TBX6")
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
  combined$value <- normalize(combined$value)
  combined$TF <- tf_name
  
  binned_list[[i]] <- combined
}

# Combine all into one data frame
all <- do.call(rbind, binned_list)

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
    alpha = 0.4,
    size = 0.2,
    linejoin = 'round'
  ) +
  
  # Center point marker
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
  scale_fill_scico_d(palette = 'batlow') +
  scale_color_scico(palette = 'batlow') +
  
  # Faceting by TF and SP
  facet_grid(
    factor(TF, levels = c('SOX2', 'TBXT', 'TBX6')) ~ as.numeric(SP)
  )+
  
  coord_fixed()


# Figure 4 Bii ------------------------------------------------------------

#Define output as 3D neighbour gradient
TFs <- c("NbrGrd_SOX2", "NbrGrd_TBXT", "NbrGrd_TBX6")

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
  combined$value <- normalize(combined$value)
  combined$TF <- tf_name
  
  binned_list[[i]] <- combined
}

# Combine all into one data frame
all <- do.call(rbind, binned_list)

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
    alpha = 0.4,
    size = 0.2,
    linejoin = 'round'
  ) +
  
  # Center point marker
  geom_point(
    data = data.frame(x = 0, y = 0),
    aes(x, y),
    size = 2,
    col = 'black',
    pch = 21,
    fill = 'transparent'
  ) +
  # 
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
  scale_fill_scico_d(palette = 'lajolla') +
  scale_color_scico(palette = 'lajolla') +
  
  # Faceting by TF and SP
  facet_grid(
    factor(TF, levels = c('SOX2', 'TBXT', 'TBX6')) ~ as.numeric(SP)
  )+
  
  coord_fixed()
  
  


# Figure C ----------------------------------------------------------------

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
plot_sox2_NFI <- ggplot(Wt_E8_data, aes(y = SOX2 / quantile(SOX2, 0.95))) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  base_theme + base_labs_NFI + vline_layer

plot_tbxt_NFI <- ggplot(Wt_E8_data, aes(y = TBXT / quantile(TBXT, 0.95))) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  base_theme + base_labs_NFI + vline_layer

plot_tbx6_NFI <- ggplot(Wt_E8_data, aes(y = TBX6 / quantile(TBX6, 0.95))) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  base_theme + base_labs_NFI + vline_layer

# Individual plots of Neigbhour Gradient

plot_sox2_NbrGrd <- ggplot(Wt_E8_data, aes(y = NbrGrd_SOX2 / quantile(NbrGrd_SOX2, 0.95))) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  base_theme + base_labs_NbrGrad + vline_layer

plot_tbxt_NbrGrd <- ggplot(Wt_E8_data, aes(y = NbrGrd_TBXT / quantile(NbrGrd_TBXT, 0.95))) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  base_theme + base_labs_NbrGrad + vline_layer

plot_tbx6_NbrGrd <- ggplot(Wt_E8_data, aes(y = NbrGrd_TBX6 / quantile(NbrGrd_TBX6, 0.99))) +
  geom_smooth(base_aes, size = 3, method = 'loess') +
  scale_colour_manual(values = SP_colours) +
  scale_fill_manual(values = SP_colours) +
  base_theme + base_labs_NbrGrad + vline_layer

# Combine to one plot
plot_sox2_NFI / plot_tbxt_NFI / plot_tbx6_NFI / plot_sox2_NbrGrd / plot_tbxt_NbrGrd / plot_tbx6_NbrGrd

