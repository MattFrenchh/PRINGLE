library(ggplot2)
library(scico)
library(patchwork)
library(factoextra)
library(RColorBrewer)
library(readxl)

# Import data
Wt_E8_data <- as.data.frame(read_excel("S1_Data.xlsx", sheet = "Wt_E8.5_data", col_names = TRUE))

# Isolate example embyro of four somite pairs where fate map is known.
SP4<-Wt_E8_data[Wt_E8_data$SP==4,]
SP41<-SP4[SP4$SP_Rep==1,] # Take first replicate
notnowidth = 50/2 # Region boxes defined as width of Notochord - approximately 50 microns
PSend<-max(SP41$Scaled_Rel_AP_Position) * 0.9 # Adjust box length to fit fate map region definitions
notostart<-(0) # Notochord origin at 0 microns
ROIdepth<-(PSend-notostart)/5 # 5 regions along PS length

# Figure 3a -------------------------------------------------------------------------

# Create plot
plot <- ggplot() +
  
  # Points: Sox2+ and T+
  geom_point(
    data = SP41[SP41$Sox2pos == 'Sox2+' & SP41$Tpos == 'T+', ],
    aes(Dist_to_Midline, Scaled_Rel_AP_Position, col = TFgate),
    size = 1,
    col = "#bf812d"
  ) +
  
  # Points: Sox2- or T-
  geom_point(
    data = SP41[SP41$Sox2pos == 'Sox2-' | SP41$Tpos == 'T-', ],
    aes(Dist_to_Midline, Scaled_Rel_AP_Position, col = TFgate),
    size = 1,
    col = "#80cdc1"
  ) +
  
  labs(x = "Distance to midline (μm)", y = "Distance to Noto. Origin  (μm)") +
  theme_dark() +
  theme(
    text = element_text(size = 25)
  ) +
  
  # Node-streak border region
  geom_rect(
    aes(xmin = -notnowidth, xmax = notnowidth, ymin = notostart, ymax = notostart - ROIdepth),
    alpha = 0,
    color = "black",
    size = 1.5
  ) +
  
  # Left/Right lateral regions
  geom_rect(
    aes(xmin =  notnowidth, xmax =  notnowidth * 3, ymin = notostart, ymax = notostart + ROIdepth),
    alpha = 0,
    color = "black",
    size = 1.5
  ) +
  geom_rect(
    aes(xmin =  notnowidth, xmax =  notnowidth * 3, ymin = notostart, ymax = notostart + ROIdepth * 2),
    alpha = 0,
    color = "black",
    size = 1.5
  ) +
  geom_rect(
    aes(xmin = -notnowidth * 3, xmax = -notnowidth, ymin = notostart, ymax = notostart + ROIdepth),
    alpha = 0,
    color = "black",
    size = 1.5
  ) +
  geom_rect(
    aes(xmin = -notnowidth * 3, xmax = -notnowidth, ymin = notostart, ymax = notostart + ROIdepth * 2),
    alpha = 0,
    color = "black",
    size = 1.5
  ) +
  
  coord_fixed()

# Display plot
plot

# Figure 3B ---------------------------------------------------------------

# Normalise signal values to 0-1 and convert into RGB hex codes
SP41$RawRGB<-rgb((SP41$TBXT-min(SP41$TBXT))/(max(SP41$TBXT)-min(SP41$TBXT)),
                 (SP41$SOX2-min(SP41$SOX2))/(max(SP41$SOX2)-min(SP41$SOX2)),
                 (SP41$TBX6-min(SP41$TBX6))/(max(SP41$TBX6)-min(SP41$TBX6)),
                 maxColorValue = 1)
# Normalise smoothed signal values to 0-1 and convert into RGB hex codes
SP41$SmoothRGB<-rgb((SP41$Smooth_TBXT-min(SP41$Smooth_TBXT))/(max(SP41$Smooth_TBXT)-min(SP41$Smooth_TBXT)),
                    (SP41$Smooth_SOX2-min(SP41$Smooth_SOX2))/(max(SP41$Smooth_SOX2)-min(SP41$Smooth_SOX2)),
                    (SP41$Smooth_TBX6-min(SP41$Smooth_TBX6))/(max(SP41$Smooth_TBX6)-min(SP41$Smooth_TBX6)),
                    maxColorValue = 1)


#Plot raw vs smoothed side by side
SP41$X_left  <- sqrt(SP41$Dist_to_Midline^2) * -1
SP41$X_right <- sqrt(SP41$Dist_to_Midline^2)

# Plot
ggplot() +
  
  # Left (raw RGB)
  geom_point(
    data = SP41,
    aes(x = X_left, y = Scaled_Rel_AP_Position),
    col = SP41$RawRGB,
    size = 0.5
  ) +
  
  # Right (smoothed RGB)
  geom_point(
    data = SP41,
    aes(x = X_right, y = Scaled_Rel_AP_Position),
    col = SP41$SmoothRGB,
    size = 0.5
  ) +
  
  # Midline reference to split the sides further - aesthetic choice
  geom_vline(xintercept = 0, col = "white", size = 2) +
  
  theme_classic() +
  theme(
    text = element_text(size = 18),
    axis.title = element_text(size = 20),
    panel.background = element_rect(color = "black", fill = "black")
  ) +
  labs(
    x = "Distance to midline (μm)",
    y = "Distance to Noto. Origin (μm)"
  )
  
# Figure 3Ci ---------------------------------------------------------------

#Perform PCA on the locally smoothed values (natural log transformation)
logTF.pca<-prcomp((Wt_E8_data[,c("log_Smooth_TBX6","log_Smooth_TBXT","log_Smooth_SOX2")]),center = TRUE,scale. = TRUE)

# Get standard deviation of each PC
sdev <- logTF.pca$sdev

# Compute variance
var <- sdev^2

# Compute proportion of variance explained
pve <- var / sum(var)

# Extract first two PCA coordinates
PCs<-as.data.frame(logTF.pca[["x"]])

Wt_E8_data$PC1<-PCs$PC1
Wt_E8_data$PC2<-PCs$PC2

# Extract PC loadings to plot directionality of TF contribution
PCAloadings <- data.frame(Variables = c("TBX6","TBXT","SOX2"), logTF.pca$rotation)

#Define function to get local density of points in PCA space for FACS style plot
get_density <- function(x, y, ...) {
  density <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, density$x)
  iy <- findInterval(y, density$y)
  ii <- cbind(ix, iy)
  return(density$z[ii])
}

Wt_E8_data$density <- get_density(Wt_E8_data$PC1, Wt_E8_data$PC2, n = 300)


#Define density colour scheme
brewer_spectral <- rev(brewer.pal(11, "Spectral"))
# PCA plot with density & loadings
ggplot() +  
  geom_point(
    data = Wt_E8_data[order(Wt_E8_data$density), ],
    aes(x = PC1, y = PC2 * -1, col = density / max(density, na.rm = TRUE)),
    size = 0.5
  ) +
  scale_color_gradientn(colours = brewer_spectral) +
  
  # Arrows: loadings
  geom_segment(
    data = PCAloadings,
    aes(x = 0, y = 0, xend = PC1 * 2, yend = PC2 * -2),
    arrow = arrow(length = unit(1, "picas")),
    color = "black",
    lineend = "round",
    linejoin = "round",
    size = 1.5
  ) +
  
  # Labels for loadings
  annotate(
    "text",
    x = PCAloadings$PC1 * 2.5,
    y = PCAloadings$PC2 * -2.5,
    label = PCAloadings$Variables,
    size = 5,
    alpha = 0.8
  ) +
  
  theme_void() +
  labs(
    col = "density",
    x = paste0("PC1 (", round(pve[1] * 100, 1), "%)"),
    y = paste0("PC2 (", round(pve[2] * 100, 1), "%)")
  ) +
  scale_size(range = c(0.1, 0.5)) +
  theme(
    axis.title = element_text(size = 20)
  )


# Figure 3Ci ---------------------------------------------------------------

ggplot()+  
  
  geom_point(
    data=Wt_E8_data,
    aes(x=PC1, 
        y=PC2*-1,
        col=Pseudospace/max(Pseudospace, na.rm = TRUE)),
    size=0.5)+
  
  scale_color_viridis_c(option= 'inferno')+
  
  theme_void() +
  
  labs(
    col="Pseudo-\nspace",
       x=paste0("PC1 (", as.character(round(pve[1], digits = 3)*100),"%)"), 
       y=paste0("PC2 (", as.character(round(pve[2], digits = 3)*100),"%)")
    )+
  
  scale_size(range = c(0.1,0.5))+
  
  theme(axis.title = element_text(size=20))
  

# Figure 3D ---------------------------------------------------------------

ggplot() +
  
  # Main data points colored by Pseudospace
  geom_point(
    data = SP41,
    aes(x = Dist_to_Midline, y = Scaled_Rel_AP_Position, color = Pseudospace),
    size = 1.2
  ) +
  
  # Color scale
  scale_color_viridis_c(option = 'inferno', na.value = "grey80") +
  
  # Theme settings
  theme_dark() +
  theme(
    text = element_text(size = 18),
    axis.title = element_text(size = 20)
  ) +
  
  # Central PS region
  geom_rect(
    aes(
      xmin = -notnowidth, xmax = notnowidth,
      ymin = notostart, ymax = notostart - ROIdepth
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  
  # Mid-lateral edge dividers
  geom_rect(aes(
    xmin =  notnowidth * 2, xmax =  notnowidth * 3,
    ymin =  notostart,      ymax =  notostart
  ), alpha = 0, color = "black", size = 1.5) +
  
  geom_rect(aes(
    xmin = -notnowidth * 2, xmax = -notnowidth * 3,
    ymin =  notostart,      ymax =  notostart
  ), alpha = 0, color = "black", size = 1.5) +
  
  # First layer (left and right)
  geom_rect(aes(
    xmin =  notnowidth, xmax =  notnowidth * 3,
    ymin =  notostart,  ymax =  notostart + ROIdepth
  ), alpha = 0, color = "black", size = 1.5) +
  
  geom_rect(aes(
    xmin = -notnowidth, xmax = -notnowidth * 3,
    ymin =  notostart,  ymax =  notostart + ROIdepth
  ), alpha = 0, color = "black", size = 1.5) +
  
  # Second layer (left and right)
  geom_rect(aes(
    xmin =  notnowidth, xmax =  notnowidth * 3,
    ymin =  notostart,  ymax =  notostart + ROIdepth * 2
  ), alpha = 0, color = "black", size = 1.5) +
  
  geom_rect(aes(
    xmin = -notnowidth, xmax = -notnowidth * 3,
    ymin =  notostart,  ymax =  notostart + ROIdepth * 2
  ), alpha = 0, color = "black", size = 1.5) +
  
  # Fixed aspect ratio
  coord_fixed(ratio = 1) +
  
  # Axis and legend labels
  labs(
    x = "Distance to midline (μm)",
    y = "Distance to Noto. Origin (μm)",
    color = "Pseudo-\nspace"
  )

# Figure 3E ---------------------------------------------------------------

lin14SP1 <- SP41[!is.na(SP41$Pseudospace),]
BF_ROI <- lin14SP1[lin14SP1$Pseudospace>Pseudospace_lims[1]&
                     lin14SP1$Pseudospace<Pseudospace_lims[2],]

#Define limits of Pseudospace to fit bi-fated ROIs at 4 somite pairs
Pseudospace_lims = c(2.4, 4.8)

ggplot() +
  
  # All cells to form 'underlayer' to layer subsets over
  geom_point(
    data = SP41,
    aes(x = Dist_to_Midline, y = Scaled_Rel_AP_Position),
    size = 1.2,
    col = "grey80"
  ) +
  
  # TF state 'pseudospace' trajectory cells layer
  geom_point(
    data = lin14SP1,
    aes(x = Dist_to_Midline, y = Scaled_Rel_AP_Position),
    size = 1.2,
    col = "#2166ac"
  ) +
  
  # Bi-fated region isolated points  layer
  geom_point(
    data = lin14SP1[
      lin14SP1$Pseudospace > Pseudospace_lims[1] &
        lin14SP1$Pseudospace < Pseudospace_lims[2], ],
    aes(x = Dist_to_Midline, y = Scaled_Rel_AP_Position),
    size = 1.2,
    col = "#b2182b"
  ) +
  
  # Color scale (unused here, kept in case of future use)
  scale_color_viridis_c(option = 'inferno') +
  
  # Minimal theme
  theme_minimal() +
  theme(
    text = element_text(size = 18),
    axis.title = element_text(size = 20)
  ) +
  
  # Axis labels
  labs(
    x = "Distance to midline (μm)",
    y = "Distance to Noto. Origin (μm)",
    col = ""
  ) +
  
  # Primitive streak (central) region
  geom_rect(
    aes(
      xmin = -notnowidth, xmax = notnowidth,
      ymin = notostart, ymax = notostart - ROIdepth
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  
  # Lateral limits (horizontal dividers)
  geom_rect(
    aes(
      xmin =  notnowidth * 2, xmax =  notnowidth * 3,
      ymin =  notostart,      ymax =  notostart
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  geom_rect(
    aes(
      xmin = -notnowidth * 2, xmax = -notnowidth * 3,
      ymin =  notostart,      ymax =  notostart
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  
  # Lateral region layers (1x ROI depth)
  geom_rect(
    aes(
      xmin =  notnowidth, xmax =  notnowidth * 3,
      ymin =  notostart,  ymax =  notostart + ROIdepth
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  geom_rect(
    aes(
      xmin = -notnowidth, xmax = -notnowidth * 3,
      ymin =  notostart,  ymax =  notostart + ROIdepth
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  
  # Lateral region layers (2x ROI depth)
  geom_rect(
    aes(
      xmin =  notnowidth, xmax =  notnowidth * 3,
      ymin =  notostart,  ymax =  notostart + ROIdepth * 2
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  geom_rect(
    aes(
      xmin = -notnowidth, xmax = -notnowidth * 3,
      ymin =  notostart,  ymax =  notostart + ROIdepth * 2
    ),
    alpha = 0, color = "black", size = 1.5
  ) +
  
  # Maintain aspect ratio
  coord_fixed(ratio = 1)






