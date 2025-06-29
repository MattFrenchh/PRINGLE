library(ggplot2)
library(scico)
library(patchwork)
library(readxl)

# Load the data 
Wt_E8_data <- as.data.frame(read_excel("Data_1.xlsx", sheet = "Wt_E8.5_data", col_names = TRUE))

# Plot settings
point_size <- 0.8
axis_title_size <- 22
axis_tick_text_size <- 20
plot_title_size <- 28
vjust_pos = -6

# Base theme elements
base_theme <- theme_dark() +
  theme(
    panel.background = element_rect(fill = "grey10"),
    legend.position = "none",
    axis.title = element_text(size = axis_title_size),
    axis.text = element_text(size = axis_tick_text_size),
    plot.title = element_text(size = plot_title_size, color = "white", vjust = vjust_pos)
  )

# Plot 1: SOX2
plot1 <- ggplot(data = Wt_E8_data[Wt_E8_data$Embryo == 1, ]) +
  geom_point(aes(Scaled_Rel_LR_Position, Scaled_Rel_AP_Position, col = logNormSox2, size = NewZ), size = point_size) +
  scale_color_scico(palette = "batlow") +
  labs(
    x = "",
    y = "Normalised anterior-\nposterior position (μm)",
    title = "SOX2"
  ) +
  coord_fixed() +
  base_theme

# Plot 2: TBXT
plot2 <- ggplot(data = Wt_E8_data[Wt_E8_data$Embryo == 1, ]) +
  geom_point(aes(Scaled_Rel_LR_Position, Scaled_Rel_AP_Position, col = logNormT), size = point_size) +
  scale_color_scico(palette = "batlow") +
  labs(
    x = "Normalised medial-lateral position (μm)",
    y = "",
    title = "TBXT"
  ) +
  coord_fixed() +
  base_theme

# Plot 3: TBX6
plot3 <- ggplot(data = Wt_E8_data[Wt_E8_data$Embryo == 1, ]) +
  geom_point(aes(Scaled_Rel_LR_Position, Scaled_Rel_AP_Position, col = logNormTbx6), size = point_size) +
  scale_color_scico(palette = "batlow") +
  labs(
    x = "",
    y = "",
    title = "TBX6"
  ) +
  coord_fixed() +
  base_theme

# Output one of the plots
combined_plot <- plot1 + plot2 + plot3 + plot_layout(ncol = 3)
combined_plot

#Optional - Save plot
ggsave('Figure 3c.pdf', combined_plot, units='in', width = 12, height = 4.5, dpi = 300)
