library(ggplot2)
library(patchwork)
library(reshape2)
library(dplyr)
library(RColorBrewer)


# Import data
path_2_data = getwd()
OS_Sep = '/'
Wt_E8_data <- as.data.frame(read_csv(paste(path_2_data,"Wt_E8_data.csv", sep =OS_Sep)))
hNMP_data <- as.data.frame(read_csv(paste(path_2_data,"hNMP_data.csv", sep =OS_Sep)))
Gloid_data <- as.data.frame(read_csv(paste(path_2_data,"Gastruloid_data.csv", sep =OS_Sep)))
#Define density colour scheme
brewer_spectral <- rev(brewer.pal(11, "Spectral"))

#Define function to get local density of points in PCA space for FACS style plot
get_density <- function(x, y, ...) {
  density <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, density$x)
  iy <- findInterval(y, density$y)
  ii <- cbind(ix, iy)
  return(density$z[ii])
}


# Figure 6 C --------------------------------------------------------------


Wt_E8_data$log_SOX2 <- as.numeric(Wt_E8_data$log_SOX2)
Wt_E8_data$log_TBX6 <- as.numeric(Wt_E8_data$log_TBX6)


Wt_E8_data$logTCNbrdensity <- get_density(Wt_E8_data$log_TBXT, Wt_E8_data$log_Nbr_Mean_TBXT, n = 300)
Wt_E8_data$logTbx6CNbrdensity <- get_density(Wt_E8_data$log_TBX6, Wt_E8_data$log_Nbr_Mean_TBX6, n = 300)
Wt_E8_data$logSox2CNbrdensity <- get_density(Wt_E8_data$log_SOX2, Wt_E8_data$log_Nbr_Mean_SOX2, n = 300)


WtE8_TFvsNBrplot<-
  
  ggplot()+geom_point(data=Wt_E8_data[order(Wt_E8_data$logSox2CNbrdensity), ],
                      aes(x=log_SOX2, y=log_Nbr_Mean_SOX2, col=logSox2CNbrdensity))+
  geom_line(data=Wt_E8_data,
            aes(x=log_SOX2,y=log_SOX2),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(Wt_E8_data$log_SOX2))+
  geom_hline(yintercept=median(Wt_E8_data$log_Nbr_Mean_SOX2))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())+
  
  ggplot()+geom_point(data=Wt_E8_data[order(Wt_E8_data$logTCNbrdensity), ],aes(x=log_TBXT,y=log_Nbr_Mean_TBXT, col=logTCNbrdensity))+
  geom_line(data=Wt_E8_data,aes(x=log_TBXT,y=log_TBXT),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(Wt_E8_data$log_TBXT))+
  geom_hline(yintercept=median(Wt_E8_data$log_Nbr_Mean_TBXT))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())+
  
  ggplot()+geom_point(data=Wt_E8_data[order(Wt_E8_data$logTbx6CNbrdensity), ],aes(x=log_TBX6,y=log_Nbr_Mean_TBX6, col=logTbx6CNbrdensity))+
  geom_line(data=Wt_E8_data,aes(x=log_TBX6,y=log_TBX6),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(Wt_E8_data$log_TBX6))+
  geom_hline(yintercept=median(Wt_E8_data$log_Nbr_Mean_TBX6))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())
WtE8_TFvsNBrplot


Gloid_data <- Gloid_data %>%
  filter(
    is.finite(log_SOX2),
    is.finite(log_Nbr_Mean_SOX2)
  )

Gloid_data$logTCNbrdensity <- get_density(Gloid_data$log_TBXT, Gloid_data$log_Nbr_Mean_TBXT, n = 300)
Gloid_data$logTbx6CNbrdensity <- get_density(Gloid_data$log_TBX6, Gloid_data$log_Nbr_Mean_TBX6, n = 300)
Gloid_data$logSox2CNbrdensity <- get_density(Gloid_data$log_SOX2, Gloid_data$log_Nbr_Mean_SOX2, n = 300)


Gloid_TFvsNBrplot<-
  
  ggplot()+geom_point(data=Gloid_data[order(Gloid_data$logSox2CNbrdensity), ],
                      aes(x=log_SOX2, y=log_Nbr_Mean_SOX2, col=logSox2CNbrdensity))+
  geom_line(data=Gloid_data,
            aes(x=log_SOX2,y=log_SOX2),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(Gloid_data$log_SOX2))+
  geom_hline(yintercept=median(Gloid_data$log_Nbr_Mean_SOX2))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())+
  ylim(6.5,10.5)+
  xlim(6.5,10.5)+
  
  ggplot()+geom_point(data=Gloid_data[order(Gloid_data$logTCNbrdensity), ],aes(x=log_TBXT,y=log_Nbr_Mean_TBXT, col=logTCNbrdensity))+
  geom_line(data=Gloid_data,aes(x=log_TBXT,y=log_TBXT),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(Gloid_data$log_TBXT))+
  geom_hline(yintercept=median(Gloid_data$log_Nbr_Mean_TBXT))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())+
  
  ggplot()+geom_point(data=Gloid_data[order(Gloid_data$logTbx6CNbrdensity), ],aes(x=log_TBX6,y=log_Nbr_Mean_TBX6, col=logTbx6CNbrdensity))+
  geom_line(data=Gloid_data,aes(x=log_TBX6,y=log_TBX6),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(Gloid_data$log_TBX6))+
  geom_hline(yintercept=median(Gloid_data$log_Nbr_Mean_TBX6))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())
Gloid_TFvsNBrplot


#Remove missing or inf values
hNMP_data <- hNMP_data[!is.na(hNMP_data$log_Nbr_Mean_TBXT),]

#Include only CHIR 2 or 3 micro-molar
hNMP_data_23CHIR = hNMP_data[hNMP_data$CHIR==3 | hNMP_data$CHIR==2,]

unique(hNMP_data_23CHIR$CHIR)

hNMP_data_23CHIR$logTCNbrdensity <- get_density(hNMP_data_23CHIR$log_TBXT, hNMP_data_23CHIR$log_Nbr_Mean_TBXT, n = 300)
hNMP_data_23CHIR$logTbx6CNbrdensity <- get_density(hNMP_data_23CHIR$log_TBX6, hNMP_data_23CHIR$log_Nbr_Mean_TBX6, n = 300)
hNMP_data_23CHIR$logSox2CNbrdensity <- get_density(hNMP_data_23CHIR$log_SOX2, hNMP_data_23CHIR$log_Nbr_Mean_SOX2, n = 300)


hNMP_TFvsNBrplot<-
  
  ggplot()+geom_point(data=hNMP_data_23CHIR[order(hNMP_data_23CHIR$logSox2CNbrdensity), ],aes(x=log_SOX2,y=log_Nbr_Mean_SOX2, col=logSox2CNbrdensity))+
  geom_line(data=hNMP_data_23CHIR,aes(x=log_SOX2,y=log_SOX2),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(hNMP_data_23CHIR$log_SOX2))+
  geom_hline(yintercept=median(hNMP_data_23CHIR$log_Nbr_Mean_SOX2))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())+
  
  ggplot()+geom_point(data=hNMP_data_23CHIR[order(hNMP_data_23CHIR$logTCNbrdensity), ],aes(x=log_TBXT,y=log_Nbr_Mean_TBXT, col=logTCNbrdensity))+
  geom_line(data=hNMP_data_23CHIR,aes(x=log_TBXT,y=log_TBXT),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(hNMP_data_23CHIR$log_TBXT))+
  geom_hline(yintercept=median(hNMP_data_23CHIR$log_Nbr_Mean_TBXT))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())+
  
  ggplot()+geom_point(data=hNMP_data_23CHIR[order(hNMP_data_23CHIR$logTbx6CNbrdensity), ],aes(x=log_TBX6,y=log_Nbr_Mean_TBX6, col=logTbx6CNbrdensity))+
  geom_line(data=hNMP_data_23CHIR,aes(x=log_TBX6,y=log_TBX6),size=1,linetype="dashed")+
  scale_color_gradientn(colours = brewer_spectral)+
  geom_vline(xintercept=median(hNMP_data_23CHIR$log_TBX6))+
  geom_hline(yintercept=median(hNMP_data_23CHIR$log_Nbr_Mean_TBX6))+
  theme_bw()+
  theme(legend.position='none',
        axis.text = element_blank(),
        axis.title = element_blank())
hNMP_TFvsNBrplot

#Combine into one
WtE8_TFvsNBrplot / Gloid_TFvsNBrplot / hNMP_TFvsNBrplot
