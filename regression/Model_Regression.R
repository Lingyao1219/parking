pacman::p_load(brms, data.table, car, mgcv, dplyr, mgcViz, spdep, sf, ggplot2, plyr, pls, psych, mdatools, broom, pals, metR)

# Read data
park_df <- read.csv('D://Google_Review//Parking//temp//bg_poitype_parking.csv')
park_df$CBSA <- as.factor(park_df$CBSA)
park_df[park_df$Categories %in%
                c("Transportation", "Public Service", "Parking", "Health Care", "Education", 'Others'), 'Categories'] <- 'AAOthers'
# park_df <- park_df[park_df$Categories!='Others', ]
park_df$Categories <- as.factor(park_df$Categories)
# park_df$Parking_review_pct <- park_df$total_parking_reviews/park_df$num_of_reviews
# park_df$Parking_poi_pct <- park_df$Parking_poi_density/park_df$Total_poi_density
park_df$Parking_review_density <- park_df$num_of_parking_review/park_df$ALAND_x
# summary(park_df)

# Scale the data
ind <- sapply(park_df, is.numeric)
ind[(names(ind) %in% "weight_st")] <- FALSE
park_df[ind] <- scale(park_df[ind], center = TRUE, scale = TRUE)
park_df[is.na(park_df)] <- 0
ggplot(park_df, aes(x=weight_st)) + geom_histogram(binwidth=.1, colour="black", fill="white")

# VIF, Linear Regression Assumptions and Diagnostics
# colnames(park_df)
linear_test <- lm(weight_st ~ Asian_R + Black_Non_Hispanic_R + HISPANIC_LATINO_R + Indian_R
          + Population_Density + Employment_Density + Rural_Population_R
          + Education_Degree_R + Household_Below_Poverty_R + Walkability
          + Transit_Freq_Area + Zero_car_R + One_car_R + Parking_poi_density
          + Democrat_R + Male_R + Over_65_R + Bt_45_64_R
          + Categories, data = park_df)
vif(linear_test)
summary(linear_test, robust = TRUE)

# Drop outliers
model.diag.metrics <- augment(linear_test)
park_df$.cooksd  <- model.diag.metrics$.cooksd
influential <- as.numeric(row.names(park_df)[(park_df$.cooksd > (100 / (nrow(park_df)-20)))]) # default: 2
park_df1 <- park_df[!(row.names(park_df) %in% influential),]
for (var in c('Asian_R', 'Black_Non_Hispanic_R', 'HISPANIC_LATINO_R', 'Population_Density', 'Employment_Density',
              'Household_Below_Poverty_R', 'Transit_Freq_Area', 'Zero_car_R', 'One_car_R', 'Parking_poi_density',
              'Male_R', 'Over_65_R', 'Bt_45_64_R', 'avg_poi_score')){
  nrow_raw <- nrow(park_df1)
  park_df1 <- park_df1[park_df1[,var]<quantile(park_df1[,var],0.999), ]
  print(nrow(park_df1) - nrow_raw)}

# Regression: Linear
Cross_section_model <- function (yvar, park_df1){
  if (yvar=='all'){
    GAM_RES1 <- mgcv::bam(weight_st ~ Asian_R + Black_Non_Hispanic_R + HISPANIC_LATINO_R + Indian_R
            + Population_Density + Employment_Density + Rural_Population_R
            + Household_Below_Poverty_R
            + Walkability + Transit_Freq_Area + Zero_car_R + One_car_R + Parking_poi_density
            + Democrat_R + Male_R + Over_65_R + Bt_45_64_R + avg_poi_score
            + Categories + ti(Lat, Lng, bs = 'gp') + s(CBSA, bs = 're'), data = park_df1,
            control = gam.control(trace = TRUE), method = "fREML", discrete = TRUE)}
  else {
    GAM_RES1 <- mgcv::bam(weight_st ~ Asian_R + Black_Non_Hispanic_R + HISPANIC_LATINO_R + Indian_R
            + Population_Density + Employment_Density + Rural_Population_R
            + Household_Below_Poverty_R
            + Walkability + Transit_Freq_Area + Zero_car_R + One_car_R + Parking_poi_density
            + Democrat_R + Male_R + Over_65_R + Bt_45_64_R + avg_poi_score
            + ti(Lat, Lng, bs = 'gp') + s(CBSA, bs = 're'), data = park_df1[park_df1$Categories==yvar,],
            control = gam.control(trace = TRUE), method = "fREML", discrete = TRUE)}

  print(summary(GAM_RES1,robust = TRUE))

  new.summary <- summary(GAM_RES1, robust = TRUE)
  Gamm_stable <- as.data.frame(new.summary$s.table)
  Gam_summary_coeff <- as.data.frame(new.summary$p.table)
  names(Gamm_stable) <- names(Gam_summary_coeff)
  Gam_summary_coeff <- rbind(Gam_summary_coeff, Gamm_stable)
  Gam_summary_coeff <- data.frame(names = row.names(Gam_summary_coeff), Gam_summary_coeff)
  Gam_summary_coeff$Yvar <- yvar
  Gam_summary_coeff[nrow(Gam_summary_coeff) + 1,] <- c("Adj_R2",new.summary$r.sq,'','','',yvar)
  Gam_summary_coeff[nrow(Gam_summary_coeff) + 1,] <- c("dev.expl",new.summary$dev.expl,'','','',yvar)
  Gam_summary_coeff[nrow(Gam_summary_coeff) + 1,] <- c("n",new.summary$n,'','','',yvar)
  return(Gam_summary_coeff)
}

for (var in c('all','Restaurant', 'Retail Trade', 'Recreation', 'Hotel', 'Personal Service', 'Apartment')){
  Gam_Agg_all <- Cross_section_model(var, park_df1)
  fwrite(Gam_Agg_all, paste('D:\\Google_Review\\Parking\\results\\', var, '_Gam_Pop_Pct_scale.csv'))}

# Nonlinear: Interaction by category
GAM_RES1 <- mgcv::bam(weight_st ~ Asian_R + Black_Non_Hispanic_R + HISPANIC_LATINO_R + Indian_R
            + Employment_Density + Rural_Population_R
            + Walkability + Transit_Freq_Area + Zero_car_R + One_car_R + Parking_poi_density
            + Democrat_R + Male_R + Over_65_R + Bt_45_64_R + avg_poi_score + Categories
            + s(Household_Below_Poverty_R, Population_Density, by = Categories)
            # + s(Household_Below_Poverty_R, by = Categories)
            + ti(Lat, Lng, bs = 'gp') + s(CBSA, bs = 're'), data = park_df1,
        control = gam.control(trace = TRUE), method = "fREML", discrete = TRUE)
summary(GAM_RES1)
b <- getViz(GAM_RES1, scheme = 3)
pl <- plot(b, select = 2:7)
print(pl, pages = 1)
# ggsave(paste("D:\\Google_Review\\Parking\\results\\", "nonlinear_interaction.png", sep=""), units="in", width=7, height=6, dpi=1200)

# Curve plot
plot_GAM <- function (xvar, yvar, x_label, y_label, park_df1){
  snow_recovery <- mgcv::bam(park_df1[, yvar] ~  s(park_df1[, xvar]) + ti(Lat, Lng, bs = 'gp') + s(CBSA, bs = 're'),
  family=gaussian(), data = park_df1, method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
  b <- getViz(snow_recovery, scheme = 3)
  pl <- plot(b, select = 1) + l_fitLine(colour = "red", size=1.2) + l_rug(mapping = aes(x=x, y=y), alpha = 0.5) +
    l_ciLine(mul = 5, colour = "blue", linetype = 2) + labs(x = x_label, y = y_label) + l_points(shape = 19, size = 1, alpha = 0.1) +
    theme_classic() + theme_bw() + theme(text = element_text(size=30, family="serif")) # + ylim(-5,5)
  print(pl, pages = 1)
  # ggsave(paste("D:\\Disaster\\Results\\", xvar, "_",yvar, ".png", sep=""), units="in", width=7, height=6, dpi=1200)
}

# Interaction plot
pl <- plot(b,select=1) + l_fitRaster(noiseup = FALSE) + l_fitContour(colour = 5,binwidth = 0.05) + l_rug() +
  scale_fill_gradientn(colours = coolwarm(100),na.value=NA) + geom_text_contour(stroke = 0.1,binwidth = 0.2) + # ,limits = lims
  labs() + theme_bw() +
  theme(text = element_text(size=20, family="serif")) + labs(fill = "s(x)")+
  theme(legend.position = "right")
print(pl, pages = 1)

