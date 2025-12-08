# Variable Selection Analysis for Random Forest Model using OLS Regression
# This script uses OLS regression to evaluate different cumulative rainfall periods
# (7-day, 14-day, 21-day) to determine which variable(s) to use in a Random Forest model

# Load required libraries
library(tidyverse)
library(ggplot2)

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the data
data_path <- "C:/Users/jupf0/Desktop/PSeminarDeepLearning/Risikofaktoren/WaterAndStuff/output/landslides_with_rainfall.csv"
landslide_data <- read.csv(data_path, stringsAsFactors = FALSE)

# Display data structure
cat("=== Data Structure ===\n")
str(landslide_data)
cat("\n=== Summary Statistics ===\n")
summary(landslide_data)

# Data preprocessing
# Remove rows with missing rainfall data
data_clean <- landslide_data %>%
  filter(!is.na(Rainfall_7d_mm) & !is.na(Rainfall_14d_mm) & !is.na(Rainfall_21d_mm))

cat("\n=== Data after removing NAs ===\n")
cat(sprintf("Original rows: %d\n", nrow(landslide_data)))
cat(sprintf("Clean rows: %d\n", nrow(data_clean)))
cat(sprintf("Removed rows: %d\n\n", nrow(landslide_data) - nrow(data_clean)))

# Prepare response variable (MOVEMENT_C as numeric for regression)
data_clean$MOVEMENT_C <- as.numeric(data_clean$MOVEMENT_C)

# Check for multicollinearity between rainfall variables
cat("\n=== Correlation Matrix for Rainfall Variables ===\n")
rainfall_vars <- data_clean[, c("Rainfall_7d_mm", "Rainfall_14d_mm", "Rainfall_21d_mm")]
cor_matrix <- cor(rainfall_vars, use = "complete.obs")
print(cor_matrix)
cat("\nNote: High correlation (>0.8) suggests redundancy between variables\n")

# OLS Model 1: Only 7-day rainfall
cat("\n=== OLS Model 1: 7-day Rainfall ===\n")
model_7d <- lm(MOVEMENT_C ~ Rainfall_7d_mm, data = data_clean)
sum_7d <- summary(model_7d)
print(sum_7d)
cat(sprintf("R-squared: %.4f, Adj R-squared: %.4f\n", sum_7d$r.squared, sum_7d$adj.r.squared))
cat(sprintf("F-statistic: %.4f (p-value: %.4e)\n", 
            sum_7d$fstatistic[1], 
            pf(sum_7d$fstatistic[1], sum_7d$fstatistic[2], sum_7d$fstatistic[3], lower.tail = FALSE)))

# OLS Model 2: Only 14-day rainfall
cat("\n=== OLS Model 2: 14-day Rainfall ===\n")
model_14d <- lm(MOVEMENT_C ~ Rainfall_14d_mm, data = data_clean)
sum_14d <- summary(model_14d)
print(sum_14d)
cat(sprintf("R-squared: %.4f, Adj R-squared: %.4f\n", sum_14d$r.squared, sum_14d$adj.r.squared))
cat(sprintf("F-statistic: %.4f (p-value: %.4e)\n", 
            sum_14d$fstatistic[1], 
            pf(sum_14d$fstatistic[1], sum_14d$fstatistic[2], sum_14d$fstatistic[3], lower.tail = FALSE)))

# OLS Model 3: Only 21-day rainfall
cat("\n=== OLS Model 3: 21-day Rainfall ===\n")
model_21d <- lm(MOVEMENT_C ~ Rainfall_21d_mm, data = data_clean)
sum_21d <- summary(model_21d)
print(sum_21d)
cat(sprintf("R-squared: %.4f, Adj R-squared: %.4f\n", sum_21d$r.squared, sum_21d$adj.r.squared))
cat(sprintf("F-statistic: %.4f (p-value: %.4e)\n", 
            sum_21d$fstatistic[1], 
            pf(sum_21d$fstatistic[1], sum_21d$fstatistic[2], sum_21d$fstatistic[3], lower.tail = FALSE)))

# OLS Model 4: All three rainfall periods
cat("\n=== OLS Model 4: All Rainfall Periods ===\n")
model_all <- lm(MOVEMENT_C ~ Rainfall_7d_mm + Rainfall_14d_mm + Rainfall_21d_mm, 
                data = data_clean)
sum_all <- summary(model_all)
print(sum_all)
cat(sprintf("R-squared: %.4f, Adj R-squared: %.4f\n", sum_all$r.squared, sum_all$adj.r.squared))
cat(sprintf("F-statistic: %.4f (p-value: %.4e)\n", 
            sum_all$fstatistic[1], 
            pf(sum_all$fstatistic[1], sum_all$fstatistic[2], sum_all$fstatistic[3], lower.tail = FALSE)))

# OLS Model 5: Incremental rainfall effects
# Calculate incremental rainfall between periods
data_clean <- data_clean %>%
  mutate(
    Rainfall_0_7d = Rainfall_7d_mm,
    Rainfall_8_14d = Rainfall_14d_mm - Rainfall_7d_mm,
    Rainfall_15_21d = Rainfall_21d_mm - Rainfall_14d_mm
  )

cat("\n=== OLS Model 5: Incremental Rainfall Effects ===\n")
model_incremental <- lm(MOVEMENT_C ~ Rainfall_0_7d + Rainfall_8_14d + Rainfall_15_21d, 
                        data = data_clean)
sum_inc <- summary(model_incremental)
print(sum_inc)
cat(sprintf("R-squared: %.4f, Adj R-squared: %.4f\n", sum_inc$r.squared, sum_inc$adj.r.squared))
cat(sprintf("F-statistic: %.4f (p-value: %.4e)\n", 
            sum_inc$fstatistic[1], 
            pf(sum_inc$fstatistic[1], sum_inc$fstatistic[2], sum_inc$fstatistic[3], lower.tail = FALSE)))

# Compare OLS models
cat("\n=== OLS Model Comparison ===\n")
model_comparison <- data.frame(
  Model = c("7-day", "14-day", "21-day", "All Periods", "Incremental"),
  R_squared = c(sum_7d$r.squared, sum_14d$r.squared, sum_21d$r.squared,
                sum_all$r.squared, sum_inc$r.squared),
  Adj_R_squared = c(sum_7d$adj.r.squared, sum_14d$adj.r.squared, sum_21d$adj.r.squared,
                    sum_all$adj.r.squared, sum_inc$adj.r.squared),
  AIC = c(AIC(model_7d), AIC(model_14d), AIC(model_21d),
          AIC(model_all), AIC(model_incremental)),
  BIC = c(BIC(model_7d), BIC(model_14d), BIC(model_21d),
          BIC(model_all), BIC(model_incremental)),
  Num_Variables = c(1, 1, 1, 3, 3)
)

print(model_comparison)
cat("\nNote: Higher R-squared/Adj R-squared and lower AIC/BIC indicate better model fit\n")

# Identify best model (based on highest adjusted R-squared)
best_model_idx <- which.max(model_comparison$Adj_R_squared)
best_model_name <- model_comparison$Model[best_model_idx]
cat(sprintf("\n=== Best OLS Model: %s ===\n", best_model_name))
cat(sprintf("R-squared: %.4f\n", model_comparison$R_squared[best_model_idx]))
cat(sprintf("Adjusted R-squared: %.4f\n", model_comparison$Adj_R_squared[best_model_idx]))
cat(sprintf("AIC: %.2f, BIC: %.2f\n", 
            model_comparison$AIC[best_model_idx], 
            model_comparison$BIC[best_model_idx]))

best_model <- switch(best_model_name,
                     "7-day" = model_7d,
                     "14-day" = model_14d,
                     "21-day" = model_21d,
                     "All Periods" = model_all,
                     "Incremental" = model_incremental)

# Coefficients for best model
cat("\n=== Coefficients for Best Model ===\n")
print(summary(best_model)$coefficients)

# Visualizations
cat("\n=== Creating Visualizations ===\n")

# Model performance comparison plot (R-squared)
png("ols_model_comparison.png", width = 1000, height = 600)
par(mar = c(5, 6, 4, 2))
barplot(model_comparison$Adj_R_squared, 
        names.arg = model_comparison$Model,
        main = "OLS Model Performance Comparison (Adjusted R-squared)",
        ylab = "Adjusted R-squared",
        xlab = "Model",
        col = "steelblue",
        ylim = c(0, max(model_comparison$Adj_R_squared) * 1.2),
        las = 2)
abline(h = seq(0, 1, 0.1), col = "gray80", lty = 2)
text(x = seq_len(nrow(model_comparison)) * 1.2 - 0.5, 
     y = model_comparison$Adj_R_squared + max(model_comparison$Adj_R_squared) * 0.05,
     labels = sprintf("%.4f", model_comparison$Adj_R_squared),
     cex = 0.9)
dev.off()
cat("Model comparison plot saved to: ols_model_comparison.png\n")

# Coefficient plots for multi-variable models
png("coefficient_comparison.png", width = 1200, height = 600)
par(mfrow = c(1, 2))

# All Periods model coefficients
coef_all <- summary(model_all)$coefficients[-1, ]  # Exclude intercept
barplot(coef_all[, "Estimate"],
        names.arg = rownames(coef_all),
        main = "Coefficients: All Periods",
        ylab = "Coefficient Estimate",
        las = 2,
        col = "steelblue")

# Incremental model coefficients
coef_inc <- summary(model_incremental)$coefficients[-1, ]  # Exclude intercept
barplot(coef_inc[, "Estimate"],
        names.arg = rownames(coef_inc),
        main = "Coefficients: Incremental",
        ylab = "Coefficient Estimate",
        las = 2,
        col = "darkgreen")

dev.off()
cat("Coefficient comparison plots saved to: coefficient_comparison.png\n")

# Summary report
cat("\n==========================================================\n")
cat("=== VARIABLE SELECTION RECOMMENDATION FOR RANDOM FOREST ===\n")
cat("===         (Based on OLS Regression Analysis)        ===\n")
cat("==========================================================\n\n")
cat(sprintf("Total landslide events analyzed: %d\n", nrow(data_clean)))
cat(sprintf("\nBest performing OLS model: %s\n", best_model_name))
cat(sprintf("  - R-squared: %.4f\n", model_comparison$R_squared[best_model_idx]))
cat(sprintf("  - Adjusted R-squared: %.4f\n", model_comparison$Adj_R_squared[best_model_idx]))
cat(sprintf("  - AIC: %.2f, BIC: %.2f\n", 
            model_comparison$AIC[best_model_idx], 
            model_comparison$BIC[best_model_idx]))
cat(sprintf("  - Number of variables: %d\n", model_comparison$Num_Variables[best_model_idx]))

cat("\n=== Model Performance Summary ===\n")
print(model_comparison)

cat("\n=== Correlation Between Rainfall Variables ===\n")
print(cor_matrix)

if (best_model_name %in% c("All Periods", "Incremental")) {
  cat("\n=== Coefficient Significance ===\n")
  coef_summary <- summary(best_model)$coefficients
  print(coef_summary)
  
  # Identify significant predictors
  sig_vars <- rownames(coef_summary)[coef_summary[, "Pr(>|t|)"] < 0.05][-1]  # Exclude intercept
  
  cat("\n=== RECOMMENDATION ===\n")
  cat(sprintf("For your Random Forest model, consider using:\n"))
  cat(sprintf("1. PRIMARY CHOICE: %s model (highest Adj R-squared: %.4f)\n", 
              best_model_name, model_comparison$Adj_R_squared[best_model_idx]))
  
  if (length(sig_vars) > 0) {
    cat(sprintf("   Significant predictors (p < 0.05): %s\n", paste(sig_vars, collapse = ", ")))
  } else {
    cat("   Note: No variables are statistically significant at p < 0.05\n")
  }
  
  if (best_model_name == "All Periods") {
    cat("   Variables: Rainfall_7d_mm, Rainfall_14d_mm, Rainfall_21d_mm\n")
  } else {
    cat("   Variables: Rainfall_0_7d, Rainfall_8_14d, Rainfall_15_21d\n")
  }
} else {
  cat("\n=== RECOMMENDATION ===\n")
  cat(sprintf("For your Random Forest model, use: %s\n", best_model_name))
  coef_summary <- summary(best_model)$coefficients
  p_value <- coef_summary[2, "Pr(>|t|)"]
  if (p_value < 0.05) {
    cat(sprintf("This variable is statistically significant (p = %.4f)\n", p_value))
  } else {
    cat(sprintf("Warning: This variable is not statistically significant (p = %.4f)\n", p_value))
  }
}


cat("\n=== Analysis Complete ===\n")
