library(tidyverse)
library(caret)
library(dplyr)
library(lmtest)
library(car)
library(leaps)

setwd("D:/sanduni/Level 3/Sem II/ST 3082 - Statistical Learning I/Final Project")

ride_data <- read.csv("dynamic_pricing.csv")

# Split the data into training and testing sets
set.seed(123)
train_indices <- createDataPartition(ride_data$Historical_Cost_of_Ride, p = 0.8, list = FALSE)
train_data <- ride_data[train_indices, ]
test_data <- ride_data[-train_indices, ]

# Calculate demand_multiplier based on percentile for high and low demand
high_demand_percentile <- 75
low_demand_percentile <- 25

high_demand_value <- quantile(train_data$Number_of_Riders, high_demand_percentile / 100)
low_demand_value <- quantile(train_data$Number_of_Riders, low_demand_percentile / 100)

train_data$demand_multiplier <- ifelse(train_data$Number_of_Riders > high_demand_value,
                                       train_data$Number_of_Riders / high_demand_value,
                                       train_data$Number_of_Riders / low_demand_value)

test_data$demand_multiplier <- ifelse(test_data$Number_of_Riders > high_demand_value,
                                      test_data$Number_of_Riders / high_demand_value,
                                      test_data$Number_of_Riders / low_demand_value)

# Calculate supply_multiplier based on percentile for high and low supply
high_supply_percentile <- 75
low_supply_percentile <- 25

high_supply_value <- quantile(train_data$Number_of_Drivers, high_supply_percentile / 100)
low_supply_value <- quantile(train_data$Number_of_Drivers, low_supply_percentile / 100)

train_data$supply_multiplier <- ifelse(train_data$Number_of_Drivers > low_supply_value,
                                       high_supply_value / train_data$Number_of_Drivers,
                                       low_supply_value / train_data$Number_of_Drivers)

test_data$supply_multiplier <- ifelse(test_data$Number_of_Drivers > low_supply_value,
                                      high_supply_value / test_data$Number_of_Drivers,
                                      low_supply_value / test_data$Number_of_Drivers)

# Define price adjustment factors for high and low demand/supply
demand_threshold_high <- 1.2  # Higher demand threshold
demand_threshold_low <- 0.8    # Lower demand threshold
supply_threshold_high <- 0.8  # Higher supply threshold
supply_threshold_low <- 1.2    # Lower supply threshold

# Calculate adjusted_ride_cost for dynamic pricing
train_data$adjusted_ride_cost <- train_data$Historical_Cost_of_Ride * (
  pmax(train_data$demand_multiplier, demand_threshold_low) *
    pmax(train_data$supply_multiplier, supply_threshold_high)
)

test_data$adjusted_ride_cost <- test_data$Historical_Cost_of_Ride * (
  pmax(test_data$demand_multiplier, demand_threshold_low) *
    pmax(test_data$supply_multiplier, supply_threshold_high)
)


# arrange the training and testing dataset to fit models
train_set <- train_data %>%
  select(-demand_multiplier, -supply_multiplier, -Historical_Cost_of_Ride)

test_set <- test_data %>%
  select(-demand_multiplier, -supply_multiplier, -Historical_Cost_of_Ride)

#write.csv(train_set, "train_set.csv", row.names = FALSE)

#### One Hot Encoding the categorical variables

# Identify categorical variables
cat_vars <- names(train_set)[sapply(train_set, is.character) | sapply(train_set, is.factor)]

# Convert character variables to factors 
train_set[cat_vars] <- lapply(train_set[cat_vars], as.factor)
test_set[cat_vars] <- lapply(test_set[cat_vars], as.factor)

# Create the dummyVars object based on train_set
dummies_model <- dummyVars(~ ., data = train_set, fullRank = TRUE) 

# Apply transformation to train and test set
train_encoded <- as.data.frame(predict(dummies_model, newdata = train_set))
test_encoded <- as.data.frame(predict(dummies_model, newdata = test_set))

# Standard scaling for numerical variables
numerical_vars <- names(train_encoded)[!names(train_encoded) %in% names(train_encoded)[grepl("\\.", names(train_encoded))] & names(train_encoded) != "adjusted_ride_cost"]

scaler <- preProcess(train_encoded[, numerical_vars], method = c("center", "scale"))

train_encoded[, numerical_vars] <- predict(scaler, train_encoded[, numerical_vars])
test_encoded[, numerical_vars] <- predict(scaler, test_encoded[, numerical_vars])

# Separate features and target variable
X_train <- train_encoded[, names(train_encoded) != "adjusted_ride_cost"]
y_train <- train_encoded$adjusted_ride_cost
X_test <- test_encoded[, names(test_encoded) != "adjusted_ride_cost"]
y_test <- test_encoded$adjusted_ride_cost

# Best Subset Selection
nvmax_value <- min(15, ncol(train_encoded) - 1)
regfit.full <- regsubsets(adjusted_ride_cost ~ ., data = train_encoded, nvmax = nvmax_value)
reg_summary <- summary(regfit.full)

# Plot Selection Criteria
par(mfrow = c(1, 2))  # 1 row, 2 columns of plots

# Plot Cp
plot(1:length(reg_summary$cp), reg_summary$cp,
     xlab = "Number of Variables", ylab = "Cp", main = "Cp Selection", type = "b")
points(which.min(reg_summary$cp), min(reg_summary$cp), col = "red", pch = 20)

# Plot Adjusted R-squaredS
plot(1:length(reg_summary$adjr2), reg_summary$adjr2,
     xlab = "Number of Variables", ylab = "Adjusted R²", main = "Adjusted R² Selection", type = "b")
points(which.max(reg_summary$adjr2), max(reg_summary$adjr2), col = "red", pch = 20)

# Prediction Function for regsubsets
predict.regsubsets <- function(object, newdata, id) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  mat[, names(coefi), drop = FALSE] %*% coefi
}

# Cross-Validation
set.seed(11)
k_folds <- 10
folds <- sample(rep(1:k_folds, length = nrow(train_encoded)))
cv.errors <- matrix(NA, k_folds, nvmax_value)

for (k in 1:k_folds) {
  best.fit <- regsubsets(adjusted_ride_cost ~ ., data = train_encoded[folds != k, ], nvmax = nvmax_value)
  
  for (i in 1:nvmax_value) {
    pred <- predict.regsubsets(best.fit, train_encoded[folds == k, ], id = i)
    actual <- train_encoded$adjusted_ride_cost[folds == k]
    cv.errors[k, i] <- mean((actual - pred)^2)
  }
}

rmse.cv <- sqrt(apply(cv.errors, 2, mean))

# Plot Cross-validation Results
plot(rmse.cv, pch = 20, type = "b", col = "black", ylab = "RMSE", xlab = "Number of Variables", main = "10-Fold CV RMSE")

# Final Model Selection
final_model_size <- which.min(rmse.cv)
regfit.final <- regsubsets(adjusted_ride_cost ~ ., data = train_encoded, nvmax = final_model_size)

# Training & Test Predictions
train_pred <- predict.regsubsets(regfit.final, train_encoded, id = final_model_size)
test_pred <- predict.regsubsets(regfit.final, test_encoded, id = final_model_size)

# Performance Metrics
train_rmse <- sqrt(mean((y_train - train_pred)^2))
test_rmse <- sqrt(mean((y_test - test_pred)^2))

train_r2 <- 1 - sum((y_train - train_pred)^2) / sum((y_train - mean(y_train))^2)
test_r2 <- 1 - sum((y_test - test_pred)^2) / sum((y_test - mean(y_test))^2)

cat(sprintf("Final Model Size: %d\n", final_model_size))

# Extracting coefficients of the final model
final_model_coefficients <- coef(regfit.final, final_model_size)
print(final_model_coefficients)

cat(sprintf("Training RMSE: %.4f, Training R²: %.4f\n", train_rmse, train_r2))
cat(sprintf("Test RMSE: %.4f, Test R²: %.4f\n", test_rmse, test_r2))

# Create a data frame for residual diagnostics
residuals <- y_train - train_pred  # residuals
fitted_values <- train_pred        # fitted values

# Residuals vs Fitted Values Plot
par(mfrow = c(1, 1))  # Reset plotting area
plot(fitted_values, residuals, 
     main = "Residuals vs Fitted",
     xlab = "Fitted Values", 
     ylab = "Residuals", 
     pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2, lty = 2)

# Histogram of Residuals
hist(residuals, breaks = 30, 
     main = "Histogram of Residuals", 
     xlab = "Residuals", 
     col = "lightblue", border = "black")

# Q-Q Plot of Residuals
qqnorm(residuals, main = "Q-Q Plot of Residuals", pch = 20)
qqline(residuals, col = "red", lwd = 2)

# Shapiro-Wilk Test for Normality
shapiro_result <- shapiro.test(residuals)
print(shapiro_result)

# -----------------------------
# Fit final linear model using selected variables
# -----------------------------
# Remove intercept if present
final_model_coefficients <- final_model_coefficients[names(final_model_coefficients) != "(Intercept)"]

# Extract final variable names
selected_vars <- names(final_model_coefficients)

# Create linear model for VIF and Durbin-Watson (requires full linear model object)
final_lm_model <- lm(adjusted_ride_cost ~ ., 
                     data = train_encoded[, c("adjusted_ride_cost", selected_vars)])

# Calculate VIF
vif_values <- vif(final_lm_model)
print(vif_values)

# Durbin-Watson Test
dw_result <- dwtest(final_lm_model)
print(dw_result)
