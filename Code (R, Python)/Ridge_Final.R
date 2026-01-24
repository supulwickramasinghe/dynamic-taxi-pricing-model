# Required libraries only
library(caret)
library(dplyr)
library(glmnet)

setwd("D:/sanduni/Level 3/Sem II/ST 3082 - Statistical Learning I/Final Project")

ride_data <- read.csv("dynamic_pricing.csv")

# Split the data into training and testing sets
set.seed(123)
train_indices <- createDataPartition(ride_data$Historical_Cost_of_Ride, p = 0.8, list = FALSE)
train_data <- ride_data[train_indices, ]
test_data <- ride_data[-train_indices, ]

# Calculate demand_multiplier based on percentiles
high_demand_value <- quantile(train_data$Number_of_Riders, 0.75)
low_demand_value <- quantile(train_data$Number_of_Riders, 0.25)

train_data$demand_multiplier <- ifelse(train_data$Number_of_Riders > high_demand_value,
                                       train_data$Number_of_Riders / high_demand_value,
                                       train_data$Number_of_Riders / low_demand_value)

test_data$demand_multiplier <- ifelse(test_data$Number_of_Riders > high_demand_value,
                                      test_data$Number_of_Riders / high_demand_value,
                                      test_data$Number_of_Riders / low_demand_value)

# Calculate supply_multiplier based on percentiles
high_supply_value <- quantile(train_data$Number_of_Drivers, 0.75)
low_supply_value <- quantile(train_data$Number_of_Drivers, 0.25)

train_data$supply_multiplier <- ifelse(train_data$Number_of_Drivers > low_supply_value,
                                       high_supply_value / train_data$Number_of_Drivers,
                                       low_supply_value / train_data$Number_of_Drivers)

test_data$supply_multiplier <- ifelse(test_data$Number_of_Drivers > low_supply_value,
                                      high_supply_value / test_data$Number_of_Drivers,
                                      low_supply_value / test_data$Number_of_Drivers)

# Price adjustment thresholds
demand_threshold_low <- 0.8
supply_threshold_high <- 0.8

# Calculate adjusted_ride_cost
train_data$adjusted_ride_cost <- train_data$Historical_Cost_of_Ride *
  (pmax(train_data$demand_multiplier, demand_threshold_low) *
     pmax(train_data$supply_multiplier, supply_threshold_high))

test_data$adjusted_ride_cost <- test_data$Historical_Cost_of_Ride *
  (pmax(test_data$demand_multiplier, demand_threshold_low) *
     pmax(test_data$supply_multiplier, supply_threshold_high))

# Prepare data for modeling
train_set <- train_data %>% select(-demand_multiplier, -supply_multiplier, -Historical_Cost_of_Ride)
test_set <- test_data %>% select(-demand_multiplier, -supply_multiplier, -Historical_Cost_of_Ride)

# One Hot Encoding
cat_vars <- names(train_set)[sapply(train_set, is.character) | sapply(train_set, is.factor)]
train_set[cat_vars] <- lapply(train_set[cat_vars], as.factor)
test_set[cat_vars] <- lapply(test_set[cat_vars], as.factor)

dummies_model <- dummyVars(~ ., data = train_set, fullRank = TRUE)
train_encoded <- as.data.frame(predict(dummies_model, newdata = train_set))
test_encoded <- as.data.frame(predict(dummies_model, newdata = test_set))

# Standard scaling
numerical_vars <- names(train_encoded)[!grepl("\\.", names(train_encoded)) & names(train_encoded) != "adjusted_ride_cost"]
scaler <- preProcess(train_encoded[, numerical_vars], method = c("center", "scale"))

train_encoded[, numerical_vars] <- predict(scaler, train_encoded[, numerical_vars])
test_encoded[, numerical_vars] <- predict(scaler, test_encoded[, numerical_vars])

# Separate features and target
X_train <- train_encoded[, names(train_encoded) != "adjusted_ride_cost"]
y_train <- train_encoded$adjusted_ride_cost
X_test <- test_encoded[, names(test_encoded) != "adjusted_ride_cost"]
y_test <- test_encoded$adjusted_ride_cost

# Convert data frames to matrices for glmnet
x_train <- model.matrix(~ ., data = X_train)[, -1]  # Remove intercept
x_test <- model.matrix(~ ., data = X_test)[, -1]
y_train <- y_train

# Fit Ridge Regression model
fit.ridge <- glmnet(x_train, y_train, alpha = 0)

# Plot the coefficient paths
plot(fit.ridge, xvar = "lambda", label = TRUE, lw = 2)

# Cross-validation to find best lambda
cv.ridge <- cv.glmnet(x_train, y_train, alpha = 0)
plot(cv.ridge)
bestlam <- cv.ridge$lambda.min
print(paste("Best lambda:", bestlam))

# Coefficients under best lambda
coef(fit.ridge, s = bestlam)

# Predictions on training set
y_pred_train <- predict(fit.ridge, s = bestlam, newx = x_train)

# Training MSE, RMSE, and R^2
train_mse <- mean((y_train - y_pred_train)^2)
train_rmse <- sqrt(train_mse)
ss_res <- sum((y_train - y_pred_train)^2)
ss_tot <- sum((y_train - mean(y_train))^2)
train_r2 <- 1 - (ss_res / ss_tot)

cat("Training MSE:", train_mse, "\n")
cat("Training RMSE:", train_rmse, "\n")
cat("Training R^2:", train_r2, "\n")

# Predictions on test set
y_pred_test <- predict(fit.ridge, s = bestlam, newx = x_test)

# Test MSE, RMSE, and R^2
test_mse <- mean((y_test - y_pred_test)^2)
test_rmse <- sqrt(test_mse)
ss_res <- sum((y_test - y_pred_test)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
test_r2 <- 1 - (ss_res / ss_tot)

cat("Test MSE:", test_mse, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R^2:", test_r2, "\n")

