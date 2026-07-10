# Load necessary libraries
library(dplyr)
library(caret)
library(xgboost)
library(Matrix)
library(parallel)
library(doParallel)

# Set up parallel processing to speed up the tuning process
cl <- makePSOCKcluster(detectCores() - 1)  # Using all cores except one
registerDoParallel(cl)

# Assuming you've already run the code in paste.txt to prepare the data
library(dplyr)
library(caret)

#setwd("D:/university/Level 3/2nd Sem/ST 3082/Final project")

ride_data <- read.csv("dynamic_pricing.csv")

# Split the data into training and testing sets
set.seed(123)
train_indices <- createDataPartition(ride_data$Historical_Cost_of_Ride, p = 0.8, list = FALSE)
train_data <- ride_data[train_indices, ]
test_data <- ride_data[-train_indices, ]

colSums(is.na(train_data))
colSums(is.na(test_data))

duplicates_train <- train_data[duplicated(train_data), ]
print(duplicates_train) # Displays duplicate rows

duplicates_test <- test_data[duplicated(test_data), ]
print(duplicates_test)

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

# Extract target variable for training and testing sets
train_y <- train_data$adjusted_ride_cost
test_y <- test_data$adjusted_ride_cost

# Using the encoded data from your previous code
# Convert to matrix format for xgboost
train_features <- as.matrix(train_encoded[, !names(train_encoded) %in% "adjusted_ride_cost"])
test_features <- as.matrix(test_encoded[, !names(test_encoded) %in% "adjusted_ride_cost"])

# Create xgb.DMatrix objects
dtrain <- xgb.DMatrix(data = train_features, label = train_y)
dtest <- xgb.DMatrix(data = test_features, label = test_y)

# This grid search isn't used in the updated approach, but keeping for reference
# Set up grid search for hyperparameter tuning
# xgb_grid <- expand.grid(
#   nrounds = c(100, 200, 300),
#   max_depth = c(3, 5, 7),
#   eta = c(0.01, 0.05, 0.1),
#   gamma = c(0, 0.1, 0.2),
#   colsample_bytree = c(0.6, 0.8, 1.0),
#   min_child_weight = c(1, 3, 5),
#   subsample = c(0.6, 0.8, 1.0)
# )

# Function to evaluate model with specific parameters
evaluate_model <- function(nrounds, max_depth, eta, gamma, colsample_bytree, 
                           min_child_weight, subsample) {
  params <- list(
    objective = "reg:squarederror",
    max_depth = max_depth,
    eta = eta,
    gamma = gamma,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    subsample = subsample
  )
  
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 10,
    verbose = 0,
    prediction = TRUE
  )
  
  # Get the best iteration based on test_rmse_mean
  best_iter <- which.min(cv_results$evaluation_log$test_rmse_mean)
  
  # Return the best RMSE
  return(cv_results$evaluation_log$test_rmse_mean[best_iter])
}

# Use an alternative approach with caret package for hyperparameter tuning
# This is more robust and handles errors better
train_control <- trainControl(
  method = "cv",
  number = 5, 
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Define a more comprehensive parameter grid for full hyperparameter tuning
tune_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.6, 0.8, 1.0),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.7, 0.85, 1.0)
)

cat("Starting hyperparameter tuning with caret...\n")

# Train the model using caret with the comprehensive grid
start_time <- Sys.time()
xgb_model <- train(
  x = train_features,
  y = train_y,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE",
  verbose = TRUE
)
end_time <- Sys.time()

# Stop the cluster
stopCluster(cl)

# Report tuning time
tuning_time <- difftime(end_time, start_time, units = "mins")
cat("\nFull hyperparameter tuning completed in", round(tuning_time, 2), "minutes\n")

# Save detailed tuning results to a file
tuning_results <- as.data.frame(xgb_model$results)
tuning_results <- tuning_results[order(tuning_results$RMSE), ]
write.csv(tuning_results, "xgboost_tuning_results.csv", row.names = FALSE)

# Print the results
cat("\nTop 5 parameter combinations:\n")
print(head(tuning_results, 5))
cat("\nBest parameters:\n")
print(xgb_model$bestTune)


# Store best parameters
best_params <- xgb_model$bestTune

# Train final model with best parameters
cat("\nTraining final model with best parameters...\n")
final_params <- list(
  objective = "reg:squarederror",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  gamma = best_params$gamma,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  subsample = best_params$subsample
)

final_model <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = best_params$nrounds,
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 50
)

# Make predictions on both train and test data
train_pred <- predict(final_model, dtrain)
test_pred <- predict(final_model, dtest)

# Generate detailed evaluation reports
train_metrics <- create_evaluation_report(train_y, train_pred, "Training Data")
test_metrics <- create_evaluation_report(test_y, test_pred, "Test Data")

# Save metrics to a data frame for export
all_metrics <- data.frame(
  Dataset = c("Training", "Test"),
  RMSE = c(train_metrics$RMSE, test_metrics$RMSE),
  MSE = c(train_metrics$MSE, test_metrics$MSE),
  R_squared = c(train_metrics$R_squared, test_metrics$R_squared),
  MAE = c(train_metrics$MAE, test_metrics$MAE),
  MAPE = c(train_metrics$MAPE, test_metrics$MAPE)
)

# Export metrics
write.csv(all_metrics, "xgboost_performance_metrics.csv", row.names = FALSE)

# Function to calculate performance metrics
calculate_metrics <- function(actual, predicted) {
  # RMSE
  rmse <- sqrt(mean((predicted - actual)^2))
  
  # MSE (Mean Squared Error)
  mse <- mean((predicted - actual)^2)
  
  # R-squared
  ss_total <- sum((actual - mean(actual))^2)
  ss_residual <- sum((actual - predicted)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # MAE (Mean Absolute Error)
  mae <- mean(abs(predicted - actual))
  
  # MAPE (Mean Absolute Percentage Error)
  mape <- mean(abs((actual - predicted) / actual)) * 100
  
  return(list(
    RMSE = rmse,
    MSE = mse,
    R_squared = r_squared,
    MAE = mae,
    MAPE = mape
  ))
}

# Create a more detailed evaluation report
create_evaluation_report <- function(actual, predicted, dataset_name) {
  metrics <- calculate_metrics(actual, predicted)
  
  cat("\n", dataset_name, "Metrics:\n", sep="")
  cat("--------------------------------------------------------\n")
  cat(sprintf("%-15s: %f\n", "RMSE", metrics$RMSE))
  cat(sprintf("%-15s: %f\n", "MSE", metrics$MSE))
  cat(sprintf("%-15s: %f\n", "R-squared", metrics$R_squared))
  cat(sprintf("%-15s: %f\n", "MAE", metrics$MAE))
  cat(sprintf("%-15s: %f%%\n", "MAPE", metrics$MAPE))
  cat("--------------------------------------------------------\n")
  
  return(metrics)
}

# Variable importance
importance_matrix <- xgb.importance(feature_names = colnames(train_features), model = final_model)
print(importance_matrix[1:15,])

# Export variable importance
write.csv(importance_matrix, "xgboost_variable_importance.csv", row.names = FALSE)

# Plot variable importance
importance_plot <- xgb.plot.importance(importance_matrix, top_n = 15,col = "navy")
print(importance_plot)

# Optional: Plot actual vs predicted values
plot(test_y, test_pred, 
     xlab = "Actual Price", ylab = "Predicted Price",
     main = "Actual vs Predicted Prices (Test Data)",
     pch = 16, col = "blue")
abline(a = 0, b = 1, col = "red")

# Save the model
xgb.save(final_model, "xgboost_dynamic_pricing_model.model")
cat("\nModel saved to 'xgboost_dynamic_pricing_model.model'\n")
