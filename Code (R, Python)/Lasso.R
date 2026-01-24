library(dplyr)
library(caret)
library(glmnet)

setwd("D:/university/Level 3/2nd Sem/ST 3082/Final project")

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

#str(train_encoded)
#str(test_encoded)

###### Fitting Lasso ########

x <- model.matrix(adjusted_ride_cost ~ . - 1, data = train_encoded)
y <- train_encoded$adjusted_ride_cost

fit.elasticnet <- glmnet(x, y, alpha = 0.5)
plot(fit.elasticnet, xvar = "lambda", label = TRUE, lw = 2)

# Cross-validation to find best lambda
cv.elasticnet <- cv.glmnet(x, y, alpha = 0.5)
plot(cv.elasticnet)
bestlam <- cv.elasticnet$lambda.min
print(paste("Best Lambda:", bestlam))

# Coefficients at best lambda
print(coef(fit.elasticnet, s = bestlam))

# Predictions on training set
y_pred_train <- predict(fit.elasticnet, s = bestlam, newx = x)

# Training MSE
train_mse <- mean((y - y_pred_train)^2)
print(paste("Training MSE:", train_mse))

# Training R^2
ss_res_train <- sum((y - y_pred_train)^2)
ss_tot_train <- sum((y - mean(y))^2)
train_r2 <- 1 - (ss_res_train / ss_tot_train)
print(paste("Training R^2:", train_r2))

# Prepare test matrix
x_test <- model.matrix(adjusted_ride_cost ~ . - 1, data = test_encoded)
y_test <- test_encoded$adjusted_ride_cost

# Test set predictions
y_pred_test <- predict(fit.elasticnet, s = bestlam, newx = x_test)

# Test MSE
test_mse <- mean((y_test - y_pred_test)^2)
print(paste("Test MSE:", test_mse))

# Test R^2
ss_res_test <- sum((y_test - y_pred_test)^2)
ss_tot_test <- sum((y_test - mean(y_test))^2)
test_r2 <- 1 - (ss_res_test / ss_tot_test)
print(paste("Test R^2:", test_r2))
