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
print(duplicates_train)  # Displays duplicate rows

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
demand_threshold_low <- 0.8   # Lower demand threshold
supply_threshold_high <- 0.8  # Higher supply threshold
supply_threshold_low <- 1.2   # Lower supply threshold

# Calculate adjusted_ride_cost for dynamic pricing
train_data$adjusted_ride_cost <- train_data$Historical_Cost_of_Ride * (
  pmax(train_data$demand_multiplier, demand_threshold_low) *
    pmax(train_data$supply_multiplier, supply_threshold_high)
)

test_data$adjusted_ride_cost <- test_data$Historical_Cost_of_Ride * (
  pmax(test_data$demand_multiplier, demand_threshold_low) *
    pmax(test_data$supply_multiplier, supply_threshold_high)
)




############################ Exploratory Data Analysis #####################################

# Defining the categorical variables
cat_vars = train_data[, c('Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type')]

# Defining the numerical variables
num_vars = train_data[, c('Expected_Ride_Duration', 'Number_of_Riders', 'Number_of_Drivers', 'Average_Ratings', 'Number_of_Past_Rides')]

# Defining the response variable
res = train_data[, c('adjusted_ride_cost')]


############### One-way Analysis #################

# Descriptive statistics for categorical variables
summary(cat_vars)
# Descriptive statistics for numerical variables
summary(num_vars)
summary(res)

library(ggplot2)

# Plot histograms for numerical variables from training data
par(mfrow = c(2, 3))
for (col in colnames(num_vars)) {
  hist(train_data[[col]], 
       main = paste(col, "Distribution"), 
       col = "#800000", 
       border = "white", 
       probability = FALSE, 
       xlab = col)
}


par(mfrow = c(1, 1))

# Histogram for the response variable (adjusted_ride_cost)
hist(train_data$adjusted_ride_cost, 
     main = "Adjusted Ride Cost Distribution", 
     col = "#800000", 
     border = "white", 
     xlab = "Adjusted Ride Cost", 
     ylab = "Count",
     probability = FALSE)  

## For categorical variables

library(ggplot2)
library(dplyr)
library(gridExtra)
# Initialize an empty list to store the plots
plot_list <- list()

# Loop through each categorical variable
for (col in colnames(cat_vars)) {
  
  # Calculate the percentage for each level in the categorical variable
  cat_percentage <- as.data.frame(table(train_data[[col]]))
  cat_percentage$percentage <- (cat_percentage$Freq / sum(cat_percentage$Freq)) * 100
  
  # Create the bar plot with percentage on y-axis
  p <- ggplot(cat_percentage, aes(x = Var1, y = percentage)) +
    geom_bar(stat = "identity", fill = "#800000", color = "white") +
    labs(title = paste("Percentage of", col), x = col, y = "Percentage") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels if needed
  
  # Add the plot to the list
  plot_list[[col]] <- p
}

# Arrange the plots in a grid layout (2 rows, 3 columns)
grid.arrange(grobs = plot_list, ncol = 3)



############### Two-way Analysis #################

library(ggplot2)

### Scatter plots for numerical variables with the adjusted cost

for (col in colnames(num_vars)) {
 p= ggplot(train_data, aes_string(x = col, y = "adjusted_ride_cost")) +
    geom_point(color = "#800000", alpha = 0.7) + 
    geom_smooth(method = "lm", color = "#800000", linetype = "solid",se = FALSE) + 
   labs(title = paste("Scatter Plot of", gsub("_", " ", col), "vs", "Adjusted Ride Cost"),
        x = gsub("_", " ", col), y = "Adjusted Ride Cost") +
    theme_minimal() 
 
 print(p)
}

# scatter plot of number of riders
NR= num_vars[,c("Number_of_Riders")]
p <- ggplot(train_data, aes_string(x = NR, y = "adjusted_ride_cost")) +
  geom_point(color = "#800000", alpha = 0.7) +
  geom_smooth(method = "loess", color = "#800000", se = FALSE) +
  labs(title = paste("Scatter Plot of", "Number of Riders", "vs", "Adjusted Ride Cost"),
       x = "Number of Riders", y = "Adjusted ride cost") +
  theme_minimal()

# Print the plot
print(p)


# Scatter plot between Historical_Cost_of_Ride and adjusted_ride_cost
ggplot(train_data, aes(x = adjusted_ride_cost, y = Historical_Cost_of_Ride)) +
  geom_point(color = "#800000", alpha = 0.7) +
  geom_smooth(method = "lm", color = "#F28585",se = FALSE) +
  theme_minimal() +
  labs(title = "Scatter Plot of Historical Cost of Ride vs Adjusted Ride Cost", x = "Adjusted Ride Cost",y = "Historical Cost of Ride")






### Box plots for numerical variables with the adjusted cost

# List to store boxplots
boxplot_list <- list()

# Loop through each numerical variable
for (col in colnames(num_vars)) {
  
  # Bin the numeric variable into 4 groups (you can adjust breaks as needed)
  train_data[[paste0(col, "_bin")]] <- cut(train_data[[col]], breaks = 4, include.lowest = TRUE)
  
  # Create boxplot of adjusted_ride_cost by binned numerical variable
  p <- ggplot(train_data, aes_string(x = paste0(col, "_bin"), y = "adjusted_ride_cost")) +
    geom_boxplot(fill = "#800000", alpha = 0.6) +
    labs(title = paste("Boxplot of Adjusted Ride Cost by", col),
         x = paste0(col, " (binned)"), y = "Adjusted Ride Cost") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  boxplot_list[[col]] <- p
}

# Arrange plots in a grid (adjust ncol as needed)
grid.arrange(grobs = boxplot_list, ncol = 2)




### Bar plots for categorical variables with the average adjusted cost

plot_list <- list()
for (col in colnames(cat_vars)) {
  p <- ggplot(train_data, aes_string(x = col, y = "adjusted_ride_cost")) +
    geom_bar(stat = "summary", fun = "mean", fill = "#800000", color = "white", width = 0.5) +
    labs(title = paste("Avg Adjusted Ride Cost by", col),
         x = col, y = "Average Adjusted Ride Cost") +
    theme_minimal()
  plot_list[[col]] <- p
}
grid.arrange(grobs = plot_list, ncol = 2)






### Correlation plot between the response and the numerical predictors

library(ggplot2)
library(dplyr)
library(reshape2)
library(RColorBrewer)


# Compute correlations with the target variable
correlations <- sapply(num_vars, function(x) cor(x, res, use = "complete.obs"))

# Convert to data frame
cor_df <- data.frame(
  variable = names(correlations),
  correlation = as.numeric(correlations)
)

# Sort by absolute correlation values in ascending order
cor_df <- cor_df %>% arrange(correlation)  # Ascending order by correlation

# Add dummy x-axis column for target
cor_df$target <- "adjusted_ride_cost"

# Reorder factor levels based on sorted correlation values in ascending order
cor_df$variable <- factor(cor_df$variable, levels = cor_df$variable)

# Create plot
ggplot(cor_df, aes(x = target, y = variable, fill = correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(correlation, 4)), size = 4, color = "white") +
  scale_fill_gradient2(low = "#05002B", mid = "#B33D79", high = "#FFE6E6",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlations between predictors and Adjusted Ride Cost",
       x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks = element_blank())


### Correlation heat map of numerical predictor variables

# Load required libraries
library(ggplot2)
library(reshape2)
# Compute correlation matrix
cor_matrix <- round(cor(num_vars, use = "complete.obs"), 2)

# Melt the correlation matrix for ggplot
cor_melted <- melt(cor_matrix)

cor_melted$Var2 <- factor(cor_melted$Var2, levels = rev(unique(cor_melted$Var1)))

# Define the color palette to match the plot
custom_palette <- c("black","#40154D","#440154", "#800000","#F1948A", "#FFE6E6")

# Now plot the heatmap with the custom color scheme
ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "black") +
  geom_text(aes(label = value, color = ifelse(value == 1, "black", "white")), size = 4) +
  scale_fill_gradientn(colors = custom_palette,
                       limit = c(-1, 1),
                       name = "Correlation") +
  scale_color_identity() + 
  theme_minimal() +
  theme(axis.text.x = element_text( vjust = 4,
                                   size = 10.5, hjust = 0.5),
        axis.text.y = element_text(size = 12)) +
  labs(title = "Correlation Heat Map", x = "", y = "")



