# Script Settings and Resources 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(haven) 
library(tidyverse) 
library(caret) 
# Adding parallel processing and timing libraries demonstrated in class
library(parallel)
library(doParallel)
library(tictoc)

# Data Import and Cleaning 
gss_tbl <- read_sav("../data/GSS2016.sav") |> 
  drop_na(mosthrs) |>
  mutate(mosthrs = as.integer(mosthrs)) |> 
  select(-hrs1, -hrs2) |> 
  select(where(~ mean(is.na(.)) < 0.75)) |>
  mutate(across(everything(), as.numeric)) 

# Visualization 
gss_tbl |> 
  ggplot(aes(x = mosthrs)) + 
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  labs(
    title = "Distribution of Maximum Hours Worked Last Week",
    subtitle = "GSS 2016 Data",
    x = "Work Hours",
    y = "Number of Respondents"
  ) +
  theme_minimal()

# Analysis 

## Training and test sets 
set.seed(42) 
index <- createDataPartition(gss_tbl$mosthrs, p = 0.75, list = FALSE) 
train_data <- gss_tbl[index, ] 
test_data <- gss_tbl[-index, ] 

## 10 fold cross-validation 
cv_ten <- trainControl(
  method = "cv", 
  number = 10 
)

## Different models to test 
### Impute (saves time)
medimp <- "medianImpute"

# Sequential exectution (Origional) 
## make sure its running sequentially 
registerDoSEQ()
### OLS model 
tic() # Start timer 
ols_model_seq <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "lm", 
  preProcess =  c("nzv", medimp), 
  trControl = cv_ten, 
  na.action = na.pass 
)
ols_time_seq <- toc() # Saves the time output 
### Elastic net model 
enet_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = seq(0.0001, 0.1, length = 10)
) 

#### Actual model specifications 
tic() #Start timer
en_model_seq <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "glmnet", 
  preProcess =  c("nzv", medimp), 
  tuneGrid = enet_grid, 
  trControl = cv_ten,
  na.action = na.pass
) 
en_time_seq <- toc() #Save output

### Random Forest 
rf_grid <- expand.grid(
  mtry = c(100, 190, 280), 
  splitrule = "variance", 
  min.node.size = 5 
)

#### Actual model specifications
tic() # Start timer
rf_model_seq <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "ranger", 
  preProcess =  c("nzv", medimp), 
  tuneGrid = rf_grid, 
  trControl = cv_ten, 
  na.action = na.pass,
  num.threads = 1
) 
rf_time_seq <- toc()
### XGboost model 
xgb_grid <- expand.grid(
  nrounds = c(50, 100), 
  eta = c(0.01, 0.1), 
  max_depth = c(3, 6), 
  subsample = c(0.8, 1), 
  colsample_bytree = c(0.33, 0.66, 1), 
  gamma = 0, 
  min_child_weight = 1 
)

#### Actual model specifications 
tic() # start timer 
xgb_model_seq <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "xgbTree", 
  preProcess = c("nzv", medimp),
  trControl = cv_ten, 
  tuneGrid = xgb_grid,  
  na.action = na.pass
) 
xgb_time_seq <- toc() #save output 



# Parallel execution



## Cluster (8 cores total for my machine)
local_cluster <- makeCluster(4) # changed to 4 due to errors 
registerDoParallel(local_cluster) # registered for Caret 

### OLS model 
tic() # Start timer 
ols_model_par <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "lm", 
  preProcess =  c("nzv", medimp), 
  trControl = cv_ten, 
  na.action = na.pass 
)
ols_time_par <- toc() # Saves the time output 
### Elastic net model 
enet_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = seq(0.0001, 0.1, length = 10)
) 

#### Actual model specifications 
tic() #Start timer
en_model_par <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "glmnet", 
  preProcess =  c("nzv", medimp), 
  tuneGrid = enet_grid, 
  trControl = cv_ten,
  na.action = na.pass
) 
en_time_par <- toc() #Save output

### Random Forest 
rf_grid <- expand.grid(
  mtry = c(100, 190, 280), 
  splitrule = "variance", 
  min.node.size = 5 
)

#### Actual model specifications
tic() # Start timer
rf_model_par <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "ranger", 
  preProcess =  c("nzv", medimp), 
  tuneGrid = rf_grid, 
  trControl = cv_ten, 
  na.action = na.pass,
  num.threads = 1
) 
rf_time_par <- toc()
### XGboost model 
xgb_grid <- expand.grid(
  nrounds = c(50, 100), 
  eta = c(0.01, 0.1), 
  max_depth = c(3, 6), 
  subsample = c(0.8, 1), 
  colsample_bytree = c(0.33, 0.66, 1), 
  gamma = 0, 
  min_child_weight = 1 
)

#### Actual model specifications 
tic() # start timer 
xgb_model_par <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "xgbTree", 
  preProcess = c("nzv", medimp),
  trControl = cv_ten, 
  tuneGrid = xgb_grid,  
  na.action = na.pass
) 
xgb_time_par <- toc() #save output 

# Stopping cluster to free up respurces 
stopCluster(local_cluster)
# go back to sequential 
registerDoSEQ()

## 10 fold CV estimates (training set)
cv_est <- rbind(
  OLS = getTrainPerf(ols_model_par),
  ElasticNet = getTrainPerf(en_model_par),
  RandomForest = getTrainPerf(rf_model_par),
  XGBoost = getTrainPerf(xgb_model_par)
)

### Print to check 
print(cv_est)

## Predictions that was missing from previous assignment (note: used parallel predictions here, same code, should be the same)
ols_preds <- predict(ols_model_par, newdata = test_data, na.action = na.pass)
en_preds <- predict(en_model_par, newdata = test_data, na.action = na.pass)
rf_preds <- predict(rf_model_par, newdata = test_data, na.action = na.pass)
xgb_preds <- predict(xgb_model_par, newdata = test_data, na.action = na.pass)

## Holdout CV estimates (Test set) dataframe
holdout_est <- as.data.frame(rbind(
  OLS = postResample(pred = ols_preds, obs = test_data$mosthrs),
  ElasticNet = postResample(pred = en_preds, obs = test_data$mosthrs),
  RandomForest = postResample(pred = rf_preds, obs = test_data$mosthrs),
  XGBoost = postResample(pred = xgb_preds, obs = test_data$mosthrs)
))

### Print to check 
print(holdout_est)

# Publication 

## Formatting function 
format_ml_assign <- function(x) {
  x_rounded <- sprintf("%.2f", x) 
  x_no_zero <- str_replace(x_rounded, "^0\\.", ".") 
  x_final <- str_replace(x_no_zero, "^-0\\.", "-.") 
  return(x_final)
} 
## Table 1 tibble 
table1_tbl <- tibble(
  algo = c("OLS regression", "elastic net", "random forest", "eXtreme Gradient Boosting"),
  cv_rsq = format_ml_assign(cv_est$TrainRsquared), 
  ho_rsq = format_ml_assign(holdout_est$Rsquared) 
) 

## Write csv
write_csv(table1_tbl, "../out/table1.csv")
### NOTE: I couldn't figure out exactly what I did wrong to have my holdout r^2 so much higher :( I know its a leakage problem of some sort, but, in understanding that this assignment is about performance between multi-core processing and super computing, I am going to leave it as is.

# table 2
table2_tbl <- tibble(
  algo = c("OLS regression", "elastic net", "random forest", "eXtreme Gradient Boosting"),
  original = c(
    ols_time_seq$toc - ols_time_seq$tic, # each subseque
    en_time_seq$toc - en_time_seq$tic,
    rf_time_seq$toc - rf_time_seq$tic,
    xgb_time_seq$toc - xgb_time_seq$tic
  ),
  parallelized = c(
    ols_time_par$toc - ols_time_par$tic,
    en_time_par$toc - en_time_par$tic,
    rf_time_par$toc - rf_time_par$tic,
    xgb_time_par$toc - xgb_time_par$tic
  )
)

#Write CSV
write_csv(table2_tbl, "../out/table2.csv")

# 11: 
## 1: Based on the outputs of table 2, the all models except OLS benefited 
## greatly from the palatalization. Elastic net, Random forest, and XGBoost all
## had reductions in run time to about 30% of their original time. That being
## said, the overall compute time was most greatly reduced in XGBoost, thus, 
## this is the model which benefitted the most. 

calc_time <- xgb_time_par$tic - ols_time_par$tic
print(calc_time)
## 2: The difference between the fastest and slowest parralalized model was
## 29.057 seconds. 
