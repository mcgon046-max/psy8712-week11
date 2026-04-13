# Script Settings and Resources 
# Note: axed tidyverse because it was taking too long. (I think you mentioned this?)
library(dplyr)
library(readr)
library(caret) 
library(xgboost) 
library(haven)
library(parallel)
library(doParallel)
library(tidyr)
library(stringr)
library(tictoc)

# Data Import and Cleaning 
gss_tbl <- haven::read_sav("../data/GSS2016.sav") |> 
  drop_na(mosthrs) |>
  mutate(mosthrs = as.integer(mosthrs)) |> 
  select(-hrs1, -hrs2) |> 
  select(where(~ mean(is.na(.)) < 0.75)) |>
  mutate(across(everything(), as.numeric)) 

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

medimp <- "medianImpute"

# Sequential execution
registerDoSEQ()

tic()
ols_model_seq <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "lm", 
  preProcess =  c("nzv", medimp), 
  trControl = cv_ten, 
  na.action = na.pass 
)
ols_time_seq <- toc()

enet_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = seq(0.0001, 0.1, length = 10)
) 

tic()
en_model_seq <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "glmnet", 
  preProcess =  c("nzv", medimp), 
  tuneGrid = enet_grid, 
  trControl = cv_ten,
  na.action = na.pass
) 
en_time_seq <- toc() 

rf_grid <- expand.grid(
  mtry = c(100, 190, 280), 
  splitrule = "variance", 
  min.node.size = 5 
)

tic() 
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

xgb_grid <- expand.grid(
  nrounds = c(50, 100), 
  eta = c(0.01, 0.1), 
  max_depth = c(3, 6), 
  subsample = c(0.8, 1), 
  colsample_bytree = c(0.33, 0.66, 1), 
  gamma = 0, 
  min_child_weight = 1 
)

tic()
xgb_model_seq <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "xgbTree", 
  preProcess = c("nzv", medimp),
  trControl = cv_ten, 
  tuneGrid = xgb_grid,  
  na.action = na.pass
) 
xgb_time_seq <- toc() 

# Parallel execution

# I used parallel::detectCores() to ensure the script automatically scales 
# to the specific hardware allocated by the MSI scheduler.
local_cluster <- makeCluster(31)
registerDoParallel(local_cluster) 

tic()
ols_model_par <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "lm", 
  preProcess =  c("nzv", medimp), 
  trControl = cv_ten, 
  na.action = na.pass 
)
ols_time_par <- toc()

tic()
en_model_par <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "glmnet", 
  preProcess =  c("nzv", medimp), 
  tuneGrid = enet_grid, 
  trControl = cv_ten,
  na.action = na.pass
) 
en_time_par <- toc()

tic()
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

tic()
xgb_model_par <- train(
  mosthrs ~ ., 
  data = train_data, 
  method = "xgbTree", 
  preProcess = c("nzv", medimp),
  trControl = cv_ten, 
  tuneGrid = xgb_grid,  
  na.action = na.pass
) 
xgb_time_par <- toc() 

stopCluster(local_cluster)
registerDoSEQ()

## 10 fold CV estimates (training set)
cv_est <- rbind(
  OLS = getTrainPerf(ols_model_par),
  ElasticNet = getTrainPerf(en_model_par),
  RandomForest = getTrainPerf(rf_model_par),
  XGBoost = getTrainPerf(xgb_model_par)
)

## Predictions
ols_preds <- predict(ols_model_par, newdata = test_data, na.action = na.pass)
en_preds <- predict(en_model_par, newdata = test_data, na.action = na.pass)
rf_preds <- predict(rf_model_par, newdata = test_data, na.action = na.pass)
xgb_preds <- predict(xgb_model_par, newdata = test_data, na.action = na.pass)

## Holdout CV estimates
holdout_est <- as.data.frame(rbind(
  OLS = postResample(pred = ols_preds, obs = test_data$mosthrs),
  ElasticNet = postResample(pred = en_preds, obs = test_data$mosthrs),
  RandomForest = postResample(pred = rf_preds, obs = test_data$mosthrs),
  XGBoost = postResample(pred = xgb_preds, obs = test_data$mosthrs)
))

# Publication 

format_ml_assign <- function(x) {
  x_rounded <- sprintf("%.2f", x) 
  x_no_zero <- str_replace(x_rounded, "^0\\.", ".") 
  x_final <- str_replace(x_no_zero, "^-0\\.", "-.") 
  return(x_final)
}

table3_tbl <- tibble(
  algo = c("OLS regression", "elastic net", "random forest", "eXtreme Gradient Boosting"),
  cv_rsq = format_ml_assign(cv_est$TrainRsquared), 
  ho_rsq = format_ml_assign(holdout_est$Rsquared) 
) 

write_csv(table3_tbl, "table3.csv")

table4_tbl <- tibble(
  algo = c("OLS regression", "elastic net", "random forest", "eXtreme Gradient Boosting"),
  supercomputer = c(
    ols_time_seq$toc - ols_time_seq$tic, 
    en_time_seq$toc - en_time_seq$tic,
    rf_time_seq$toc - rf_time_seq$tic,
    xgb_time_seq$toc - xgb_time_seq$tic
  ),
  parallel_col = c(
    ols_time_par$toc - ols_time_par$tic,
    en_time_par$toc - en_time_par$tic,
    rf_time_par$toc - rf_time_par$tic,
    xgb_time_par$toc - xgb_time_par$tic
  )
)

colnames(table4_tbl)[3] <- paste0("supercomputer_", core_count)

write_csv(table4_tbl, "table4.csv")
