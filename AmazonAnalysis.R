###################################
# Amazon Employee Access Analysis #
###################################

# load libraries ----------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding
library(ggmosaic)

# load in data ------------------------------------------------------------
train <- vroom("./train.csv") %>% 
  mutate(ACTION = as.factor(ACTION))

test <- vroom("./test.csv") %>% 
  select(-1)

# plots -------------------------------------------------------------------

# distribution of Action
train %>%
  ggplot(mapping = aes(x = factor(ACTION))) + 
  geom_bar() +
  labs(title = 'Distribution of ACTION',
       x = 'ACTION',
       y = 'Count')

# Distribution of Action for the 15 Managers with most 0s
ACTION0_managers <- train %>%
  filter(ACTION == 0) %>%
  group_by(MGR_ID) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(15) %>%
  pull(MGR_ID)

subset_train <- train %>%
  filter(MGR_ID %in% ACTION0_managers)

ggplot(subset_df, aes(x = factor(MGR_ID), fill = factor(ACTION))) +
  geom_bar(position = 'dodge', stat = 'count') +
  labs(title = 'Managerial Impact on Resource Approval',
       x = 'Manager ID',
       y = 'Count',
       fill = 'Approval Result') +
  scale_fill_manual(values = c("0" = "red", "1" = "green")) 

# recipe ------------------------------------------------------------------

my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>%  # combines categorical values that occur <1% into an "other" value
  step_dummy(all_nominal_predictors())  # dummy variable encoding
  
# apply the recipe to the data
prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = train) # should have 112 columns
baked

# logistic regression ------------------------------------------------------

logistic_mod <- logistic_reg() %>% 
  set_engine("glm")

logistic_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logistic_mod) %>%
  fit(data = train) # Fit the workflow

amazon_predictions <- predict(logistic_workflow,
                              new_data = test,
                              type = "prob") # "class" or "prob"

kaggle_submission <- amazon_predictions %>%
  mutate(Id = row_number()) %>% 
  rename("Action" = ".pred_1") %>% 
  select(3,2)

# write predictions to csv
vroom_write(x=kaggle_submission, file="./amazon_predictions.csv", delim=",")
# private - 0.70429
# public - 0.69688

# penalized logistic regression -------------------------------------------

penalized_logistic_mod <- logistic_reg(mixture = tune(),
                                       penalty = tune()) %>% #Type of model
  set_engine("glmnet")

target_encoding_recipe <- recipe(ACTION ~ ., train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>%  # combines categorical values that occur <1% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) # target encoding (must be 2-factor)

penalized_logistic_workflow <- workflow() %>%
  add_recipe(target_encoding_recipe) %>%
  add_model(penalized_logistic_mod)

## Grid of values to tune over
pen_tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
pen_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- penalized_logistic_workflow %>%
  tune_grid(resamples = pen_folds,
            grid = pen_tuning_grid,
            metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy

## Find Best Tuning Parameters
pen_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_pen_wf <- penalized_logistic_workflow %>%
  finalize_workflow(pen_bestTune) %>%
  fit(data = train)

## Predict
penalized_logistic_predictions <- final_pen_wf %>%
  predict(new_data = test,
          type = "prob")

pen_log_submission <- penalized_logistic_predictions %>%
  mutate(Id = row_number()) %>% 
  rename("Action" = ".pred_1") %>% 
  select(3,2)

# write to csv
vroom_write(x=pen_log_submission, file="./penalized_logistic_predictions.csv", delim=",")
# private - 0.79076
# public - 0.7832




