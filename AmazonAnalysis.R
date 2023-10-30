###################################
# Amazon Employee Access Analysis #
###################################

# load libraries ----------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding
library(discrim)
library(naivebayes)
library(doParallel)
library(kknn)
library(kernlab)
library(themis)
library(stacks)

#cl <- makePSOCKcluster(20)
#registerDoParallel(cl)

# load in data ------------------------------------------------------------
train <- vroom("./train.csv") %>% 
  mutate(ACTION = as.factor(ACTION))

test <- vroom("./test.csv") %>% 
  select(-1)

# functions to limit repeated code ----------------------------------------

predict_and_format <- function(workflow, new_data, filename){
  predictions <- workflow %>%
    predict(new_data = new_data,
            type = "prob")
  
  submission <- predictions %>%
    mutate(Id = row_number()) %>% 
    rename("Action" = ".pred_1") %>% 
    select(3,2)
  
  vroom_write(x = submission, file = filename, delim=",")
}

# plots -------------------------------------------------------------------

# # distribution of Action
# train %>%
#   ggplot(mapping = aes(x = factor(ACTION))) +
#   geom_bar() +
#   labs(title = 'Distribution of ACTION',
#        x = 'ACTION',
#        y = 'Count')
# 
# # Distribution of Action for the 15 Managers with most 0s
# ACTION0_managers <- train %>%
#   filter(ACTION == 0) %>%
#   group_by(MGR_ID) %>%
#   summarise(count = n()) %>%
#   arrange(desc(count)) %>%
#   head(15) %>%
#   pull(MGR_ID)
# 
# subset_train <- train %>%
#   filter(MGR_ID %in% ACTION0_managers)
# 
# ggplot(subset_train, aes(x = factor(MGR_ID), fill = factor(ACTION))) +
#   geom_bar(position = 'dodge', stat = 'count') +
#   labs(title = 'Managerial Impact on Resource Approval',
#        x = 'Manager ID',
#        y = 'Count',
#        fill = 'Approval Result') +
#   scale_fill_manual(values = c("0" = "red", "1" = "green"))

# recipes ------------------------------------------------------------------

recipe1 <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>%  # combines categorical values that occur <1% into an "other" value
  step_dummy(all_nominal_predictors())  # dummy variable encoding

balance_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) %>% #Everything numeric for SMOTE
  step_downsample(all_outcomes())

# apply the recipe to the data
prepped_recipe <- prep(balance_recipe)
baked <- bake(prepped_recipe, new_data = train) # should have 112 columns
baked

# logistic regression ------------------------------------------------------

logistic_mod <- logistic_reg() %>%
  set_engine("glm")

logistic_workflow <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(logistic_mod) %>%
  fit(data = train) # Fit the workflow

predict_and_format(logistic_workflow, test, "./logistic_predictions.csv")
# private - 0.69771
# public - 0.69626

# penalized logistic regression -------------------------------------------

penalized_logistic_mod <- logistic_reg(mixture = tune(),
                                       penalty = tune()) %>% #Type of model
  set_engine("glmnet")

# because of the penalty, this regression can handle categories with only a few observations
# so I removed the step_other()
penalized_reg_recipe <- recipe(ACTION ~ ., train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) # target encoding (must be 2-factor)

balance_recipe_2 <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #Everything numeric for SMOTE
  step_downsample(all_outcomes())

penalized_logistic_workflow <- workflow() %>%
  add_recipe(penalized_reg_recipe) %>%
  add_model(penalized_logistic_mod)

## Grid of values to tune over
pen_tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
pen_folds <- vfold_cv(train, v = 5, repeats = 3)

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

predict_and_format(final_pen_wf, test, "./penalized_logistic_predictions.csv")
# normal recipe
# private - 0.85639
# public - 0.86225

# balanced recipe without step_other
# private - 0.8662
# public - 0.87432

# random forests ----------------------------------------------------------

rand_forest_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees = 1000) %>% # or 1000
  set_engine("ranger") %>%
  set_mode("classification")

target_encoding_recipe <- recipe(ACTION ~ ., train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>%  # combines categorical values that occur <1% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) # target encoding (must be 2-factor)

balance_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #Everything numeric for SMOTE
  step_downsample(all_outcomes())

rand_forest_workflow <- workflow() %>%
  add_recipe(target_encoding_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
                                     min_n(),
                                     levels = 5) ## L^2 total tuning possibilities

## Split data for CV
forest_folds <- vfold_cv(train, v = 5, repeats = 2)

## Run the CV
CV_results <- rand_forest_workflow %>%
  tune_grid(resamples = forest_folds,
            grid = rand_forest_tuning_grid,
            metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy

## Find Best Tuning Parameters
forest_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_forest_wf <- rand_forest_workflow %>%
  finalize_workflow(forest_bestTune) %>%
  fit(data = train)

predict_and_format(final_forest_wf, test, "./random_forest_predictions.csv")
# private - 0.86376
# public - 0.87344

# balanced
# private - 0.83542
# public - 0.83599

# naive bayes -------------------------------------------------------------

naive_bayes_model <- naive_Bayes(Laplace = tune(),
                                 smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library


naive_bayes_wf <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(naive_bayes_model)

# cross validation
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                                levels = 5)

## Split data for CV
nb_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- naive_bayes_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_grid,
            metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy

# find nest tuning parameters
nb_bestTune <- CV_results %>%
  select_best("roc_auc")

# finalize workflow
final_nb_wf <- naive_bayes_wf %>%
  finalize_workflow(nb_bestTune) %>%
  fit(data = train)

predict_and_format(final_nb_wf, test, "./naive_bayes_predictions.csv")
# private - 0.76438
# public - 0.75864

# balanced
# private - 0.76264
# public - 0.7604

# k-nearest neighbors -----------------------------------------------------

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_recipe <- recipe(ACTION ~ ., train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>%  # combines categorical values that occur <1% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # target encoding (must be 2-factor)
  step_normalize(all_nominal_predictors())

knn_workflow <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(knn_model)

# cross validation
knn_tuning_grid <- grid_regular(neighbors(),
                               levels = 5)

knn_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- knn_workflow %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tuning_grid,
            metrics = metric_set(roc_auc))

knn_bestTune <- CV_results %>%
  select_best("roc_auc")

# finalize workflow
final_knn_wf <- knn_workflow %>%
  finalize_workflow(knn_bestTune) %>%
  fit(data = train)

predict_and_format(final_knn_wf, test, "./knn_predictions.csv")
# private - 0.8142
# public - 0.80905

# balanced
# private - 0.81341
# public - 0.81456


# principal component dim reduction ---------------------------------------

# pcdr_recipe <- recipe(ACTION ~ ., train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .001) %>%  # combines categorical values that occur <1% into an "other" value
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # target encoding (must be 2-factor)
#   step_normalize(all_nominal_predictors()) %>% 
#   step_pca(all_predictors(), threshold = .9) #Threshold is between 0 and 1
#   
# naive bayes
# private - 0.77292
# public - 0.76914
# difference in public score with this recipe: 0.0105

# knn
# private - 0.75911
# public - 0.7566


# support vector machines -------------------------------------------------

## SVM models
# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# ## Fit or Tune Model HERE
# 
# svm_wf <- workflow() %>% 
#   add_model(svmRadial) %>% 
#   add_recipe(pcdr_recipe)
# 
# # cross validation
# svm_tuning_grid <- grid_regular(cost(),
#                                 degree(),
#                                 levels = 5)
# 
# svm_folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# ## Run the CV
# CV_results <- svm_wf %>%
#   tune_grid(resamples = svm_folds,
#             grid = svm_tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# svm_bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # finalize workflow
# final_svm_wf <- svm_wf %>%
#   finalize_workflow(svm_bestTune) %>%
#   fit(data = train)
# 
# predict_and_format(final_svm_wf, test, "./svmLinear_predictions.csv")

# svmRadial
# private - 0.77253
# public - 0.7704

# Linear got the same predictions as above


# model stacking ----------------------------------------------------------

folds <- vfold_cv(train, v = 5, repeats=2)
untunedModel <- control_stack_grid()

preg_models <- penalized_logistic_workflow %>%
  tune_grid(resamples=folds,
            grid=pen_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

randforest_models <- rand_forest_workflow %>%
  tune_grid(resamples=folds,
            grid=rand_forest_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

# Specify with models to include
my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(randforest_models)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## Use the stacked data to get a prediction

predictions <- stack_mod %>%
  predict(new_data = test,
          type = "prob")

submission <- predictions %>%
  mutate(Id = row_number()) %>% 
  rename("Action" = ".pred_1") %>% 
  select(3,2)

vroom_write(x = submission, file = "stacked_predictions.csv", delim=",")
# private - 0.87883
# public - 0.89001

#stopCluster(cl)



