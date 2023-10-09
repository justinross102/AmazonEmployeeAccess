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





