---
title: Recommendations - Index
---

This is the home page for the Group 9 project (recommendations) report.

# Project Structure

We first perform [EDA](eda.html) on a random sample of the entire dataset. Then we construct one training set, one meta-training set and one test set (three sets). Due to the sparseness of the original dataset, random sampling does not work. So we do special [data sampling](data-sampling.html). Two baseline regression models, the matrix factorization and some of models of our own choices are trained on the training set. We ensemble those models with stacking and the meta-regressor is trained on the meta-training set. The final model (ensembled) is evaluated on both the test set and the remaining set (data not contained in those three sets). Details of the modeling steps are in the [model page](model.html). Details about how we select our own models and how we select predictors are in [model exploration page](model-exploration.html).

Please click the links for details

# Results

## Baseline Regression

Training $R^2$: 0.11
Test $R^2$: 0.03

## Matrix Factorization

Training $R^2$: 0.9
Test $R^2$: 0.3

## Ensemble Model (Final Model)

Training $R^2$: 0.8
Test $R^2$: 0.5
Test $R^2$ on the remaining set: 0.39

# Other Files
Since methods implementations (e.g. matrix factorization) are long, we put them in seperate Python scripts.
+ Two Baseline Regression: [BaselineRegression.py](src/BaselineRegression.py)
+ Matrix Factorization (ALS): [cfals.py](src/cfals.py)

We read those [references](reference.html) and their results in our project.

# Discussions
## How to Recommend Restaurants for Users

Now we have a model (final model) that predicts the rating given user information and restaurant information. If a user asks for recommendation, we search for all restaurants near him (maybe 1 km), predict what rating the user would give those restaurants if he goes there, and recommend the user those restaurants he will give high ratings.

## Challenges We Faced in the Project

1. Sparseness of the data
2. Too many predictors (for content-based filtering) after appropriate one-hot encoding
3. Training on a large dataset takes long time
