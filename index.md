---
title: Recommendations - Index
---

This is the home page for the Group 9 project (recommendations) report.

# Problem Statement and Motivation
Suppose we have historical data for user's rating on restaurants, and now we want to give user suggestions of restaurants. In order to so, for a pair of user and restaurant (u, r), we want to predict the rating of the user given the restaurant on yelp: rating(u, r). And suggest user the restaurant that we predicted to be highest rated by him or her.

# Introduction and Description of Data

## Original dataset

The original dataset consists of business data, user data, review data, and checkin data: all in form of json files. Business data contains location data, attributes, and categories, while user data contains friend mapping and other metadata of users. Checkin data contains number of checkins for each business. Review data contains full review text data including the user_id that wrote the review and the business_id the review is written for. 

## Data Processing

We first joined user and business data to review data, and then partitioned original data set into a subsample and heldout data. Where subsample contains restaurants that have more than 17 reviews and users that have more than 5 reviews. We did this to solve the sparsity problem of our dataset, where a restaurant/user that have very few reviews are likely to become noise in our baseline models. We will try to apply other models to solve these held out data.
For the subsample we got, we further parition it into training, meta training, and test set.

# Literature Review/Related Work

## Matrix Factorization

[1] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.109&rep=rep1&type=pdf

We used this paper to implement our matrix factorization model. We can construct the residual matrix from baseline models, and factorize the matrix into P and Q, where P contains latent features for users ((U,l) matrix, l is number of latent features) and Q contains latent features for business ((l,B) matrix). And we try to minimize squared loss on observed data, added by a regularization term. The paper also suggested an ALS approach to solve this problem.  

## Item-based knn collaborative filtering

[2] http://files.grouplens.org/papers/www10_sarwar.pdf

We followed this paper to explore item-based knn cf model. According to the paper, we can define similairty of two business by using the correlation of reviews given by different users who have been to both restaruants. And we can predict the rating of a user to a restaurant by a weighted average of the the rating the user has given to other similar restaurants. 


# Modeling Approach and Project Trajectory

We first perform [EDA](eda.html) on a random sample of the entire dataset. Then we construct one training set, one meta-training set and one test set (three sets). Due to the sparseness of the original dataset, random sampling does not work. So we do special [data sampling](data-sampling.html). Two baseline regression models, the matrix factorization and some of models of our own choices are trained on the training set. We ensemble those models with stacking and the meta-regressor is trained on the meta-training set. The final model (ensembled) is evaluated on both the test set and the remaining set (data not contained in those three sets). Details of the modeling steps are in the [model page](model.html). Details about how we select our own models and how we select predictors are in [model exploration page](model-exploration.html).

Please click the links for details


# Results

+ BaseLine Regression (no regularization, use mean)
  + (Training) $R^2$: 0.413268705482, MSE: 0.943767605504
  + (Test) $R^2$: -0.0867599375195, MSE: 1.46768604971
+ BaseLine Regression (with L2 regularization, $\lambda$ chosen by cross-validation)
  + (Training) $R^2$: 0.298990002089, MSE: 1.12758690962
  + (Test) $R^2$: 0.133367873022, MSE: 1.17040004796
+ Matrix Factorization
  + (Training) $R^2$: 0.299639269683, MSE: 1.12654255128
  + (Test) $R^2$: 0.133367983645, MSE: 1.17039989857
+ Our own models (those included in the final model)
  + Ridge
    + (Training) $R^2$: 0.258900935955, MSE: 1.1920708775
    + (Test) $R^2$: 0.226618647701, MSE: 1.1920708775
  + Lasso
    + (Training) $R^2$: 0.242634995273, MSE: 1.21823492915
    + (Test) $R^2$: 0.21596654473, MSE: 1.05884926844
  + Random Forest
    + (Training) $R^2$: 0.782483160748, MSE: 0.349879661193
    + (Test) $R^2$: 0.216967316301, MSE: 1.057497711
+ Final model (ensembled with random forest meta regressor)
  + (Training) $R^2$: 0.2711184242, MSE: 1.16211937945
  + (Test) $R^2$: 0.228895927942, MSE: 1.04138793708
  + (Remaining) $R^2$: 0.448124242262, MSE: 1.20182656604
  + (Whole set except training) $R^2$: 0.380203974883, MSE: 1.13440407154

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
