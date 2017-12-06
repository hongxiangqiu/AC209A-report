---
title: Recommendations - Main Report
---

This is the main report page for the Group 9 project (recommendations). **All important information is included in this single page**. For implementation of all **required** tasks, see [modeling](model.html) page. Other pages contain details of sub-tasks like [EDA](eda.html), [predictor and model selection](model-exploration.html) and [stratified sampling](data-sampling.html). The merged jupyter notebook (containing all code and documentation) can be found [here](project_merged.html)

## Problem Statement and Motivation

Suppose we have historical data for user's rating on restaurants, and now we want to give user suggestions of restaurants. In order to so, for a pair of user and restaurant (u, r), we want to predict the rating of the user given the restaurant on yelp: rating(u, r). And suggest user the restaurant that we predicted to be highest rated by him or her.

## Introduction and Description of Data

One important feature provided by applications like yelp is to recommend restaurants to users. Usually a list of candidates are present, which are usally based on location, and how we should rank these candidates is an open question. Here we formulate this problem under typical machine learning framework: we have labeled inputs (user's reviews on restaurants), and we have some unlabeled input and we want to predict their labels. The main difficulty with this problem is that our data is very sparse. Through our EDA we have seen that many features might have some predictive power, so we are going to try to do some content filtering along with collaborative filtering and baseline models to have an ensemble model to solve this problem.

### Original dataset

The original dataset consists of business data, user data, review data, and checkin data: all in form of json files. Business data contains location data, attributes, and categories, while user data contains friend mapping and other metadata of users. Checkin data contains number of checkins for each business. Review data contains full review text data including the user_id that wrote the review and the business_id the review is written for. 

### Data Processing

We first joined user and business data to review data, and then partitioned original data set into a subsample and heldout data. Where subsample contains restaurants that have no less than 16 reviews and users that have no less than 6 reviews. We did this to solve the sparsity problem of our dataset, where a restaurant/user that have very few reviews are likely to become noise in our baseline models. We will try to apply other models to solve these held out data.
For the subsample we got, we further parition it into training, meta training, and test set.

## Literature Review/Related Work

For a full list of references, please refer to [this page](reference.html). Below are the references for implementing methods not covered in lecture (AC209A requirement).

### Matrix Factorization

[1] Scalable Collaborative Filtering with Jointly Derived Neighborhood Interpolation Weights, [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.109&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.109&rep=rep1&type=pdf)

We used this paper to implement our matrix factorization model. We can construct the residual matrix from baseline models, and factorize the matrix into P and Q, where P contains latent features for users ((U,l) matrix, l is number of latent features) and Q contains latent features for business ((l,B) matrix). And we try to minimize squared loss on observed data, added by a regularization term. The paper also suggested an ALS approach to solve this problem.  

### Item-based knn collaborative filtering

[2] Item-Based Collaborative Filtering Recommendation Algorithms, [http://files.grouplens.org/papers/www10_sarwar.pdf](http://files.grouplens.org/papers/www10_sarwar.pdf)

We followed this paper to explore item-based knn cf model. According to the paper, we can define similairty of two business by using the correlation of reviews given by different users who have been to both restaruants. And we can predict the rating of a user to a restaurant by a weighted average of the the rating the user has given to other similar restaurants. 

### Neural Network

[3] Deep MNIST for Experts, [https://www.tensorflow.org/get_started/mnist/pros](https://www.tensorflow.org/get_started/mnist/pros)

We followed this website tutorial to implement our own fully connected neural network (content filtering based).

## Modeling Approach and Project Trajectory

We first perform [EDA](eda.html) on a random sample of the entire dataset. Then we construct one training set, one meta-training set and one test set (three sets). Due to the sparseness of the original dataset, random sampling does not work. So we do special [data sampling](data-sampling.html)(as described in Data Processing part).

(1) The baseline model is two regression models (based only on `business_id` and `user_id`). (2) Beyond the baseline model, we implemented matrix factorization, item-based KNN, content-filtering based models (predictor selection, neural network, lasso, ridge, random forest, KNN, LinearSVR) and an ensemble model. (3) We've made changes for implementations: after EDA, we find user and business meta data are useful so we integrated content-filtering based models. We expected neural network to have good performance but it's not, so we didn't use it in our final ensemble (maybe there're some problems on the network structure or training).

Two baseline regression models, the matrix factorization and some of models of our own choices are trained on the training set. We ensemble those models with stacking and the meta-regressor is trained on the meta-training set. The final model (ensembled) is evaluated on both the test set and the remaining set (data not contained in those three sets). Details of the modeling steps are in the [model page](model.html). Details about how we select our own models and how we select predictors are in [model exploration page](model-exploration.html).

Please click the links for details


## Results
### Summary

As our project goes, we successfully improved our test $R^2$ step by step. Our final model is an ensemble model built upon the collabrative filtering regularized regression, matrix factorization, content filtering based ridge and lasso regression and random forest regression (content filtering based). The **meta-regressor is random forest**. Our final model yields $R^2=0.38$ **on the whole dataset excluding the training set**. The test $R^2$ is $0.23$. The difference of two $R^2$ is caused by the `remaining` set. Those are data where each user or restaurant are seldomly seen and therefore, their average ratings have very strong correlation with the actual rating (response). In this case, our content-filtering based models will be 'extremely' accurate. However, in real world, if a user or restaurant is so inactive, we won't know their average rating. So we think it's more reasonable to say our model will have $R^2 \approx 0.23$ on predicting future data although the whole set $R^2=0.38$.

### All Results

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
  + (Whole set excluding training) $R^2$: 0.380203974883, MSE: 1.13440407154

As previously discussed. The high $R^2$ in remaining set is not likely to be valid. We think the real $R^2$ for predicting future data will be near $0.23$.

### Model Analysis

Strength of the model: Collabrative filtering is commonly used in recommendation systems and the performance is good. However, in sparse problem like this project, it might not work well. In this case, content filtering based model is better. Our model is based on both collabrative filtering and content filtering and can take advantage of both. So we expect it to be robust.

Weakness of the model: The data of this project is too sparse for the matrix factorization to work well and thus our final model doesn't give much weight to the matrix factorization result. The $R^2$ is not high enough. We suppose average rating for users and restaurants are known and meaningful in our model but this is not always true (for new users or some users rating only a few restaurants).

### Future work

+ Clearly, we should collect more data to make the problem denser so that matrix factorization and all collabrative filtering based models can work better.
+ Neural networks are known to have good performance. We can try to make it work so we'll have better test $R^2$.
+ We should allow missing average rating and if the review count is too small, we consider the average rating is missing.
+ We have a model to predict the rating but not a full system yet. We can implement a system like mentioned in the discussion part.
+ We can do more feature engineering to find more meaningful predictors.

## Discussions
### How to Recommend Restaurants for Users

Now we have a model (final model) that predicts the rating given user information and restaurant information. If a user asks for recommendation, we search for all restaurants near him (maybe 1 km), predict what rating the user would give those restaurants if he goes there, and recommend the user those restaurants he will give high ratings.

### Challenges We Faced in the Project

1. Sparseness of the data
2. Too many predictors (for content-based filtering) after appropriate one-hot encoding
3. Training on a large dataset takes long time

## Files

Please click the links to go to files you are interested in.

Main files
+ [EDA](eda.html)
+ [predictor and model selection](model-exploration.html)
+ [stratified data sampling](data-sampling.html)
+ [modeling (main part)](model.html)
+ [complete jupyter notebook with all code](project_merged.html)
+ [references](references.html)

Since methods implementations (e.g. matrix factorization) are long, we put them in seperate Python scripts.
+ Two Baseline Regression: [BaselineRegression.py](src/BaselineRegression.html)
+ Matrix Factorization (ALS): [cfals.py](src/cfals.html).
+ Matrix Factorization Parallelization [cfals_mp.py](src/cfals_mp.html).
+ Feature Filtering: [filter_features.py](src/filter_features.html)
+ Neural Net: [nn.py](src/nn.py)
