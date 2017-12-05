---
title: Modeling
notebook: model.ipynb
nav_include:2
---

## Contents
{:.no_toc}
*  
{: toc}





## Load Data

Since the problem is sparse, we performed a stratified random sampling after joining, cleaning and filtering data. See [Data Procession](data-processing.html) page for the details about how we process data.

Then we perform [feature selection](model-exploration.html#predictor-selection) to select important predictors so we can make our content filtering based models work.

Here, we just load our pre-processed data. The training set is used to train all base models. The meta training set is used to train the meta regressor (ensembling). The test set is used to evaluate our models and the remaining set is all data that's not suitable to train our model (either user or restaurant appears too few times). Though we'll evaluate our final model on the remaining set.



```python
df_train = pd.read_csv('subsample_training.csv')
df_meta_train = pd.read_csv('subsample_meta_training.csv')
df_test = pd.read_csv('subsample_test.csv')
df_remaining = pd.read_csv('remaining.csv')

X_train = df_train.drop('stars_review',axis=1)
y_train = df_train.stars_review.values
X_meta_train = df_meta_train.drop('stars_review',axis=1)
y_meta_train = df_meta_train.stars_review.values
X_test = df_test.drop('stars_review',axis=1)
y_test = df_test.stars_review.values
X_remaining = df_remaining.drop('stars_review',axis=1)
y_remaining = df_remaining.stars_review.values
```


## Baseline Model - Regression
### No regularization
Solving the linear regression without regularization is hard since the system is under-determined ($X^TX$ is singular). An alternative is to set parameters to means. Given a dataset, we first calculate the mean of all ratings (denoted as $\hat{\mu}$). For each user $u$, we calculate the difference of the mean of all ratings he gives out and the global mean $\hat{\mu}$ (difference denoted as $\hat{\theta_u}$). For each business $m$, we calculate the difference of the mean of all ratings the business receives and the global mean $\hat{\mu}$ (difference denoted as $\hat{\gamma_m}$). The prediction of rating for user $u$ giving business $m$ would then be $\hat{\mu}+\hat{\theta_u}+\hat{\gamma_m}$.

Since the implementation is not short, we put it [here](BaselineRegression.py) as a seperate Python script.



```python
from src.BaselineRegression import SparseRidgeRegression, MeanRegression

reg = MeanRegression()
reg.fit(X_train.loc[:,['business_id','user_id']].as_matrix(),y_train)
print("Training R^2:", reg.score(X_train.loc[:,['business_id','user_id']].as_matrix(),y_train))
print("Training MSE:", reg.mse(X_train.loc[:,['business_id','user_id']].as_matrix(),y_train))

print("Test R^2:",reg.score(X_test.loc[:,['business_id','user_id']].as_matrix(),y_test))
print("Test MSE:", reg.mse(X_test.loc[:,['business_id','user_id']].as_matrix(),y_test))
```


    Training R^2: 0.413268705482
    Training MSE: 0.943767605504
    Test R^2: -0.0867599375195
    Test MSE: 1.46768604971


### L2 Regularization
This is same as fitting a `ridge` regression on **dummied predictors** `business_id` and `user_id`. The challenge of this regression is solving the sparse matrix. In the complete data, the `X` (dummied predictor) matrix will have around one million columns. Since ridge regression has closed form solution ($\beta=(X^TX+\lambda I)^{-1}X^Ty$, since we don't want to regularize intercept, the top-left element of $I$ should be set to 0), We solve the system with `scipy.spsolve`. Note here we have regularization terms, the system is solvable.

L2 Regularization is implemented in the same class as the previous regression model we put it ([Python script](BaselineRegression.py)). If $\alpha>0$, regularized regression model will be fit.



```python
l2_reg = GridSearchCV(SparseRidgeRegression(),{"alpha":[0.1,1,10]})
l2_reg.fit(X_train.loc[:,['business_id','user_id']].as_matrix(),y_train)
print("best_alpha:", l2_reg.best_params_['alpha'])
l2_reg = SparseRidgeRegression(alpha=l2_reg.best_params_['alpha'])
l2_reg.fit(X_train.loc[:,['business_id','user_id']].as_matrix(),y_train)
print("Training R^2:", l2_reg.score(X_train.loc[:,['business_id','user_id']].as_matrix(),y_train))
print("Training MSE:", l2_reg.mse(X_train.loc[:,['business_id','user_id']].as_matrix(),y_train))

print("Test R^2:", l2_reg.score(X_test.loc[:,['business_id','user_id']].as_matrix(),y_test))
print("Test MSE:", l2_reg.mse(X_test.loc[:,['business_id','user_id']].as_matrix(),y_test))
```


    best_alpha: 10
    Training R^2: 0.298990002089
    Training MSE: 1.12758690962
    Test R^2: 0.133367873022
    Test MSE: 1.17040004796


Significantly improved test $R^2$ (negative to positive).

## Matrix Factorization

We first construct the residual matrix by subtracting predicted values (from L2 reg model) from actual ratings. Then we factorize the residual matrix into P \* Q while minimizing (squared residual + alpha \* (sum of squared elements of P and Q)). To reduce running time we applied massive paralleization.



```python
baseline_predicted_train = l2_reg.predict(X_train.loc[:,['business_id','user_id']].as_matrix())
baseline_predicted_test = l2_reg.predict(X_test.loc[:,['business_id','user_id']].as_matrix())
```




```python
def residual_to_matrix(X_train, residual):
    user_id_encoding = np.unique(X_train.user_id)
    business_id_encoding = np.unique(X_train.business_id)
    reverse_user_id_encoding = {k:v for v,k in enumerate(user_id_encoding)}
    reverse_business_id_encoding = {k:v for v,k in enumerate(business_id_encoding)}
    
    X_residual = scipy.sparse.dok_matrix((len(user_id_encoding),len(business_id_encoding)))
    for i in range(len(X_train)):
        r = residual[i]
        uid = int(X_train.iloc[i]['user_id'])
        bid = int(X_train.iloc[i]['business_id'])
        X_residual[reverse_user_id_encoding[uid],reverse_business_id_encoding[bid]]=r
        
    return X_residual,user_id_encoding,business_id_encoding,reverse_user_id_encoding,reverse_business_id_encoding
```




```python
residual = np.array(y_train) - baseline_predicted_train
X_residual,user_id_encoding,business_id_encoding,reverse_user_id_encoding,reverse_business_id_encoding = residual_to_matrix(X_train, residual)
X_residual
```





    <82946x21826 sparse matrix of type '<class 'numpy.float64'>'
    	with 331784 stored elements in Dictionary Of Keys format>





```python
from src.cfals import CfALS
from src.cfals_mp import CFALSExecutor

h = 15 # number of latent variables
alpha = 25 # regularization constant
cfals = CfALS(X_residual, h=h, alpha=alpha)
```




```python
cfals.initialize()
```




```python
executor = CFALSExecutor(n_workers=20)
executor.initialize(cfals)
```




```python
prev_loss = cfals.cur_loss
print(prev_loss)
for i in range(10):
    cfals.steps(1, executor)
    cur_loss = cfals.cur_loss
    if i < 10 or (i%10==0):
        print(cur_loss)
    if prev_loss is not None and (abs(cur_loss-prev_loss) < 1 or cur_loss > prev_loss):
        break
    prev_loss = cur_loss
```


    None
    444851.71365
    375018.926381
    374199.641155
    374127.347447
    374110.677804
    374104.585515
    374101.727605
    374100.158678
    374099.197665




```python
executor.close()
```




```python
def get_predicted_residuals(cfals, X):
    result = np.zeros(len(X))
    missing_values = 0 
    for i in range(len(X)):
        uid = int(X.iloc[i]['user_id'])
        bid = int(X.iloc[i]['business_id'])
        try:
            uid = reverse_user_id_encoding[uid]
            bid = reverse_business_id_encoding[bid]
        except KeyError:
            #print(uid,bid,"doesn't exist in training data")
            result[i] = 0
            missing_values += 1
            continue
        result[i] = cfals.get_r(uid, bid)
    return result, missing_values
```




```python
train_resid,_ = get_predicted_residuals(cfals, X_train)
predicted_y_train = baseline_predicted_train + train_resid

print("Training R^2:", r2_score(y_train,predicted_y_train))
print("Training MSE:", mean_squared_error(y_train,predicted_y_train))
```


    Training R^2: 0.299639269683
    Training MSE: 1.12654255128




```python
test_resid,missing_values = get_predicted_residuals(cfals, X_test)
predicted_y_test = baseline_predicted_test + test_resid

print("Test R^2:", r2_score(y_test,predicted_y_test))
print("Test MSE:", mean_squared_error(y_test,predicted_y_test))
```


    Test R^2: 0.133367983645
    Test MSE: 1.17039989857


Test $R^2$ improved slightly compared to just regression alone

## Models of Our Own Choice
For models of our own choice, we've tested `item-based KNN` (collaborative filtering), `ridge regression`, `lasso regression`, `fully connected neural network`, `KNN` (content filtering) and `random forest regression` and `Linear SVR`. Please refer to [Model Exploration](model-exploration.html) for details. `fully connected neural network` and `item-based KNN` are not covered in lectures.

Predictors for those content filtering based models are carefully selected by `forward selection`. Details are in [Model Exploration (predictor selection)](model-exploration.html#predictor-selection) page.

We choose `ridge`, `lasso` and `random forest` as our own models since they give the best performance. We load the pre-trained models here from [Model Exploration](model-exploration.html) step.



```python
import pickle
def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
ridge = load_model('ridge')
lasso = load_model('lasso')
rf = load_model('rf')
```


## Ensemble

We want to combine weighted regression, matrix factorization and the model of our own choice. We treat those three models as base learner and fit a meta-regressor (stacking).

Idealy, if we have `N` data points, we should train our base learners on `N-1` data points and generate predictions for the left out data point. The process should be repeated for `N` times to get base learner prediction for all `N` points. However, since the training dataset is really large, using the ideal method will take ages. As a workaround, we split a seperate training set (`meta_train`) from the test set and use `meta_train` to train the meta-regressor. The meta-regressors we will test are `ridge`, `KNN` and `RandomForestRegressor`

First, we use all base predictors to predict results on `meta_training` set, `test` set and `remaining` set. Those are **predictors** of the meta-regressor.



```python
baseline_predicted_meta_train = l2_reg.predict(X_meta_train.loc[:,['business_id','user_id']].as_matrix())
meta_train_resid,_ = get_predicted_residuals(cfals, X_meta_train)
predicted_y_meta_train = meta_train_resid + baseline_predicted_meta_train

baseline_predicted_remaining = l2_reg.predict(X_remaining.loc[:,['business_id','user_id']].as_matrix())
remaining_resid,_ = get_predicted_residuals(cfals, X_remaining)
predicted_y_remaining = remaining_resid + baseline_predicted_remaining

predicted_y_ridge_meta = ridge.predict(X_meta_train.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_lasso_meta = lasso.predict(X_meta_train.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_rf_meta = rf.predict(X_meta_train.drop(['business_id','user_id'],axis=1).as_matrix())

meta_train_df = pd.DataFrame(data={'label':y_meta_train,
                                   "MF":predicted_y_meta_train,
                                   "l2reg":baseline_predicted_meta_train})
meta_test_df = pd.DataFrame(data={'label':y_test,
                                  "MF":predicted_y_test,
                                  "l2reg":baseline_predicted_test})
meta_remaining_df = pd.DataFrame(data={'label':y_remaining,
                                       "MF":predicted_y_remaining,
                                       "l2reg":baseline_predicted_remaining})
```




```python
predicted_y_ridge_meta = ridge.predict(X_meta_train.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_lasso_meta = lasso.predict(X_meta_train.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_rf_meta = rf.predict(X_meta_train.drop(['business_id','user_id'],axis=1).as_matrix())
meta_train_df['ridge'] = predicted_y_ridge_meta
meta_train_df['lasso'] = predicted_y_lasso_meta
meta_train_df['rf'] = predicted_y_rf_meta
```




```python
predicted_y_ridge_test = ridge.predict(X_test.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_lasso_test = lasso.predict(X_test.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_rf_test = rf.predict(X_test.drop(['business_id','user_id'],axis=1).as_matrix())
meta_test_df['ridge'] = predicted_y_ridge_test
meta_test_df['lasso'] = predicted_y_lasso_test
meta_test_df['rf'] = predicted_y_rf_test
```




```python
predicted_y_ridge_remaining = ridge.predict(X_remaining.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_lasso_remaining = lasso.predict(X_remaining.drop(['business_id','user_id'],axis=1).as_matrix())
predicted_y_rf_remaining = rf.predict(X_remaining.drop(['business_id','user_id'],axis=1).as_matrix())
meta_remaining_df['ridge'] = predicted_y_ridge_remaining
meta_remaining_df['lasso'] = predicted_y_lasso_remaining
meta_remaining_df['rf'] = predicted_y_rf_remaining
```




```python
meta_train_df = meta_train_df[['label', 'MF', 'l2reg', 'ridge', 'lasso', 'rf']]
meta_test_df = meta_test_df[['label', 'MF', 'l2reg', 'ridge', 'lasso', 'rf']]
meta_remaining_df = meta_remaining_df[['label', 'MF', 'l2reg', 'ridge', 'lasso', 'rf']]
```




```python
meta_train_df.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>MF</th>
      <th>l2reg</th>
      <th>ridge</th>
      <th>lasso</th>
      <th>rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>3.585668</td>
      <td>3.585668</td>
      <td>3.741885</td>
      <td>3.753655</td>
      <td>3.712874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>4.287673</td>
      <td>4.287673</td>
      <td>5.043061</td>
      <td>4.707099</td>
      <td>4.747548</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>3.409098</td>
      <td>3.409104</td>
      <td>4.034256</td>
      <td>4.030910</td>
      <td>4.025013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3.894862</td>
      <td>3.894862</td>
      <td>2.832780</td>
      <td>3.111310</td>
      <td>3.044585</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4.171702</td>
      <td>4.171702</td>
      <td>4.482754</td>
      <td>4.297518</td>
      <td>4.386246</td>
    </tr>
  </tbody>
</table>
</div>



### Ridge

We further split the `meta_training` set into `meta_train` and `meta_valid` so that we have a validation set to do model selection.



```python
X_meta_train = meta_train_df.drop('label',axis=1).as_matrix()
X_meta_test = meta_test_df.drop('label',axis=1).as_matrix()
y_meta_train = meta_train_df.label.values
y_meta_test = meta_test_df.label.values
X_meta_train, X_meta_valid, y_meta_train, y_meta_valid = train_test_split(
     X_meta_train, y_meta_train, test_size=0.15, random_state=100)

meta_ridge = RidgeCV()
meta_ridge.fit(X_meta_train,y_meta_train)
print("Training R^2:", meta_ridge.score(X_meta_train,y_meta_train))
print("Training MSE:", mean_squared_error(y_meta_train,meta_ridge.predict(X_meta_train)))

print("Validation R^2:", meta_ridge.score(X_meta_valid,y_meta_valid))
print("Validation MSE:", mean_squared_error(y_meta_valid,meta_ridge.predict(X_meta_valid)))
```


    Training R^2: 0.26599607561
    Training MSE: 1.17028638594
    Validation R^2: 0.24860091311
    Validation MSE: 1.19602391727


### KNN



```python
meta_knn = KNeighborsRegressor(n_jobs=-1)
meta_knn.fit(X_meta_train,y_meta_train)
print("Training R^2:", meta_knn.score(X_meta_train,y_meta_train))
print("Training MSE:", mean_squared_error(y_meta_train,meta_knn.predict(X_meta_train)))

print("Validation R^2:", meta_knn.score(X_meta_valid,y_meta_valid))
print("Validation MSE:", mean_squared_error(y_meta_valid,meta_knn.predict(X_meta_valid)))
```


    Training R^2: 0.411283632967
    Training MSE: 0.93864177919
    Validation R^2: 0.109805245424
    Validation MSE: 1.41694904356


KNN has the problem of overfitting!

### Random Forest Regressor



```python
meta_rf = GridSearchCV(RandomForestRegressor(
                n_jobs=-1),{"max_depth":[5,10,30],"n_estimators":[10,50,100,200,300]})
meta_rf.fit(X_meta_train,y_meta_train)
print("Training R^2:", meta_rf.score(X_meta_train,y_meta_train))
print("Training MSE:", mean_squared_error(y_meta_train,meta_rf.predict(X_meta_train)))

print("Validation R^2:", meta_rf.score(X_meta_valid,y_meta_valid))
print("Validation MSE:", mean_squared_error(y_meta_valid,meta_rf.predict(X_meta_valid)))
```


    Training R^2: 0.2711184242
    Training MSE: 1.16211937945
    Validation R^2: 0.250405757565
    Validation MSE: 1.19315109353


## Final Model

We will choose the ensemble model with random forest regressor as the meta-classifier as it yields best validation $R^2$. Let's test the performance of our final model on test set!



```python
print("Test R^2:", meta_rf.score(X_meta_test,y_meta_test))
print("Test MSE:", mean_squared_error(y_meta_test,meta_rf.predict(X_meta_test)))
print("Test Set Size:",len(y_meta_test))
```


    Test R^2: 0.228895927942
    Test MSE: 1.04138793708
    Test Set Size: 889363




```python
X_meta_remaining = meta_remaining_df.drop('label',axis=1).as_matrix()
y_meta_remaining = meta_remaining_df.label.values
print("Remaining Set R^2:", meta_rf.score(X_meta_remaining,y_meta_remaining))
print("Remaining Set MSE:", mean_squared_error(y_meta_remaining,meta_rf.predict(X_meta_remaining)))
print("Remaining Set Size:", len(y_meta_remaining))
```


    Remaining Set R^2: 0.448124242262
    Remaining Set MSE: 1.20182656604
    Remaining Set Size: 1226966


Our meta-regressor successfully improves the test $R^2$ score on the test set. We know collaborative filtering methods (like matrix factorization) does not work on the `remaining` set. But since we also have content filtering based methods, we can predict pretty well on the `remaining` set. Since the `remaining` set has size `1226966`, the high $R^2$ is not likely by random.

We try to predict on all data except training.



```python
X_not_train = np.concatenate((X_meta_remaining,X_meta_test),axis=0)
y_not_train = np.concatenate((y_meta_remaining,y_meta_test))
print("Entire sets except training R^2:", meta_rf.score(X_not_train,y_not_train))
print("Entire sets except training MSE:", mean_squared_error(y_not_train,meta_rf.predict(X_not_train)))
print("Entire sets except training - Size:", len(y_not_train))
```


    Entire sets except training R^2: 0.380203974883
    Entire sets except training MSE: 1.13440407154
    Entire sets except training - Size: 2116329


## How to Recommend Restaurants for Users

Now we have a model (final model) that predicts the rating given user information and restaurant information. If a user asks for recommendation, we search for all restaurants near him (maybe 1 km), predict what rating the user would give those restaurants if he goes there, and recommend the user those restaurants he will give high ratings.
