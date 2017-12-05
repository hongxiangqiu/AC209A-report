---
title: Model Exploration
notebook: model-exploration.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}





In this part, we want to select some models (serve as `something else of your choice`) to be integrated into the main ensemble model. In addition, we want to select good predictors we can use for content filtering based models {based on EDA, there will be too many predictors (after appropriate one-hot encoding) if we use all of them}.


To be fair, all content filtering based models will use those predictors selected in this section. The selection is conducted on the training set. Then the selected columns will be used to filter meta-training set and test set.

Unfortunately, the dataset has too many predictors to perform forward selection efficiently. As a workaround, we use recursive feature elimination (RFE) which select features by recursively considering smaller and smaller sets of features. (Since interpretation for the recommendation system is important, we don't do PCA)



```python
df_all = pd.read_csv('subsample_1.csv')
```




```python
df_all.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool_review</th>
      <th>date_review</th>
      <th>funny_review</th>
      <th>review_id</th>
      <th>stars_review</th>
      <th>useful_review</th>
      <th>user_id</th>
      <th>attributes_business</th>
      <th>categories_business</th>
      <th>...</th>
      <th>compliment_writer_user</th>
      <th>cool_user_user</th>
      <th>elite_user</th>
      <th>fans_user</th>
      <th>friends_user</th>
      <th>funny_user</th>
      <th>name_user</th>
      <th>review_count_user</th>
      <th>useful_user</th>
      <th>yelping_since_user</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30546</td>
      <td>2</td>
      <td>2013-03-11</td>
      <td>0</td>
      <td>DwXbGQx5oyYc2hnbUc1hQg</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>{'Alcohol': 'beer_and_wine', 'HasTV': True, 'N...</td>
      <td>['Sandwiches', 'Food', 'American (New)', 'Amer...</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>[2013, 2010, 2011, 2012]</td>
      <td>15</td>
      <td>['yZ3Z6SIbbp9DZWxAqHHIyg', 'LbeHQ0frxP6sJew2fK...</td>
      <td>22</td>
      <td>Monera</td>
      <td>245</td>
      <td>67</td>
      <td>2007-06-04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6500</td>
      <td>0</td>
      <td>2010-11-05</td>
      <td>1</td>
      <td>nHYLl06G_Yt8dcRpzCJFiQ</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>{'GoodForMeal': {'dessert': False, 'latenight'...</td>
      <td>['French', 'Italian', 'Restaurants']</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>[2013, 2010, 2011, 2012]</td>
      <td>15</td>
      <td>['yZ3Z6SIbbp9DZWxAqHHIyg', 'LbeHQ0frxP6sJew2fK...</td>
      <td>22</td>
      <td>Monera</td>
      <td>245</td>
      <td>67</td>
      <td>2007-06-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33528</td>
      <td>0</td>
      <td>2011-01-07</td>
      <td>4</td>
      <td>HT7owxeVvpry33QQuzzgiw</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>{'RestaurantsTableService': True, 'GoodForMeal...</td>
      <td>['Restaurants', 'Sushi Bars', 'Japanese']</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>[2013, 2010, 2011, 2012]</td>
      <td>15</td>
      <td>['yZ3Z6SIbbp9DZWxAqHHIyg', 'LbeHQ0frxP6sJew2fK...</td>
      <td>22</td>
      <td>Monera</td>
      <td>245</td>
      <td>67</td>
      <td>2007-06-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39390</td>
      <td>0</td>
      <td>2010-10-16</td>
      <td>0</td>
      <td>1ikB-TEgwg2gigixDEDSuA</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>{'RestaurantsTableService': True, 'GoodForMeal...</td>
      <td>['Thai', 'Restaurants']</td>
      <td>...</td>
      <td>9</td>
      <td>9</td>
      <td>[2013, 2010, 2011, 2012]</td>
      <td>15</td>
      <td>['yZ3Z6SIbbp9DZWxAqHHIyg', 'LbeHQ0frxP6sJew2fK...</td>
      <td>22</td>
      <td>Monera</td>
      <td>245</td>
      <td>67</td>
      <td>2007-06-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4506</td>
      <td>0</td>
      <td>2017-06-17</td>
      <td>0</td>
      <td>PiYJ0Fa6EVnFd3AAQOwvSw</td>
      <td>5</td>
      <td>0</td>
      <td>9</td>
      <td>{'RestaurantsTableService': True, 'GoodForMeal...</td>
      <td>['Restaurants', 'American (New)']</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>['VMG0qo4X0QOLmFLJ-gkOcw', 'QQlamW29Hrg4oAJYxX...</td>
      <td>1</td>
      <td>Kristin</td>
      <td>28</td>
      <td>7</td>
      <td>2016-07-28</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>





```python
df_all['rating'] = df_all['stars_review']

df_all = df_all.drop([x for x in list(df_train.columns) if x.find('_review')>-1]+
              ['review_id','user_id','business_id'],axis=1)
```




```python
_, df_sample = train_test_split(df_all, test_size=0.15, random_state=0)
df_sample=df_sample.reset_index(drop=True)
```


Select 30 features from `categories_business`



```python
df_cate = df_sample[["categories_business", "rating"]].copy()
cate_set = set()
for _, row in df_cate.iterrows():
    c_string, r_string = row['categories_business'], row['rating']
    c_string = c_string[1:-1]
    cs = c_string.split(",")
    for c in cs:
        cate_set.add(c)
        
for c in cate_set:
    #This line is the source of the warning. Any better idea to add new empty columns?
    df_cate[c]= 0
    
for i in range(df_cate.shape[0]):
    c_string, r_string = df_cate['categories_business'][i], df_cate['rating'][i]
    c_string = c_string[1:-1]
    cs = c_string.split(",")
    for c in cs:
        df_cate.at[i, c] = 1

del df_cate["categories_business"]

X_train = df_cate.iloc[:, 1:]
y_train = df_cate.iloc[:, 1]

estimator = Ridge()
#If you want automatic number, use RFECV instead
selector = RFE(estimator, 30, step=1)
selector = selector.fit(X_train, y_train)
best_features = selector.support_
```




```python
best_columns = []
for i in range(len(best_features)):
    if best_features[i]:
        best_columns.append(eval(df_cate.columns[i+1]))
print(best_columns)
```


    ['Dinner Theater', 'Buffets', 'Cafes', 'Asian Fusion', 'Vegetarian', 'Mexican', 'Mediterranean', 'French', 'Burgers', 'Breakfast & Brunch', 'Pizza', 'Fast Food', 'Bars', 'American (Traditional)', 'Japanese', 'Sandwiches', 'Steakhouses', 'Thai', 'Sushi Bars', 'Food', 'Italian', 'Seafood', 'Barbeque', 'Nightlife', 'Salad', 'Restaurants', 'American (New)', 'Chinese', 'Vietnamese', 'Coffee & Tea']


Select 30 features from `attributes_business`



```python
df_attr = df_sample[["attributes_business", "rating"]].copy()
#deal with first layer attribute
attr_list  = []
for k, value in eval(df_attr[:1]['attributes_business'].values[0]).items():
        if type(value) != dict:
            attr_list.append(k)

for a in attr_list:
    df_attr[a]= None

for i in range(df_attr.shape[0]):
        a_dict,r = eval(df_attr['attributes_business'][i]), df_attr['rating'][i]
        for a in attr_list:
            try:
                v = a_dict[a]
            except:
                pass
            df_attr.at[i, a] = v

#deal with second layer attribute
attr_list_2  = []
for key, value in eval(df_attr[:1]['attributes_business'].values[0]).items():
        if type(value) == dict:
            for k, v in value.items():
                attr_list_2.append(key+"#"+k)
for a in attr_list_2:
    df_attr[a]= None
    
for i in range(df_attr.shape[0]):
    a_dict,r = eval(df_attr['attributes_business'][i]), df_attr['rating'][i]
    for a in attr_list_2:
        k1, k2 = a.split("#")
        try:
            inner = a_dict[k1]
        except:
            pass
        try:
            v = inner[k2]
        except:
            pass
        df_attr.at[i, a] = v

del df_attr["attributes_business"]
df_attr = pd.get_dummies(df_attr, drop_first=False)

X_train = df_attr.iloc[:, 1:]
y_train = df_attr.iloc[:, 1]
estimator = Ridge()
selector = RFE(estimator, 30, step=1)
selector = selector.fit(X_train, y_train)
best_features = selector.support_
```




```python
best_columns = []
for i in range(len(best_features)):
    if best_features[i]:
        best_columns.append(df_attr.columns[i+1])
print(best_columns)
```


    ['Alcohol_False', 'Alcohol_True', 'Alcohol_2', 'Alcohol_beer_and_wine', 'Alcohol_full_bar', 'Alcohol_no', 'Alcohol_none', 'Alcohol_outdoor', 'Alcohol_yes', 'HasTV_False', 'HasTV_True', 'HasTV_2', 'HasTV_beer_and_wine', 'HasTV_full_bar', 'HasTV_none', 'HasTV_yes', 'NoiseLevel_False', 'NoiseLevel_True', 'NoiseLevel_2', 'NoiseLevel_average', 'NoiseLevel_beer_and_wine', 'NoiseLevel_full_bar', 'NoiseLevel_loud', 'NoiseLevel_none', 'NoiseLevel_quiet', 'NoiseLevel_very_loud', 'NoiseLevel_yes', 'RestaurantsAttire_False', 'RestaurantsAttire_2', 'RestaurantsAttire_yes']


Seems `Alcohol`, `NoiseLevel` and `RestaurantAttiere` are really important. Plus we know from EDA that `WiFi` is important.

Select from numerical features of the dataset:



```python
df_numeric = df_sample._get_numeric_data().copy()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_cate = df_sample.select_dtypes(exclude=numerics)
#process categorical columns
#pre-drop some non-meaningful features
df_categorical = df_cate[['city_business', 'yelping_since_user']].copy()
#engineer year feature, since date varies too much
for i in range(df_categorical.shape[0]):
    year = df_categorical.at[i, "yelping_since_user"][:4]
    df_categorical.at[i, "yelping_since_user"] = year
df_cate_final = pd.get_dummies(df_categorical, columns=['city_business', 'yelping_since_user'])

#train test split
df_final = df_cate_final.merge(df_numeric, left_index=True, right_index=True)
y_train = df_final['rating']
X_train = df_final.drop(['rating'], axis = 1)

#feature selection
estimator = Ridge()
selector = RFE(estimator, 40, step=1)
selector = selector.fit(X_train, y_train)
best_features = selector.support_ 
```




```python
best_columns = []
for i in range(len(best_features)):
    if best_features[i]:
        best_columns.append(X_train.columns[i])
print(best_columns)
```


    ['city_business_Bolton', 'city_business_Bradford West Gwillimbury', 'city_business_Brunswick', 'city_business_Burton', 'city_business_Clairton', 'city_business_Don Mills', 'city_business_Dorval', 'city_business_Elyria', 'city_business_Frazer', 'city_business_Gerlingen', 'city_business_Houston', 'city_business_LasVegas', 'city_business_Lasalle', 'city_business_Laveen Village', 'city_business_MESA', 'city_business_Mayfield', 'city_business_McKnight', 'city_business_Mentor-on-the-Lake', 'city_business_N. Las Vegas', 'city_business_N. Olmsted', 'city_business_Oakdale', 'city_business_Olmsted Falls', 'city_business_Olmsted Township', 'city_business_Presto', 'city_business_Rantoul', 'city_business_Rexdale', 'city_business_Saint Joseph', 'city_business_Sharpsburg', 'city_business_South Park', 'city_business_St-Jerome', 'city_business_Stanley', 'city_business_Sun City West', 'city_business_University Heights', 'city_business_Upper Saint Clair', 'city_business_Valley City', 'city_business_WICKLIFFE', 'city_business_Wickliffe', 'yelping_since_user_2004', 'stars_business', 'average_stars_user']


We checked all above cities and seems all of them have only a few data points. So city may not be a good predictor. We redo this selection and remove `city`



```python
df_numeric = df_sample._get_numeric_data().copy()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_cate = df_sample.select_dtypes(exclude=numerics)
#process categorical columns
#pre-drop some non-meaningful features
df_categorical = df_cate[['yelping_since_user']].copy()
#engineer year feature, since date varies too much
for i in range(df_categorical.shape[0]):
    year = df_categorical.at[i, "yelping_since_user"][:4]
    df_categorical.at[i, "yelping_since_user"] = year
df_cate_final = pd.get_dummies(df_categorical, columns=['yelping_since_user'])

#train test split
df_final = df_cate_final.merge(df_numeric, left_index=True, right_index=True)
y_train = df_final['rating']
X_train = df_final.drop(['rating'], axis = 1)

#feature selection
estimator = Ridge()
selector = RFE(estimator, 39, step=1)
selector = selector.fit(X_train, y_train)
best_features = selector.support_ 
```




```python
best_columns = []
for i in range(len(best_features)):
    if best_features[i]:
        best_columns.append(X_train.columns[i])
print(best_columns)
```


    ['yelping_since_user_2004', 'yelping_since_user_2005', 'yelping_since_user_2006', 'yelping_since_user_2007', 'yelping_since_user_2008', 'yelping_since_user_2009', 'yelping_since_user_2010', 'yelping_since_user_2011', 'yelping_since_user_2012', 'yelping_since_user_2013', 'yelping_since_user_2014', 'yelping_since_user_2015', 'yelping_since_user_2016', 'yelping_since_user_2017', 'latitude_business', 'longitude_business', 'review_count_business', 'stars_business', 'average_stars_user', 'compliment_cool_user', 'compliment_cute_user', 'compliment_funny_user', 'compliment_hot_user', 'compliment_list_user', 'compliment_more_user', 'compliment_note_user', 'compliment_photos_user', 'compliment_plain_user', 'compliment_profile_user', 'compliment_writer_user', 'cool_user_user', 'fans_user', 'funny_user', 'review_count_user', 'useful_user']


Thus, our final predictors (for content based filtering) will be: 

'yelp_since_user' (2004-2017), 'is_open_business', 'latitude_business', 'longitude_business', 'review_count_business', 'stars_business', 'average_stars_user', 'compliment_cool_user', 'compliment_cute_user', 'compliment_funny_user', 'compliment_hot_user', 'compliment_list_user', 'compliment_more_user', 'compliment_note_user', 'compliment_photos_user', 'compliment_plain_user', 'compliment_profile_user', 'compliment_writer_user', 'cool_user_user', 'fans_user', 'funny_user', 'review_count_user', 'useful_user'

Categorical: 'Dinner Theater', 'Buffets', 'Cafes', 'Asian Fusion', 'Vegetarian', 'Mexican', 'Mediterranean', 'French', 'Burgers', 'Breakfast & Brunch', 'Pizza', 'Fast Food', 'Bars', 'American (Traditional)', 'Japanese', 'Sandwiches', 'Steakhouses', 'Thai', 'Sushi Bars', 'Food', 'Italian', 'Seafood', 'Barbeque', 'Nightlife', 'Salad', 'Restaurants', 'American (New)', 'Chinese', 'Vietnamese', 'Coffee & Tea'

Attributes: Alcohol, HasTV, NoiseLevel, RestaurantAttiere and WiFi

Next, we write a function to extract those features from a given dataframe and make new dataset with only selected predictors (and ids for collaborative filtering). The implementation is in [filter_features.py](src/filter_features.py). We performed the feature extraction on the whole dataset. All our content-filtering based models will be trained and evaluated on the new dataset.



```python
df_train = pd.read_csv('subsample_training.csv')

df_test = pd.read_csv('subsample_test.csv')

X_train = df_train.drop('stars_review',axis=1)
y_train = df_train.stars_review
X_test = df_test.drop('stars_review',axis=1)
y_test = df_test.stars_review
```




```python
X_train.head()
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
      <th>business_id</th>
      <th>user_id</th>
      <th>latitude_business</th>
      <th>longitude_business</th>
      <th>review_count_business</th>
      <th>stars_business</th>
      <th>average_stars_user</th>
      <th>compliment_cool_user</th>
      <th>compliment_cute_user</th>
      <th>compliment_funny_user</th>
      <th>...</th>
      <th>noiseLevel_quiet</th>
      <th>noiseLevel_loud</th>
      <th>noiseLevel_veryloud</th>
      <th>RestaurantsAttire</th>
      <th>RestaurantsAttire_casual</th>
      <th>RestaurantsAttire_dressy</th>
      <th>RestaurantsAttire_formal</th>
      <th>WiFi_no</th>
      <th>WiFi_free</th>
      <th>WiFi_paid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29615</td>
      <td>0</td>
      <td>36.100877</td>
      <td>-115.314710</td>
      <td>87</td>
      <td>4.0</td>
      <td>3.97</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9333</td>
      <td>0</td>
      <td>36.144514</td>
      <td>-115.277522</td>
      <td>129</td>
      <td>4.0</td>
      <td>3.97</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44237</td>
      <td>0</td>
      <td>36.159438</td>
      <td>-115.316629</td>
      <td>571</td>
      <td>4.0</td>
      <td>3.97</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50675</td>
      <td>0</td>
      <td>36.198061</td>
      <td>-115.282510</td>
      <td>464</td>
      <td>4.5</td>
      <td>3.97</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36778</td>
      <td>9</td>
      <td>33.581493</td>
      <td>-111.923509</td>
      <td>85</td>
      <td>4.5</td>
      <td>4.60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




## First, construct rating matrix



```python
def df_to_matrix(X_train, y):
    user_id_encoding = np.unique(X_train.user_id)
    business_id_encoding = np.unique(X_train.business_id)
    reverse_user_id_encoding = {k:v for v,k in enumerate(user_id_encoding)}
    reverse_business_id_encoding = {k:v for v,k in enumerate(business_id_encoding)}
    
    X_result = scipy.sparse.dok_matrix((len(user_id_encoding),len(business_id_encoding)))
    for i in range(len(X_train)):
        r = y.iloc[i]
        uid = int(X_train.iloc[i]['user_id'])
        bid = int(X_train.iloc[i]['business_id'])
        X_result[reverse_user_id_encoding[uid],reverse_business_id_encoding[bid]]=r
        
    return X_result,user_id_encoding,business_id_encoding,reverse_user_id_encoding,reverse_business_id_encoding

X_train_matrix,user_id_encoding,business_id_encoding,reverse_user_id_encoding,\
reverse_business_id_encoding = df_to_matrix(X_train, y_train)
```


## Then construct similarity matrix



```python
M = X_train_matrix.shape[1]
item_users = [set() for _ in range(M)]
for (i,j) in X_train_matrix.keys():
    item_users[j].add(i)
no_joint = 0
for i in range(M):
    for j in range(i+1, M):
        u1 = item_users[i]
        u2 = item_users[j]
        joint = u1 & u2
        # we are supposed to calculate similarity here, but we encountered some issues...
        # more to follow
        if len(joint) == 0:
            no_joint += 1
```




```python
print(no_joint/(M*(M-1)/2)*100, "% of our similairty matrix is undefined!") 
```


    99.80996927799993 % of our similairty matrix is undefined!


While constructing our similarity matrix, we observed that we have so many restaurant pairs that don't share common users. sim(a,b) will be undefined according to [1]. Therefore item-based knn won't work because of the sparsity we have in this dataset.
[1] http://cs229.stanford.edu/proj2008/Wen-RecommendationSystemBasedOnCollaborativeFiltering.pdf

From now on, we try some content filtering based models. So, we should drop `user_id` and `business_id`.



```python
X_train = X_train.drop(['user_id','business_id'],axis=1)
X_test = X_test.drop(['user_id','business_id'],axis=1)
```





```python
ridge = RidgeCV()
ridge.fit(X_train,y_train)
print("Ridge:")
print("Training R^2:", ridge.score(X_train,y_train))
print("Training MSE:",mean_squared_error(y_train,ridge.predict(X_train)))
print("Test R^2:", ridge.score(X_test,y_test))
print("Training MSE:",mean_squared_error(y_train,ridge.predict(X_train)))
```


    Ridge:
    Training R^2: 0.258900935955
    Training MSE: 1.1920708775
    Test R^2: 0.226618647701
    Training MSE: 1.1920708775





```python
lasso = LassoCV(n_jobs=-1)
lasso.fit(X_train,y_train)
print("Lasso:")
print("Training R^2:", lasso.score(X_train,y_train))
print("Training MSE:",mean_squared_error(y_train,lasso.predict(X_train)))
print("Test R^2:", lasso.score(X_test,y_test))
print("Test MSE:",mean_squared_error(y_test,lasso.predict(X_test)))
```


    Lasso:
    Training R^2: 0.242634995273
    Training MSE: 1.21823492915
    Test R^2: 0.21596654473
    Test MSE: 1.05884926844



We made the training and the test of the neural network as a separate Python script [here](src/nn.py). We used a three layer fully-connected network. We used some code in the Tensorflow MNIST tutorial [2] (https://www.tensorflow.org/get_started/mnist/pros) to implement our network.



```python
with open('nn.log', 'r') as log:
    print(log.read())
```


    Start training...
    Training loss: 2.23115
    training r2: -0.387087
    Training loss: 1.59364
    training r2: 0.009248
    Training loss: 1.52451
    training r2: 0.052229
    Training loss: 1.50292
    training r2: 0.065650
    Training loss: 1.4922
    training r2: 0.072312
    Training loss: 1.48826
    training r2: 0.074762
    Training loss: 1.48622
    training r2: 0.076029
    Training loss: 1.48413
    training r2: 0.077331
    Training loss: 1.48324
    training r2: 0.077884
    Training loss: 1.48187
    training r2: 0.078738
    Training loss: 1.48063
    training r2: 0.079506
    Training loss: 1.48121
    training r2: 0.079146
    Training loss: 1.4807
    training r2: 0.079464
    Training loss: 1.48034
    training r2: 0.079687
    Training loss: 1.48089
    training r2: 0.079343
    Training loss: 1.47942
    training r2: 0.080258
    Training loss: 1.47968
    training r2: 0.080095
    Training loss: 1.4787
    training r2: 0.080706
    Training loss: 1.47902
    training r2: 0.080510
    Training loss: 1.47954
    training r2: 0.080182
    Training loss: 1.4788
    training r2: 0.080642
    Training loss: 1.47792
    training r2: 0.081189
    Training loss: 1.47866
    training r2: 0.080730
    Training loss: 1.47789
    training r2: 0.081210
    Training loss: 1.47791
    training r2: 0.081197
    Training loss: 1.4785
    training r2: 0.080833
    test r2: 0.068026
    





```python
knn_content = GridSearchCV(KNeighborsRegressor(n_jobs=-1),{'n_neighbors':[5,10,30]})
knn_content.fit(X_train,y_train)
print('KNN:')
print("Training R^2:", knn_content.score(X_train,y_train))
print("Training MSE:",mean_squared_error(y_train,knn_content.predict(X_train)))
print("Test R^2:", knn_content.score(X_test,y_test))
print("Test MSE:",mean_squared_error(y_test,knn_content.predict(X_test)))
```


    KNN:
    Training R^2: 0.103936211431
    Training MSE: 1.44133436212
    Test R^2: 0.0135932748919
    Test MSE: 1.33215748925





```python
rf = GridSearchCV(RandomForestRegressor(n_estimators=256,
                                        n_jobs=-1,max_features="sqrt"),{"max_depth":[5,10,30]})
rf.fit(X_train,y_train)
print('Random Forest:')
print("Training R^2:", rf.score(X_train,y_train))
print("Training MSE:",mean_squared_error(y_train,rf.predict(X_train)))
print("Test R^2:", rf.score(X_test,y_test))
print("Test MSE:",mean_squared_error(y_test,rf.predict(X_test)))
```


    Random Forest:
    Training R^2: 0.782483160748
    Training MSE: 0.349879661193
    Test R^2: 0.216967316301
    Test MSE: 1.057497711



SVR is very slow. As a work around, we use LinearSVR, which is suitable for large-scale data.



```python
svr = LinearSVR()
svr.fit(X_train,y_train)
print("LinearSVR:")
print("Training R^2:", svr.score(X_train,y_train))
print("Training MSE:",mean_squared_error(y_train,svr.predict(X_train)))
print("Test R^2:", svr.score(X_test,y_test))
print("Test MSE:",mean_squared_error(y_test,svr.predict(X_test)))
```


    LinearSVR:
    Training R^2: -10.3161445972
    Training MSE: 18.2022175907
    Test R^2: -35.5750380933
    Test MSE: 49.3951528059



We'll choose Ridge, Lasso and RandomForestRegressor to be integrated into our ensembled model. Let's save those models to binary.



```python
import pickle
def save_model(file_name,model):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

save_model('ridge',ridge)
save_model('lasso',lasso)
save_model('rf',rf)
```

