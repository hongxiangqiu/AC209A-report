---
title: Data Sampling
notebook: data-sampling.ipynb
nav_include: 4
---

We first join the three json files downloaded from Yelp. The joined dataset is saved as `data.csv`.

We reset the column name to make the meaning of each column clear.



```python
df = pd.read_csv("data.csv")
col_names = ['business_id', 
'cool_review', 
'date_review', 
'funny_review',
'review_id', 
'stars_review', 
'useful_review', 
'user_id',
'attributes_business', 
'categories_business',
'city_business',
'hours_business',
'is_open_business',
'latitude_business',
'longitude_business',
'name_business',
'neighborhood_business',
'postal_code_business',
'review_count_business',
'stars_business',
'state_business',
'average_stars_user',
'compliment_cool_user',
'compliment_cute_user',
'compliment_funny_user',
'compliment_hot_user',
'compliment_list_user',
'compliment_more_user',
'compliment_note_user',
'compliment_photos_user',
'compliment_plain_user', 
'compliment_profile_user', 
'compliment_writer_user',
'cool_user_user', 
'elite_user', 
'fans_user', 
'friends_user', 
'funny_user', 
'name_user',
'review_count_user',
'useful_user',
'yelping_since_user']
del df['Unnamed: 0']
df=df.rename(columns=dict(zip(list(df.columns), col_names)))
```


Now we subsample our data: we remove all business with less than 16 reviews, all closed business, and users with less than 6 reviews. Those data is not good for the training purpose (especially for collaborative filtering based models).



```python
df = df[df.is_open_business == 1]
df_by_business_id = df.groupby('business_id')
business_counts = df_by_business_id.count()['review_id']
threshold_business = 16
valid_business = set(business_counts.index[business_counts>=threshold_business])
b_selector = df.business_id.map(lambda b:b in valid_business)
filtered = df[b_selector]
```




```python
df_by_user_id = filtered.groupby('user_id')
user_counts = df_by_user_id.count()['review_id']
threshold_user = 6
valid_user = set(user_counts.index[user_counts>=threshold_user])
u_selector = filtered.user_id.map(lambda b:b in valid_user)
filtered = filtered[u_selector]
filtered.to_csv("filtered.csv",index=False)
```




```python
filtered_ind = df.index.isin(set(filtered.index))
filtered_out = df[~filtered_ind]
filtered_out.to_csv("filtered_out.csv",index=False) # remaining
```


Now we partition our subsample into `training`, `meta_training`, and `test` set



```python
df = pd.read_csv("filtered.csv")
```


Make sure in `training` set, each user appears 4 times (stratified random sampling)



```python
train1 = df.groupby('user_id', group_keys=False).apply(lambda x: x.sample(4))
```




```python
len(train1)
```





    331784





```python
df_ex_train1 = df[~df.index.isin(set(train1.index))]
train2 = df_ex_train1.groupby('user_id', group_keys=False).apply(lambda x: x.sample(1))
len(train2)
```





    82946





```python
df_ex_train2 = df_ex_train1[~df_ex_train1.index.isin(set(train2.index))]
len(df_ex_train2)
```





    889363





```python
train1.to_csv("subsample_1.csv", index=False) # train
train2.to_csv("subsample_2.csv", index=False) # meta_train
df_ex_train2.to_csv("subsample_3.csv", index=False) # test
```

After sampling the data, we perform predictor selection on
`subsample_1.csv` and then extract those predictors on all
those sets using a [Python script](src/filter_features.html).
For details, see [model exploration page](model-exploration.html).
