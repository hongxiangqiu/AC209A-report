---
title: filter_features.py
---

``` python
import pandas as pd
import numpy as np

def filter_features(old_df):
    s_col = ['latitude_business', 'longitude_business', 'review_count_business',
     'stars_business', 'average_stars_user', 'compliment_cool_user', 'compliment_cute_user',
     'compliment_funny_user', 'compliment_hot_user', 'compliment_list_user', 'compliment_more_user',
     'compliment_note_user', 'compliment_photos_user', 'compliment_plain_user',
     'compliment_profile_user', 'compliment_writer_user', 'cool_user_user', 'fans_user',
     'funny_user', 'review_count_user', 'useful_user']
    new_df = old_df[['stars_review','business_id','user_id']+s_col].copy()
    year_col = set()
    for i in range(2004,2018):
        new_df['yelping_since_user_'+str(i)] = 0
        year_col.add(str(i))

    cats = set(['Dinner Theater', 'Buffets', 'Cafes', 'Asian Fusion',
            'Vegetarian', 'Mexican', 'Mediterranean', 'French', 'Burgers',
            'Breakfast & Brunch', 'Pizza', 'Fast Food', 'Bars', 'American (Traditional)',
            'Japanese', 'Sandwiches', 'Steakhouses', 'Thai', 'Sushi Bars', 'Food', 'Italian',
            'Seafood', 'Barbeque', 'Nightlife', 'Salad', 'Restaurants', 'American (New)',
            'Chinese', 'Vietnamese', 'Coffee & Tea'])
    for x in cats:
        new_df[x] = 0

    #attrib
    new_df['Alcohol'] = 1
    new_df['Alcohol_full_bar'] = 0
    new_df['hasTV'] = 1
    new_df['noiseLevel_quiet'] = 0
    new_df['noiseLevel_loud'] = 0
    new_df['noiseLevel_veryloud'] = 0
    new_df['RestaurantsAttire'] = 1
    new_df['RestaurantsAttire_casual']=0
    new_df['RestaurantsAttire_dressy']=0
    new_df['RestaurantsAttire_formal']=0
    new_df['WiFi_no'] = 0
    new_df['WiFi_free'] = 0
    new_df['WiFi_paid'] = 0
    no_col = ['false','no','none']
    col_id_dict = {val:rid for rid,val in enumerate(list(new_df.columns))}
    old_id_dict = {val:rid for rid,val in enumerate(list(old_df.columns))}
    for i in range(len(new_df)):
        year = old_df.iat[i,old_id_dict['yelping_since_user']][:4]
        if year in year_col:
            new_df.iat[i,col_id_dict['yelping_since_user_'+str(year)]] = 1
        cat_b = set(eval(old_df.iat[i,old_id_dict['categories_business']]))
        for x in cat_b & cats:
            new_df.iat[i,col_id_dict[x]] = 1
        adict = eval(old_df.iat[i,old_id_dict['attributes_business']])
        if 'Alcohol' not in adict or adict['Alcohol'] == False or str(adict['Alcohol']).lower() in no_col:
            new_df.iat[i,col_id_dict['Alcohol']] = 0
        elif str(adict['Alcohol']).lower() == 'full_bar':
            new_df.iat[i,col_id_dict['Alcohol_full_bar']] = 1
        if 'HasTV' not in adict or adict['HasTV'] == False or str(adict['HasTV']).lower() in no_col:
            new_df.iat[i,col_id_dict['hasTV']] = 0
        if 'NoiseLevel' not in adict or adict['NoiseLevel'] == False or str(adict['NoiseLevel']).lower() in no_col+['quiet']:
            new_df.iat[i,col_id_dict['noiseLevel_quiet']] = 1
        elif adict['NoiseLevel'] == True or str(adict['NoiseLevel']).lower() in ['loud','yes','true']:
            new_df.iat[i,col_id_dict['noiseLevel_loud']] = 1
        elif str(adict['NoiseLevel']).lower().replace(' ','').replace('_','')=='veryloud':
            new_df.iat[i,col_id_dict['noiseLevel_veryloud']] = 1
        if 'RestaurantsAttire' not in adict or adict['RestaurantsAttire'] == False or str(adict['RestaurantsAttire']).lower() in no_col:
            new_df.iat[i,col_id_dict['RestaurantsAttire']] = 0
        elif str(adict['RestaurantsAttire']).lower() == 'casual':
            new_df.iat[i,col_id_dict['RestaurantsAttire_casual']] = 1
        elif str(adict['RestaurantsAttire']).lower() == 'dressy':
            new_df.iat[i,col_id_dict['RestaurantsAttire_dressy']] = 1
        elif str(adict['RestaurantsAttire']).lower() == 'formal':
            new_df.iat[i,col_id_dict['RestaurantsAttire_formal']] = 1
        if 'WiFi' not in adict or adict['WiFi'] == False or str(adict['WiFi']).lower() in no_col:
            new_df.iat[i,col_id_dict['WiFi_no']] = 1
        elif str(adict['WiFi']).lower() == 'free':
            new_df.iat[i,col_id_dict['WiFi_free']] = 1
        elif str(adict['WiFi']).lower() == 'paid':
            new_df.iat[i,col_id_dict['WiFi_paid']] = 1
    return new_df

savep = str(input("save_to: "))
old = pd.read_csv(str(input("old_df_name: ")),memory_map=True)
new = filter_features(old)
new.to_csv(savep, index=False)
```
