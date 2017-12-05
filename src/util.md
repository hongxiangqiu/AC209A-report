---
title: util.py
---

``` python
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix


def to_matrix(data):
    """

    Parameters
    ----------
    data: pd.DataFrame
        input data, must have 'user_id', 'business_id' and 'stars' columns

    Returns
    -------
    (dok_matrix, dict[int, int], dict[int, int])
        sparse matrix (DOK) generated from input data, mapping from user_id to matrix index, mapping from business_id
        to matrix index
    """

    business_id_map = {k: v for v, k in enumerate(np.unique(data.business_id))}
    user_id_map = {k: v for v, k in enumerate(np.unique(data.user_id))}
    result_matrix = dok_matrix((len(user_id_map), len(business_id_map)))
    for i, row in data.iterrows():
        ui = user_id_map[row['user_id']]
        bi = business_id_map[row['business_id']]
        result_matrix[ui, bi] = row['stars']
    return result_matrix, user_id_map, business_id_map
```
