from datetime import date
import unittest
import pandas as pd
import numpy as np

from m5_forecasting.data.sales import PreprocessSales


class TestPreprocessSalse(unittest.TestCase):

    def setUp(self) -> None:
        self.sales = pd.DataFrame(dict(
            id=['id1', 'id2', 'id3'],
            item_id=['HOBBIES_1_001', 'HOBBIES_1_002', 'HOBBIES_1_003'],
            dept_id=['HOBBIES_1', 'HOBBIES_1', 'HOBBIES_1'],
            cat_id=['HOBBIES', 'HOBBIES', 'HOBBIES'],
            store_id=['CA_1'] * 3,
            state_id=['CA'] * 3,
            d_1=[0, 1, 2],
            d_2=[3, 4, 5],
            d_3=[6, 7, 8],
            ))

    def test_run(self):
        actual = PreprocessSales._reshape_sales(self.sales, drop_d=0, is_small=False)
       
# id        item_id        dept_id    cat_id      store_id   state_id  d  demand
# 0    id1  HOBBIES_1_001  HOBBIES_1  HOBBIES     CA_1       CA     1     0.0
# 1    id2  HOBBIES_1_002  HOBBIES_1  HOBBIES     CA_1       CA     1     1.0
# 2    id3  HOBBIES_1_003  HOBBIES_1  HOBBIES     CA_1       CA     1     2.0
# 3    id1  HOBBIES_1_001  HOBBIES_1  HOBBIES     CA_1       CA     2     3.0
# 4    id2  HOBBIES_1_002  HOBBIES_1  HOBBIES     CA_1       CA     2     4.0
# ..   ...            ...        ...      ...      ...      ...   ...     ...
# 172  id2  HOBBIES_1_002  HOBBIES_1  HOBBIES     CA_1       CA  1968     NaN
# 173  id3  HOBBIES_1_003  HOBBIES_1  HOBBIES     CA_1       CA  1968     NaN
# 174  id1  HOBBIES_1_001  HOBBIES_1  HOBBIES     CA_1       CA  1969     NaN
# 175  id2  HOBBIES_1_002  HOBBIES_1  HOBBIES     CA_1       CA  1969     NaN
# 176  id3  HOBBIES_1_003  HOBBIES_1  HOBBIES     CA_1       CA  1969     NaN
