import unittest
import pandas as pd

from m5_forecasting.data.feature_engineering import GetFirstSoldDate


class TestGetFirstSoldDate(unittest.TestCase):

    def setUp(self) -> None:
        self.sales = pd.DataFrame(dict(
            id=['id1', 'id2', 'id3'],
            item_id=['HOBBIES_1_001', 'HOBBIES_1_002', 'HOBBIES_1_003'],
            dept_id=['HOBBIES_1', 'HOBBIES_1', 'HOBBIES_1'],
            cat_id=['HOBBIES', 'HOBBIES', 'HOBBIES'],
            store_id=['CA_1'] * 3,
            state_id=['CA'] * 3,
            d_1=[0, 1, 2],
            d_2=[0, 0, 5],
            d_3=[6, 7, 8],
            ))

    def test_run(self):
        actual = GetFirstSoldDate._run(self.sales)
