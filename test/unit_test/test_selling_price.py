from datetime import date
import unittest
import pandas as pd
import numpy as np

from m5_forecasting.data.selling_price import PreprocessSellingPrice


class TestPreprocessSalse(unittest.TestCase):

    def setUp(self) -> None:
        self.selling_price = pd.DataFrame(dict(
            store_id=['store1'] * 6 + ['store2'] * 6,
            item_id=['item1'] * 3 + ['item2'] * 3 + ['item1'] * 3 + ['item2'] * 3,
            wm_yr_wk=['100', '101', '102'] * 4,
            sell_price=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

    def test_reshape_sales(self):
        actual = PreprocessSellingPrice._run(self.selling_price)
