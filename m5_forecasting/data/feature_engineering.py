from logging import getLogger

import gc
import gokart
import luigi
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.data.sales import PreprocessSales

logger = getLogger(__name__)


class MergeData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    calendar_data_task = gokart.TaskInstanceParameter()
    selling_price_data_task = gokart.TaskInstanceParameter()
    sales_data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return dict(calendar=self.calendar_data_task, selling_price=self.selling_price_data_task,
                    sales=self.sales_data_task)

    def run(self):
        calendar = self.load_data_frame('calendar')
        selling_price = self.load_data_frame('selling_price')
        sales = self.load_data_frame('sales')
        output = self._run(calendar, selling_price, sales)
        self.dump(output)

    @staticmethod
    def _run(calendar, selling_price, sales):
        sales = sales.merge(calendar, how="left", on="d")
        gc.collect()

        sales = sales.merge(selling_price, how="left", on=["store_id", "item_id", "wm_yr_wk"])
        sales.drop(["wm_yr_wk"], axis=1, inplace=True)
        gc.collect()
        del selling_price

        return sales


class GetSoldOutDate(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()
    window_size: int = luigi.IntParameter(default=60)
    popular_item_threshold = luigi.IntParameter(default=10)

    def requires(self):
        return dict(sales=PreprocessSales(is_small=self.is_small), calendar=LoadInputData(filename='calendar.csv'),
                    sell_prices=LoadInputData(filename='sell_prices.csv'))

    def run(self):
        sales = self.load_data_frame('sales')
        calendar = self.load_data_frame('calendar')
        sell_prices = self.load_data_frame('sell_prices')
        output = self._run(sales, calendar, sell_prices, self.window_size, self.popular_item_threshold)
        self.dump(output)

    @staticmethod
    def _run(sales, calendar, sell_prices, window_size, popular_item_threshold):
        # selling priceが存在しないitemに対し、擬似的にdemand+=1する
        calendar = calendar.assign(d=calendar['d'].str[2:].astype(int))
        sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id']
        sell_prices = pd.merge(sell_prices[['id', 'wm_yr_wk', 'sell_price']], calendar[['d', 'wm_yr_wk']], on='wm_yr_wk', how='left')
        sales = pd.merge(sales, sell_prices[['id', 'd', 'sell_price']], on=['id', 'd'], how='left').fillna(0)
        sales['is_zero_sell_price'] = sales['sell_price'] == 0
        sales['demand'] = sales['demand'] + sales['is_zero_sell_price']

        # 売り切れ期間を見つける
        sales['rolling_sum_backward'] = sales.groupby('id')['demand'].transform(lambda x: x.shift(0).rolling(window_size).sum())
        sales['rolling_sum_forward'] = sales.groupby('id')['demand'].transform(lambda x: x.rolling(window_size).sum().shift(1-window_size))
        sold_out_df = sales[(sales['rolling_sum_backward'] == 0) | (sales['rolling_sum_forward'] == 0)]

        # 人気itemのみに絞る（偶然0だった場合と区別するため）
        sales_agg = sales.groupby('id', as_index=False).agg({'demand': 'mean'})
        sales_agg['is_popular'] = sales_agg['demand'] > popular_item_threshold
        sold_out_df = pd.merge(sold_out_df, sales_agg[['id', 'is_popular']], on='id', how='left')
        sold_out_df = sold_out_df[sold_out_df['is_popular']]

        return sold_out_df[['id', 'd']]


class MakeFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    delete_sold_out_date = luigi.BoolParameter(default=False)
    merged_data_task = gokart.TaskInstanceParameter()
    is_small: bool = luigi.BoolParameter()

    def requires(self):
        return dict(data=self.merged_data_task, sold_out_date=GetSoldOutDate(is_small=self.is_small))

    def run(self):
        data = self.load_data_frame('data')
        sold_out_date = self.load_data_frame('sold_out_date')
        self.dump(self._run(data, sold_out_date, self.delete_sold_out_date))

    @classmethod
    def _run(cls, data, sold_out_date, delete_sold_out_date):
        data = cls._delete_sold_out_date(data, sold_out_date) if delete_sold_out_date else data
        data = cls._label_encode(data)
        return data

    @staticmethod
    def _delete_sold_out_date(data, sold_out_date):
        sold_out_date['did'] = sold_out_date['d'].astype(str) + sold_out_date['id']
        data['did'] = data['d'].astype(str) + data['id']
        # sold_out_date = pd.merge(data, sold_out_date, on=['id', 'd'], how='left').fillna(False)
        # return sold_out_date[~sold_out_date['drop']]
        data = data[~data['did'].isin(sold_out_date['did'])]
        return data.drop('did', axis=1)

    @staticmethod
    def _label_encode(data):
        for i, v in tqdm(enumerate(["item_id", "dept_id", "store_id", "cat_id", "state_id"])):
            data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]]).astype("int16") + 1
        return data
