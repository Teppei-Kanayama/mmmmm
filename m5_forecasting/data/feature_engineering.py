from logging import getLogger

import gc
import gokart
import luigi
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

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
    popular_item_threshold = luigi.IntParameter(default=1)

    def requires(self):
        return PreprocessSales(is_small=self.is_small, drop_old_data_days=0)

    def run(self):
        sales = self.load_data_frame()
        output = self._run(sales, self.window_size, self.popular_item_threshold)
        self.dump(output)

    @staticmethod
    def _run(sales, window_size, popular_item_threshold):
        sales['rolling_sum_backward'] = sales.groupby('id')['demand'].transform(lambda x: x.shift(0).rolling(window_size).mean())
        sales['rolling_sum_forward'] = sales.groupby('id')['demand'].transform(lambda x: x.rolling(window_size).mean().shift(1-window_size))
        sold_out_df = sales[(sales['rolling_sum_backward'] == 0) | (sales['rolling_sum_forward'] == 0)]

        sales_agg = sales.groupby('id', as_index=False).agg({'demand': 'mean'})
        sales_agg['is_popular'] = sales_agg['demand'] > popular_item_threshold
        sold_out_df = pd.merge(sold_out_df, sales_agg[['id', 'is_popular']], on='id', how='left')
        sold_out_df = sold_out_df[sold_out_df['is_popular']]
        return sold_out_df[['id', 'd']]


class MakeFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    delete_sold_out_date = luigi.BoolParameter(default=True)
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
        sold_out_date['drop'] = True
        sold_out_date = pd.merge(data, sold_out_date, on=['id', 'd'], how='left').fillna(False)
        return sold_out_date[~sold_out_date['drop']]

    @staticmethod
    def _label_encode(data):
        for i, v in tqdm(enumerate(["item_id", "dept_id", "store_id", "cat_id", "state_id"])):
            data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]]).astype("int16") + 1
        return data
