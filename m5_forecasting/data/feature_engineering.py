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
        # sales.drop(["wm_yr_wk"], axis=1, inplace=True)  # 結局落とすの？
        gc.collect()
        del selling_price

        return sales


class GetFirstSoldDate(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    def requires(self):
        return PreprocessSales(is_small=self.is_small, drop_old_data_days=0)

    def run(self):
        sales = self.load_data_frame()
        output = self._run(sales)
        self.dump(output)

    @staticmethod
    def _run(sales):
        sales = sales[~sales['demand'].isna()]
        sales = sales[sales['demand'] != 0]  # 非ゼロの中で最小の日付
        return sales.groupby('id', as_index=False).agg({'d': 'min'}).rename(columns={'d': 'first_sold_date'})


class MakeFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    delete_first_sold_date = luigi.BoolParameter()
    merged_data_task = gokart.TaskInstanceParameter()
    is_small: bool = luigi.BoolParameter()

    def requires(self):
        return dict(data=self.merged_data_task, first_sold_date=GetFirstSoldDate(is_small=self.is_small))

    def run(self):
        data = self.load_data_frame('data')
        first_sold_date = self.load_data_frame('first_sold_date')
        self.dump(self._run(data, first_sold_date, self.delete_first_sold_date))

    @classmethod
    def _run(cls, data, first_sold_date, delete_first_sold_date):
        data = cls._delete_before_first_sold_date(data, first_sold_date) if delete_first_sold_date else data
        data = cls._label_encode(data)
        return data

    @staticmethod
    def _delete_before_first_sold_date(data, first_sold_date):
        data = pd.merge(data, first_sold_date, on='id', how='left').fillna(0)
        data = data[data['d'] >= data['first_sold_date']]
        data = data.drop('first_sold_date', axis=1)
        return data

    @staticmethod
    def _label_encode(data):
        for i, v in tqdm(enumerate(["item_id", "dept_id", "store_id", "cat_id", "state_id"])):
            data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]]).astype("int16") + 1
        return data
