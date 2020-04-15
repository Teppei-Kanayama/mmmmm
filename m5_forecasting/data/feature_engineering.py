from logging import getLogger

import gc
import gokart
import luigi
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

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


class MakeFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    merged_data_task = gokart.TaskInstanceParameter()
    is_small: bool = luigi.BoolParameter()

    def requires(self):
        return dict(data=self.merged_data_task)

    def run(self):
        data = self.load_data_frame('data')
        self.dump(self._run(data))

    @classmethod
    def _run(cls, data):
        data = cls._label_encode(data)
        return data

    @staticmethod
    def _label_encode(data):
        for i, v in tqdm(enumerate(["item_id", "dept_id", "store_id", "cat_id", "state_id"])):
            data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]]).astype("int16") + 1
        return data
