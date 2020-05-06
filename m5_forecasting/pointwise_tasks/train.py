from logging import getLogger

import gokart
import luigi
import pandas as pd

from m5_forecasting.data.calendar import PreprocessCalendar
from m5_forecasting.data.feature_engineering import MergeData, MakeFeature
from m5_forecasting.data.sales import PreprocessSales, MekeSalesFeature
from m5_forecasting.data.selling_price import PreprocessSellingPrice
from m5_forecasting.data.train_validation_split import TrainValidationSplit
from m5_forecasting.pointwise_tasks.run_lgbm import TrainPointwiseLGBM


logger = getLogger(__name__)


class DummyPredictTask(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def run(self):
        self.dump(pd.DataFrame(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "demand"]))


class TrainPointwiseModel(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    def requires(self):
        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = PreprocessSellingPrice()
        sales_data_task = MekeSalesFeature(sales_data_task=PreprocessSales(is_small=self.is_small),
                                           predicted_sales_data_task=DummyPredictTask())
        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task,
                                     sales_data_task=sales_data_task)
        feature_task = MakeFeature(merged_data_task=merged_data_task)
        model_task = TrainPointwiseLGBM(feature_task=TrainValidationSplit(data_task=feature_task))
        return model_task

    def output(self):
        return self.input()


 # python main.py m5-forecasting.TrainPointwiseModel --local-scheduler
 # DATA_SIZE=small python main.py m5-forecasting.TrainPointwiseModel --local-scheduler