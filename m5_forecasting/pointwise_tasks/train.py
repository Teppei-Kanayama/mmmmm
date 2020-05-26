from logging import getLogger

import gokart
import luigi
import pandas as pd

from m5_forecasting.adversarial_validation.adversarial_validation import FilterByAdversarialValidation
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
    train_to_date: int = luigi.IntParameter()
    filter_by_adversarial_validation: bool = luigi.BoolParameter()  # TODO: adversarial validation shold be set by setting file.

    def requires(self):
        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = PreprocessSellingPrice()
        sales_data_task = MekeSalesFeature(sales_data_task=PreprocessSales(is_small=self.is_small),
                                           predicted_sales_data_task=DummyPredictTask())
        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task,
                                     sales_data_task=sales_data_task)
        if self.filter_by_adversarial_validation:
            feature_task = FilterByAdversarialValidation(feature_task=MakeFeature(merged_data_task=merged_data_task), is_small=self.is_small)
        else:
            feature_task = MakeFeature(merged_data_task=merged_data_task)

        model_task = TrainPointwiseLGBM(feature_task=TrainValidationSplit(data_task=feature_task, train_to_date=self.train_to_date))
        return model_task

    def output(self):
        return self.input()
