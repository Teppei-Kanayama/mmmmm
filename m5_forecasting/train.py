from logging import getLogger

import gokart
import luigi

from m5_forecasting.data.calendar import PreprocessCalendar
from m5_forecasting.data.feature_engineering import MergeData, MakeFeature
from m5_forecasting.data.sales import PreprocessSales
from m5_forecasting.data.selling_price import PreprocessSellingPrice
from m5_forecasting.data.train_validation_split import TrainValidationSplit
from m5_forecasting.tasks.run_lgbm import TrainLGBM


logger = getLogger(__name__)


class Train(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    dark_magic: float = luigi.FloatParameter(default=None)

    def requires(self):
        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = PreprocessSellingPrice()
        sales_data_task = PreprocessSales(is_small=self.is_small)
        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task,
                                     sales_data_task=sales_data_task)
        feature_task = MakeFeature(merged_data_task=merged_data_task)
        model_task = TrainLGBM(feature_task=TrainValidationSplit(data_task=feature_task))
        return model_task

    def output(self):
        return self.input()
