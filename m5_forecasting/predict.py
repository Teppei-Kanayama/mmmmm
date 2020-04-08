from logging import getLogger
from typing import List

import gokart
import luigi
import pandas as pd
import numpy as np
from lightgbm import Booster

from m5_forecasting.data.calendar import PreprocessCalendar
from m5_forecasting.data.feature_engineering import MergeData, MakeFeature
from m5_forecasting.data.sales import PreprocessSales
from m5_forecasting.data.selling_price import PreprocessSellingPrice
from m5_forecasting.data.train_validation_split import TrainValidationSplit
from m5_forecasting.tasks.run_lgbm import TrainLGBM

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class Predict(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    def output(self):
        return self.make_target('submission.csv', use_unique_id=self.is_small)

    def requires(self):
        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = PreprocessSellingPrice()
        sales_data_task = PreprocessSales(is_small=self.is_small)
        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task,
                                     sales_data_task=sales_data_task)
        feature_task = MakeFeature(merged_data_task=merged_data_task)
        model_task = TrainLGBM(feature_task=TrainValidationSplit(data_task=feature_task))
        sample_submission_data_task = LoadInputData(filename='sample_submission.csv')
        return dict(model=model_task, sample_submission=sample_submission_data_task, feature=feature_task)

    def run(self):
        model = self.load('model')['model']
        feature_columns = self.load('model')['feature_columns']
        sample_submission = self.load('sample_submission')
        feature = self.load_data_frame('feature')
        output = self._run(model, feature_columns, feature, sample_submission)
        self.dump(output)

    @staticmethod
    def _run(model: Booster, feature_columns: List[str], feature: pd.DataFrame, sample_submission: pd.DataFrame, dark_magic=False) -> pd.DataFrame:
        test = feature[(feature['d'] >= 1914)]
        pred = model.predict(test[feature_columns])
        if dark_magic:
            pred = pred / pred[test["id"].str.endswith("validation")].mean() * 1.447147
        test['demand'] = pred
        test = test.assign(id=test.id + "_" + np.where(test.d <= 1941, "validation", "evaluation"),
                           F="F" + (test.d - 1913 - 28 * (test.d > 1941)).astype("str"))
        submission = test.pivot(index="id", columns="F", values="demand").reset_index()[sample_submission.columns]
        return submission

# python main.py m5-forecasting.Predict --local-scheduler
# DATA_SIZE=full python main.py m5-forecasting.Predict --local-scheduler
# DATA_SIZE=small python main.py m5-forecasting.Predict --local-scheduler
