from logging import getLogger
from typing import List

import gokart
import luigi
import pandas as pd
import numpy as np
from lightgbm import Booster

from m5_forecasting.data.calendar import PreprocessCalendar
from m5_forecasting.data.feature_engineering import MergeData, MakeFeature
from m5_forecasting.data.sales import PreprocessSales, MekeSalesFeature
from m5_forecasting.data.selling_price import PreprocessSellingPrice
from m5_forecasting.data.load import LoadInputData
from m5_forecasting.train import Train

logger = getLogger(__name__)


class DummyTask(gokart.TaskOnKart):

    is_dummy = luigi.BoolParameter(default=True)


class Predict(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    predict_from_date: int = luigi.IntParameter()
    predict_to_date: int = luigi.IntParameter()
    latest_prediction_task: gokart.TaskOnKart = gokart.TaskInstanceParameter()
    trained_model_task: gokart.TaskOnKart = gokart.TaskInstanceParameter()

    def requires(self):
        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = PreprocessSellingPrice()

        sales_data_task = PreprocessSales(is_small=self.is_small) if 'is_dummy' in self.latest_prediction_task.param_kwargs.keys() \
            else self.latest_prediction_task
        sales_feature_task = MekeSalesFeature(sales_data_task=sales_data_task)

        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task,
                                     sales_data_task=sales_feature_task)
        feature_task = MakeFeature(merged_data_task=merged_data_task)

        return dict(model=self.trained_model_task, sales=sales_data_task, feature=feature_task)

    def run(self):
        model = self.load('model')['model']
        feature_columns = self.load('model')['feature_columns']
        feature = self.load_data_frame('feature')
        sales = self.load_data_frame('sales')

        output = self._run(model, feature_columns, feature, sales, self.predict_from_date, self.predict_to_date)
        self.dump(output)

    @staticmethod
    def _run(model: Booster, feature_columns: List[str], feature: pd.DataFrame, sales, predict_from_date, predict_to_date) -> pd.DataFrame:
        test = feature[(predict_from_date <= feature['d']) & (feature['d'] < predict_to_date)]  # 1914)
        pred = model.predict(test[feature_columns])
        sales.loc[sales[(sales['id'].isin(test['id'])) & (sales['d'].isin(test['d']))].index, 'demand'] = pred
        return sales


class PredictAll(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def output(self):
        return self.make_target('submission.csv')

    def requires(self):
        model_task = Train()
        predict0 = Predict(predict_from_date=1914, predict_to_date=1914+7, trained_model_task=model_task, latest_prediction=DummyTask())
        predict1 = Predict(predict_from_date=1914+7, predict_to_date=1914+14, trained_model_task=model_task, latest_prediction=predict0)
        predict2 = Predict(predict_from_date=1914+14, predict_to_date=1914+21, trained_model_task=model_task, latest_prediction=predict1)

        sample_submission_data_task = LoadInputData(filename='sample_submission.csv')
        return dict(predict=predict2, sample_submission=sample_submission_data_task)

    def run(self):
        test = self.load_data_frame('predict')
        sample_submission = self.load_data_frame('sample_submission')
        output = self._run(test, sample_submission)
        self.dump(output)

    @staticmethod
    def _run(test, sample_submission):
        test = test.fillna(-1)  # evaluation scores
        test = test.assign(id=test.id + "_" + np.where(test.d <= 1941, "validation", "evaluation"),
                           F="F" + (test.d - 1913 - 28 * (test.d > 1941)).astype("str"))
        submission = test.pivot(index="id", columns="F", values="demand").reset_index()[sample_submission.columns]
        return submission


# python main.py m5-forecasting.Predict --local-scheduler
# DATA_SIZE=full python main.py m5-forecasting.Predict --local-scheduler
# DATA_SIZE=small python main.py m5-forecasting.Predict --local-scheduler

# DATA_SIZE=small python main.py m5-forecasting.PredictAll --local-scheduler
